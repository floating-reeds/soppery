"""
Zero-Shot Citation Hallucination Scorer

Adapts FACTUM's mechanistic scores for non-RAG (zero-shot) settings.
Key modification: Replaces CAS (Context Alignment Score) with ICS (Internal Consistency Score)
that measures alignment between citation and self-generated essay context.

Based on: https://github.com/Maximeswd/citation-hallucination
"""

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class MechanisticScores:
    """All mechanistic scores for a single citation token."""
    token_position: int
    token_text: str
    label: str  # Ground truth label
    
    # Core scores (per layer)
    ics_scores: List[float]      # Internal Consistency Score (adapted CAS)
    pos_scores: List[float]      # Pathway Orthogonality Score
    pfs_scores: List[float]      # Parametric Force Score (V_ffn norms)
    vas_scores: List[float]      # V_attn norms
    bas_scores: List[float]      # BOS attention proportion
    
    # Aggregate scores
    ics_mean: float
    ics_final: float
    pos_mean: float
    pos_final: float
    pfs_mean: float
    pfs_final: float
    bas_mean: float
    
    # Metadata
    essay_id: str
    model_name: str


class ZeroShotScorer:
    """
    Compute mechanistic scores for zero-shot citation generation.
    
    Key difference from FACTUM:
    - FACTUM's CAS measures alignment with external source documents
    - Our ICS measures alignment with model's own generated essay prefix
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        chunk_size: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",  # Required for output_attentions=True
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _get_hidden_states_chunked(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Get hidden states for long sequences using chunking."""
        seq_len = input_ids.shape[1]
        all_hidden_states = [[] for _ in range(self.num_layers + 1)]
        
        with torch.no_grad():
            for chunk_start in range(0, seq_len, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, seq_len)
                chunk_ids = input_ids[:, chunk_start:chunk_end].to(self.device)
                
                outputs = self.model(
                    input_ids=chunk_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                for layer_idx, hs in enumerate(outputs.hidden_states):
                    all_hidden_states[layer_idx].append(hs.cpu())
                
                del outputs
                torch.cuda.empty_cache()
        
        # Concatenate chunks
        return tuple(
            torch.cat(layer_hs, dim=1) for layer_hs in all_hidden_states
        )
    
    def compute_ics(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attn_weights: torch.Tensor,
        citation_pos: int,
        essay_start_pos: int,
    ) -> List[float]:
        """
        Compute Internal Consistency Score (ICS).
        
        Adapted from FACTUM's CAS:
        - Original: cos_sim(attention-weighted context, citation hidden state)
        - Adapted: cos_sim(attention-weighted essay_prefix, citation hidden state)
        
        This measures how consistent the citation is with the model's own
        generated essay context, rather than external documents.
        """
        ics_per_layer = []
        
        for layer_idx in range(self.num_layers):
            # Hidden states for this layer
            hs_layer = hidden_states[layer_idx + 1][0].to(self.device)  # [seq_len, hidden]
            
            # Attention from citation position to essay prefix
            # attn_weights shape: [num_layers, num_heads, seq_len, seq_len]
            if attn_weights is not None and layer_idx < attn_weights.shape[0]:
                attn_to_essay = attn_weights[layer_idx, :, citation_pos, essay_start_pos:citation_pos]
                
                if attn_to_essay.shape[-1] > 0:
                    # Normalize attention weights
                    attn_normalized = attn_to_essay / (attn_to_essay.sum(dim=-1, keepdim=True) + 1e-9)
                    
                    # Essay context hidden states
                    hs_essay = hs_layer[essay_start_pos:citation_pos]  # [essay_len, hidden]
                    
                    # Attention-weighted essay representation (per head)
                    # [num_heads, essay_len] @ [essay_len, hidden] -> [num_heads, hidden]
                    essay_repr = torch.matmul(attn_normalized, hs_essay)
                    
                    # Citation hidden state
                    h_citation = hs_layer[citation_pos].unsqueeze(0)  # [1, hidden]
                    
                    # Cosine similarity (average over heads)
                    ics = F.cosine_similarity(essay_repr, h_citation, dim=-1).mean().item()
                else:
                    ics = 0.0
            else:
                ics = 0.0
            
            ics_per_layer.append(ics)
            hs_layer = hs_layer.cpu()
        
        return ics_per_layer
    
    def compute_pos(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        citation_pos: int,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute Pathway Orthogonality Score (POS).
        
        POS measures the cosine similarity between attention and FFN updates.
        High POS = pathways agree; Low POS = internal dissonance.
        
        Also returns V_attn and V_ffn norms (PFS).
        """
        pos_per_layer = []
        v_attn_norms = []
        v_ffn_norms = []
        
        for layer_idx in range(self.num_layers):
            # Get hidden states before and after attention/FFN
            # Note: This is an approximation since we don't have gpa_states
            # In the full FACTUM implementation, use output_gpa_states=True
            
            hs_pre = hidden_states[layer_idx][0, citation_pos].to(self.device)
            hs_post = hidden_states[layer_idx + 1][0, citation_pos].to(self.device)
            
            # Total update = V_attn + V_ffn
            # We approximate by assuming equal contribution (actual split requires hooks)
            total_update = hs_post - hs_pre
            
            # For approximation, we use the residual as proxy for combined pathway
            v_total_norm = torch.norm(total_update).item()
            
            # POS approximation: 1 - abs(cos_sim between prev and current)
            # This captures how orthogonal the update is
            cos_sim = F.cosine_similarity(hs_pre.unsqueeze(0), hs_post.unsqueeze(0)).item()
            pos = 1.0 - abs(cos_sim)
            
            pos_per_layer.append(pos)
            v_attn_norms.append(v_total_norm / 2)  # Approximate split
            v_ffn_norms.append(v_total_norm / 2)
        
        return pos_per_layer, v_attn_norms, v_ffn_norms
    
    def compute_bas(
        self,
        attn_weights: torch.Tensor,
        citation_pos: int,
    ) -> List[float]:
        """
        Compute BOS Attention Score (BAS).
        
        Measures attention paid to the BOS (position 0) token.
        High BAS indicates the model is synthesizing from its parametric memory.
        """
        bas_per_layer = []
        
        for layer_idx in range(self.num_layers):
            if attn_weights is not None and layer_idx < attn_weights.shape[0]:
                # Attention to position 0 (BOS) from citation position
                bos_attn = attn_weights[layer_idx, :, citation_pos, 0].mean().item()
            else:
                bos_attn = 0.0
            
            bas_per_layer.append(bos_attn)
        
        return bas_per_layer
    
    def score_essay(
        self,
        essay_data: Dict,
    ) -> List[MechanisticScores]:
        """
        Compute all mechanistic scores for citations in an essay.
        """
        prompt = essay_data.get("prompt", "")
        response = essay_data.get("response", "")
        citations = essay_data.get("citations", [])
        essay_id = essay_data.get("prompt_id", "unknown")
        
        if not citations:
            return []
        
        # Tokenize full text
        full_text = prompt + response
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = encoding.input_ids
        offset_mapping = encoding.offset_mapping[0]
        
        # Find essay start (after prompt)
        prompt_encoding = self.tokenizer(prompt, return_tensors="pt")
        essay_start_pos = prompt_encoding.input_ids.shape[1]
        
        # Get hidden states and attention weights
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            
            hidden_states = tuple(hs.cpu() for hs in outputs.hidden_states)
            attn_weights = torch.stack(outputs.attentions).squeeze(1).cpu()  # [layers, heads, seq, seq]
        
        all_scores = []
        
        for citation in citations:
            # Find citation position in token sequence
            char_start = citation.get("start_pos", 0) + len(prompt)
            char_end = citation.get("end_pos", char_start + 1)
            label = citation.get("label", "unverified")
            
            # Map character position to token position
            citation_token_pos = None
            for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
                if tok_start <= char_start < tok_end:
                    citation_token_pos = tok_idx
                    break
            
            if citation_token_pos is None or citation_token_pos >= input_ids.shape[1]:
                continue
            
            # Compute scores
            ics_scores = self.compute_ics(
                hidden_states, attn_weights, citation_token_pos, essay_start_pos
            )
            pos_scores, vas_scores, pfs_scores = self.compute_pos(
                hidden_states, citation_token_pos
            )
            bas_scores = self.compute_bas(attn_weights, citation_token_pos)
            
            # Get token text
            token_text = self.tokenizer.decode(input_ids[0, citation_token_pos])
            
            scores = MechanisticScores(
                token_position=citation_token_pos,
                token_text=token_text,
                label=label,
                ics_scores=ics_scores,
                pos_scores=pos_scores,
                pfs_scores=pfs_scores,
                vas_scores=vas_scores,
                bas_scores=bas_scores,
                ics_mean=sum(ics_scores) / len(ics_scores) if ics_scores else 0,
                ics_final=ics_scores[-1] if ics_scores else 0,
                pos_mean=sum(pos_scores) / len(pos_scores) if pos_scores else 0,
                pos_final=pos_scores[-1] if pos_scores else 0,
                pfs_mean=sum(pfs_scores) / len(pfs_scores) if pfs_scores else 0,
                pfs_final=pfs_scores[-1] if pfs_scores else 0,
                bas_mean=sum(bas_scores) / len(bas_scores) if bas_scores else 0,
                essay_id=essay_id,
                model_name=self.model_name,
            )
            
            all_scores.append(scores)
        
        # Cleanup
        del hidden_states, attn_weights, outputs
        torch.cuda.empty_cache()
        
        return all_scores


def main():
    parser = argparse.ArgumentParser(description="Compute mechanistic scores for citations")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--input", type=str, required=True, help="Path to verified essays JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for scores")
    parser.add_argument("--max-essays", type=int, default=None, help="Max essays to process")
    args = parser.parse_args()
    
    # Load essays
    with open(args.input, 'r') as f:
        essays = json.load(f)
    
    if args.max_essays:
        essays = essays[:args.max_essays]
    
    print(f"Processing {len(essays)} essays")
    
    # Initialize scorer
    scorer = ZeroShotScorer(model_name=args.model)
    
    # Score all essays
    all_scores = []
    for essay in tqdm(essays, desc="Scoring essays"):
        try:
            scores = scorer.score_essay(essay)
            all_scores.extend([asdict(s) for s in scores])
        except Exception as e:
            print(f"Error scoring essay {essay.get('prompt_id', 'unknown')}: {e}")
            continue
    
    # Save scores
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_short = args.model.split("/")[-1]
    output_file = output_dir / f"{model_short}_scores.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_scores, f, indent=2)
    
    print(f"Saved {len(all_scores)} citation scores to {output_file}")
    
    # Print statistics by label
    from collections import defaultdict
    stats = defaultdict(lambda: {"count": 0, "ics_sum": 0, "pos_sum": 0, "pfs_sum": 0})
    
    for score in all_scores:
        label = score["label"]
        stats[label]["count"] += 1
        stats[label]["ics_sum"] += score["ics_mean"]
        stats[label]["pos_sum"] += score["pos_mean"]
        stats[label]["pfs_sum"] += score["pfs_mean"]
    
    print("\nScore Statistics by Label:")
    for label, data in stats.items():
        n = data["count"]
        print(f"  {label} (n={n}):")
        print(f"    ICS mean: {data['ics_sum']/n:.4f}")
        print(f"    POS mean: {data['pos_sum']/n:.4f}")
        print(f"    PFS mean: {data['pfs_sum']/n:.4f}")


if __name__ == "__main__":
    main()
