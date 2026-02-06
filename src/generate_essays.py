"""
Essay Generation Pipeline for Citation Hallucination Research

This script generates essays from LLMs using prompts designed to elicit citations,
then extracts and parses citations for verification.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class Citation:
    """Represents an extracted citation from generated text."""
    raw_text: str
    start_pos: int
    end_pos: int
    extracted_authors: Optional[List[str]] = None
    extracted_year: Optional[int] = None
    extracted_title: Optional[str] = None
    citation_type: str = "unknown"  # academic, legal, historical


@dataclass
class GeneratedEssay:
    """Represents a generated essay with metadata."""
    prompt_id: str
    domain: str
    prompt: str
    model_name: str
    response: str
    citations: List[Dict]
    generation_params: Dict
    timestamp: str


class CitationExtractor:
    """Extract and parse citations from generated text."""
    
    # Academic citation patterns
    ACADEMIC_PATTERNS = [
        # (Author et al., 2020) or (Author & Other, 2020)
        r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][a-z]+)*,?\s*\d{4}[a-z]?)\)',
        # Author et al. (2020)
        r'([A-Z][a-z]+(?:\s+et\s+al\.?))\s*\((\d{4}[a-z]?)\)',
        # "Paper Title" (Year)
        r'"([^"]{10,100})"\s*\((\d{4})\)',
        # Author (Year) showed/found/demonstrated
        r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?)\s*\((\d{4})\)\s+(?:showed|found|demonstrated|proposed|introduced)',
    ]
    
    # Legal citation patterns
    LEGAL_PATTERNS = [
        # Case v. Case (Year)
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((\d{4})\)',
        # Case v. Case, Volume Reporter Page (Year)
        r'([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+),?\s+\d+\s+[A-Z\.]+\s+\d+\s*\((\d{4})\)',
        # Statute references like "Section 1983" or "Title VII"
        r'((?:Section|Title|Article)\s+[IVXLCDM\d]+(?:\([a-z]\))?)',
    ]
    
    # Historical citation patterns
    HISTORICAL_PATTERNS = [
        # Author (Year) - for historians
        r'([A-Z][a-z]+)\s*\((\d{4})\)',
        # According to Author (Year)
        r'(?:According to|As|Per)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\((\d{4})\)',
        # The Treaty of X (Year)
        r'((?:Treaty|Act|Declaration|Proclamation)\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\((\d{4})\)',
    ]
    
    def __init__(self):
        self.all_patterns = {
            'academic': [re.compile(p) for p in self.ACADEMIC_PATTERNS],
            'legal': [re.compile(p) for p in self.LEGAL_PATTERNS],
            'historical': [re.compile(p) for p in self.HISTORICAL_PATTERNS],
        }
    
    def extract_citations(self, text: str, domain: str = "scientific") -> List[Citation]:
        """Extract all citations from text based on domain."""
        citations = []
        seen_positions = set()
        
        # Map domain to pattern type
        pattern_type = {
            'scientific': 'academic',
            'legal': 'legal',
            'historical': 'historical'
        }.get(domain, 'academic')
        
        # Try domain-specific patterns first
        patterns = self.all_patterns.get(pattern_type, self.all_patterns['academic'])
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                # Avoid duplicate overlapping matches
                if any(start <= pos <= end for pos in seen_positions):
                    continue
                    
                seen_positions.add(start)
                seen_positions.add(end)
                
                citation = Citation(
                    raw_text=match.group(0),
                    start_pos=start,
                    end_pos=end,
                    citation_type=pattern_type
                )
                
                # Try to extract year
                year_match = re.search(r'\d{4}', match.group(0))
                if year_match:
                    citation.extracted_year = int(year_match.group())
                
                # Try to extract author names
                author_pattern = r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][a-z]+)?)'
                author_match = re.search(author_pattern, match.group(0))
                if author_match:
                    citation.extracted_authors = [author_match.group(1)]
                
                citations.append(citation)
        
        return citations


class EssayGenerator:
    """Generate essays from LLMs with citation-eliciting prompts."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.citation_extractor = CitationExtractor()
        
    def generate_essay(self, prompt_data: Dict) -> GeneratedEssay:
        """Generate a single essay from a prompt."""
        prompt = prompt_data['prompt']
        prompt_id = prompt_data['id']
        domain = prompt_data['domain']
        
        # Format as instruction
        formatted_prompt = f"""You are a knowledgeable expert. Please write a detailed response to the following request. Be sure to include specific citations with author names and years where appropriate.

Request: {prompt}

Response:"""
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        response = full_response[len(formatted_prompt):].strip()
        
        # Extract citations
        citations = self.citation_extractor.extract_citations(response, domain)
        
        return GeneratedEssay(
            prompt_id=prompt_id,
            domain=domain,
            prompt=prompt,
            model_name=self.model_name,
            response=response,
            citations=[asdict(c) for c in citations],
            generation_params={
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            timestamp=datetime.now().isoformat(),
        )


def main():
    parser = argparse.ArgumentParser(description="Generate essays with citations")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--max-prompts", type=int, default=None, help="Max prompts to process")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize generator
    generator = EssayGenerator(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Generate essays
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for prompt_data in tqdm(prompts, desc="Generating essays"):
        try:
            essay = generator.generate_essay(prompt_data)
            results.append(asdict(essay))
        except Exception as e:
            print(f"Error generating essay for {prompt_data['id']}: {e}")
            continue
    
    # Save results
    model_short = args.model.split("/")[-1]
    output_file = output_dir / f"{model_short}_essays.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated {len(results)} essays. Saved to {output_file}")
    
    # Print citation statistics
    total_citations = sum(len(r['citations']) for r in results)
    print(f"Total citations extracted: {total_citations}")
    print(f"Average citations per essay: {total_citations / len(results):.2f}")


if __name__ == "__main__":
    main()
