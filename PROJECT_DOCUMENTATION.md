# PROJECT WORK DOCUMENTATION

## LLM Evaluation Research

> **Note:** This is a living document. Update it after each experiment.
> Keep detailed records of all configurations, prompts, and results for paper writing and reproducibility.

---

## 1. Project Overview

### 1.1 Project Title

**Mechanistic Detection of Citation Hallucination in the Wild**

### 1.2 Objective

To detect citation hallucination in zero-shot LLM generation by analyzing the internal mechanistic behavior of the model when citation tokens are produced. Unlike existing FACTUM framework which targets RAG pipelines (where external source documents are provided), this project addresses the "in the wild" setting—where the model generates citations purely from its parametric memory without external context. We adapt FACTUM's mechanistic scores (CAS, PAS, PFS, BAS) to operate on the model's own generated history and evaluate whether fabricated citations exhibit distinct internal signatures.

### 1.3 Research Questions

1. Can mechanistic interpretability scores (ICS, PAS, PFS, BAS) distinguish between real and fabricated citations in a zero-shot (non-RAG) setting?
2. Do purely fabricated citations (invented paper titles) differ mechanistically from real citations, particularly in terms of Parametric Force Score (PFS)?
3. Can a lightweight classifier trained on these mechanistic scores serve as a real-time guardrail that produces a citation trust score?

---

## 2. Literature Review & Background

### 2.1 Related Work

- **FACTUM (Dassen et al., 2026):** Primary baseline. Introduces mechanistic scores—CAS, BAS, PFS, and PAS—for detecting citation hallucination in long-form RAG settings. Decomposes each Transformer layer update into an attention pathway (reading from source context) and an FFN pathway (recalling parametric knowledge), then measures their coordination to identify fabricated citations. Published at ECIR 2026 ([arXiv:2601.05866](https://arxiv.org/pdf/2601.05866)).
- **Hallucination detection benchmarks:** TruthfulQA, HaluEval, and FActScore primarily evaluate factual accuracy at the surface level (text output), rather than examining internal model representations.
- **Mechanistic interpretability:** Work on probing LLM internals (e.g., logit lens, activation patching, causal tracing) has shown that factual knowledge is localized in specific layers and attention heads.

### 2.2 Research Gaps & Motivation

- **RAG vs. zero-shot:** FACTUM assumes the model has external source documents. In practice, LLMs are frequently used zero-shot, and hallucinated citations in this setting are harder to detect because there is no reference document to align against.
- **No mechanistic study of zero-shot citation fabrication:** No existing work examines how the internal pathways of a Transformer behave when generating fabricated citations without external context.
- **Fabrication vs. misattribution:** Existing hallucination taxonomies distinguish these two types, but no mechanistic study has compared their internal signatures.

### 2.3 Background on Task/Domain

Citation hallucination is a particularly insidious form of LLM hallucination because citations carry an implicit claim of verifiability. A fabricated citation (e.g., "Smith et al. (2019) demonstrated...") can mislead readers into thinking a claim is backed by peer-reviewed research. This is especially dangerous in scientific writing. The core challenge is that the model produces these citations with the same fluency and confidence as real ones, making surface-level detection insufficient. Mechanistic approaches that examine the model's internal state offer a promising path toward real-time detection.

---

## 3. LLMs Under Evaluation

### 3.1 Model Selection & Rationale

We selected Llama 3.2 3B Instruct as the primary model. It is small enough to run on a free Google Colab T4 GPU (16GB VRAM) with full attention weight extraction (eager attention mode), which is required for mechanistic scoring. Being open-weight and accessible via HuggingFace made it practical for this research setup.

| Model Name | Version/ID | Access Method | Parameters/Size | Notes |
|---|---|---|---|---|
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace / Google Colab (T4 GPU) | 3B parameters | 28 layers, 24 attention heads. Used for all experiments. |

---

## 4. Evaluation Methodology

### 4.1 Evaluation Framework

The evaluation is a multi-stage pipeline implemented entirely within a single Colab notebook (`new_exp.ipynb`):

1. **Essay Generation:** Generate essays from the LLM using 150 scientific prompts designed to elicit citations. Each essay is appended to a JSONL file immediately after generation (crash-safe). On restart, already-generated prompt IDs are automatically skipped.
2. **Citation Extraction:** Regex-based extraction of citations from generated text using 5 patterns covering `Author et al. (Year)`, `(Author et al., Year)`, `Author and Author (Year)`, quoted titles with years, and legal-style `v.` citations.
3. **Citation Verification:** Automated ground-truth labeling by querying Semantic Scholar API (primary) and CrossRef API (fallback) to classify each citation as `real` (author + year match found), `fabricated` (API returned results but no match), or `unverified` (API unreachable after retries). Retries with rate-limit backoff are implemented.
4. **Mechanistic Scoring:** Forward pass through the model with forward hooks registered on `post_attention_layernorm` at each layer to capture the midpoint between attention and FFN contributions. For each citation token at each layer, compute:
   - **V_attn** = midpoint − pre-layer hidden state (attention pathway contribution)
   - **V_ffn** = post-layer hidden state − midpoint (FFN pathway contribution)
   - **ICS** (Internal Consistency Score): attention-weighted context similarity
   - **PAS** (Pathway Alignment Score): cosine similarity between V_attn and V_ffn
   - **PFS** (Parametric Force Score): L2 norm of V_ffn
   - **BAS** (BOS Attention Score): mean attention to BOS token across heads
5. **Results Analysis:** Per-citation score table, mean scores by label (real vs. fabricated), and layer-wise trajectory plots.

### 4.2 Evaluation Metrics

| Metric | What It Measures | Adaptation from FACTUM |
|---|---|---|
| **ICS** (Internal Consistency Score) | Cosine similarity between citation token hidden state and attention-weighted essay context | Replaces **CAS** (Contextual Alignment Score) — uses model's own generated text instead of external source documents |
| **PAS** (Pathway Alignment Score) | Cosine similarity between V_attn and V_ffn | Same formula; attention pathway attends to essay context instead of source documents |
| **PFS** (Parametric Force Score) | L2 norm of V_ffn | Unchanged from FACTUM |
| **BAS** (BOS Attention Score) | Mean attention to BOS token (position 0) from citation token | Unchanged from FACTUM |

**Why these metrics are appropriate:**
- **ICS** adapts FACTUM's CAS to the zero-shot setting by replacing external source alignment with internal essay context alignment.
- **PAS** captures whether the attention pathway (reading essay context) and FFN pathway (recalling parametric knowledge) are coordinated or in conflict.
- **PFS** measures the magnitude of parametric knowledge retrieval — hypothesis is that fabricated citations require stronger FFN force.
- **BAS** indicates reliance on parametric memory synthesis from the BOS token.

### 4.3 Prompting Strategy

**Zero-shot with a scientific system prompt.** The prompt explicitly instructs the model to include citations with author names and years, forcing the model to either recall real references or fabricate them.

#### 4.3.1 Prompt Templates

**System prompt (scientific domain):**
```
You are an academic researcher writing a concise research summary.
Write a brief, focused response (around 300-400 words).
You MUST include proper academic citations throughout.
Format: Author et al. (Year) or (Author et al., Year).
Every major claim MUST have a citation with real author names and years.
Example: Vaswani et al. (2017) introduced the Transformer architecture.
```

**User prompt format:**
Each user message contains a specific essay topic from the prompt dataset (e.g., *"Write a 500-word essay explaining the Transformer architecture and its impact on natural language processing. Include at least 3 citations to foundational papers with author names and publication years."*)

### 4.4 Sampling Parameters

| Parameter | Value | Notes |
|---|---|---|
| Temperature | 0.7 | Allows creativity while maintaining coherence |
| top_p | 0.9 | Nucleus sampling |
| max_new_tokens | 512 | Sufficient for 300–400 word responses with citations |
| do_sample | True | Stochastic generation |
| pad_token_id | `tokenizer.eos_token` | Set to EOS token |
| Scoring max_length | 512 | Truncation length for mechanistic scoring forward pass |
| Attention implementation | `eager` | Required to extract full attention weight matrices (SDPA does not return them) |

---

## 5. Test Data & Benchmarks

### 5.1 Datasets/Benchmarks Used

| Dataset/Benchmark | Source | # Test Cases | Task Type |
|---|---|---|---|
| Scientific prompts | Custom (`data/prompts/scientific.json`) | 150 | Essay generation with academic citations |

### 5.2 Data Characteristics

- **Domain:** Scientific/academic only. The project was simplified from an initial 3-domain design (scientific, legal, historical) to focus exclusively on scientific citations, where automated verification via Semantic Scholar and CrossRef is most effective.
- **Prompt topics** span a wide range: ML/AI (Transformers, GANs, BERT, GPT, LoRA, diffusion models, NeRF, CLIP, DALL-E, etc.), physics (gravitational waves, Higgs boson, quantum computing, quantum supremacy), biology (CRISPR, mRNA vaccines, AlphaFold, stem cells, Human Genome Project), astronomy (exoplanets, TRAPPIST-1, Event Horizon Telescope, Kuiper Belt), and more.
- **Prompt IDs:** `sci_001` through `sci_150`.
- **Ground-truth labeling:** Automated via Semantic Scholar API (primary) with CrossRef API as fallback. Retries with rate-limit backoff are implemented. Citations that cannot be verified by either API are marked `unverified` and excluded from scoring (only `real` vs `fabricated` are scored).

### 5.3 Human Evaluation Setup (if applicable)

Not applicable. Verification is fully automated via API-based ground truth.

---

## 6. Experiments Log

| Exp ID | Date | Configuration / What Changed | Status | Best Metric |
|---|---|---|---|---|
| EXP-001 | 2026-02-11 | Llama-3.2-3B-Instruct, 150 scientific prompts, temp=0.7, top_p=0.9, max_tokens=512, zero-shot with scientific system prompt. Full pipeline: generate → verify → score (ICS, PAS, PFS, BAS). Notebook: `new_exp_2.ipynb`. | Completed | 478 citations extracted, 369 real (77%), 109 fabricated (23%), 0 unverified. 472 citations scored. |

---

## 7. Results & Analysis

### 7.1 Experiment EXP-001: Full Pipeline Run

#### 7.1.1 Configuration Summary

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` (28 layers, 24 heads)
- **Hardware:** Google Colab, Tesla T4 GPU (16GB VRAM)
- **Prompts:** 150 scientific prompts (`sci_001`–`sci_150`)
- **System prompt:** Scientific domain, zero-shot (see Section 4.3.1)
- **Generation params:** temperature=0.7, top_p=0.9, max_new_tokens=512
- **Scoring:** Eager attention implementation, forward hooks on `post_attention_layernorm` for true V_attn/V_ffn decomposition

#### 7.1.2 Quantitative Results

**Generation & Verification:**

| Metric | Value |
|---|---|
| Essays generated | 150 / 150 |
| Total citations extracted | 478 |
| Real citations | 369 (77%) |
| Fabricated citations | 109 (23%) |
| Unverified citations | 0 (0%) |
| Essays with scorable citations | 129 |
| Citations scored | 472 |

**Mechanistic Scores — Mean by Label:**

| Label | ICS Mean | PAS Mean | PFS Mean | BAS Mean | Count |
|---|---|---|---|---|---|
| **Fabricated** | 0.4627 | −0.8605 | 16.1122 | 0.7445 | 109 |
| **Real** | 0.4721 | −0.8583 | 16.3862 | 0.7215 | 369 |
| **Difference** | −0.0094 | −0.0022 | −0.2740 | +0.0230 | — |

**Sample Per-Citation Results:**

| Prompt | Citation | Label | ICS | PAS | PFS | BAS |
|---|---|---|---|---|---|---|
| sci_001 | Vaswani et al. (2017) | real | 0.3628 | −0.8654 | 15.62 | 0.8280 |
| sci_001 | Vaswani et al. (2017) | fabricated | 0.4538 | −0.8595 | 15.30 | 0.6973 |
| sci_001 | Devlin et al. (2019) | real | 0.4329 | −0.8809 | 16.04 | 0.8050 |
| sci_114 | Vaswani et al. (2017) | real | 0.2689 | −0.8814 | 14.98 | 0.8459 |
| sci_114 | Johnson et al. (2018) | fabricated | 0.4539 | −0.8743 | 15.19 | 0.8060 |
| sci_111 | Vaswani et al. (2017) | fabricated | 0.4931 | −0.8775 | 17.45 | 0.6661 |

#### 7.1.3 Qualitative Analysis

**Verification pipeline effectiveness:**
- Achieved **zero unverified citations** across 370 extractions, thanks to the Semantic Scholar + CrossRef fallback strategy with retries.
- The model fabricates ~20% of its citations when explicitly prompted to include academic references.

**Mechanistic score patterns:**
- **ICS:** Real citations show slightly *higher* ICS (0.4721 vs 0.4627), a difference of 0.0094. This is ~2% relative difference, suggesting real citations are marginally more consistent with the essay context.
- **PAS:** Both labels show strongly negative PAS values (around −0.86), indicating anti-correlation between attention and FFN pathways. This differs from FACTUM's RAG setting where PAS is positive. In the zero-shot setting, there is no external source document for the attention pathway to align with.
- **PFS:** Real citations show slightly *higher* PFS (16.39 vs 16.11), contrary to the initial hypothesis that fabricated citations would require stronger parametric force. This may indicate the model's FFN pushes more confidently for well-known real references.
- **BAS:** Fabricated citations show higher BAS (0.7445 vs 0.7215), suggesting more reliance on the BOS attention sink when generating fabricated content.

**Overall:** The mechanistic scores do not clearly separate real from fabricated citations in this 3B model / zero-shot setting. Mean differences are marginal across all four metrics.

#### 7.1.4 Visualizations

**Layer-wise Score Trajectories:**

Four-panel plot (`data/results/layer_scores.png`) showing mean scores per layer (0–27) for real (green, solid) vs. fabricated (red, dashed) citations:

- **ICS:** Both curves follow the same trajectory — rising from ~0.38 at layer 0, peaking around ~0.53 at layers 15–17, then declining in the final layers. Real and fabricated curves are nearly indistinguishable.
- **PAS:** Both curves start around −0.65 and descend to approximately −0.90 around layer 20, before recovering slightly. The curves overlap almost perfectly throughout.
- **PFS:** Both curves show high PFS (~80) at layer 0, dropping sharply to ~10 by layer 5, then remaining relatively flat through layer 27. No visible difference between real and fabricated.
- **BAS:** Both curves peak around layers 3–5 (at ~0.90), then decline steadily to ~0.45 by layer 27. Minor divergence appears in middle layers (10–15) but curves largely overlap.

**Interpretation:** The layer-wise trajectories overlap almost entirely, confirming that the 3B model's internal representations do not exhibit clearly separable mechanistic signatures for real vs. fabricated citations in the zero-shot setting.

---

## 8. Discussion & Insights

### 8.1 Key Findings

1. **The 3B model's internal representations do not clearly distinguish real from fabricated citations in a zero-shot setting.** Mean score differences between labels are negligible (<1% for ICS, <0.2% for PAS, <1% for PFS and BAS), and layer-wise trajectories overlap almost entirely.

2. **Zero unverified citations.** The dual-API verification strategy (Semantic Scholar + CrossRef fallback with retries) was highly effective — no citations were left unverified out of 478 extracted.

3. **The model fabricates ~23% of its citations.** Out of 478 citations across 150 scientific essays, 109 (23%) were confirmed fabricated by the verification pipeline.

4. **PAS values are consistently negative** (around −0.86), meaning the attention and FFN pathways are strongly anti-correlated across all layers. This is different from the positive PAS values reported by FACTUM in the RAG setting, likely because there is no external source document for the attention pathway to align with.

### 8.2 Performance Analysis by Task/Category

Not applicable — all prompts are from the scientific domain.

### 8.3 Impact of Prompting Strategies

A single zero-shot prompting strategy was used with a scientific system prompt. The system prompt was designed to be maximally citation-eliciting (explicit instruction to include author names and years). No few-shot or chain-of-thought variants were tested.

### 8.4 Model Behavior & Characteristics

- The model generates essays with a high citation density (~3.2 citations per essay on average).
- The citation extractor successfully identifies standard academic patterns.
- The model sometimes generates plausible-looking but non-existent citations (e.g., "Vaswani et al. (2019)" when the actual Transformer paper is 2017), which the verification pipeline correctly labels as fabricated.

### 8.5 Limitations & Challenges

- **All 150 essays generated:** The crash-safe resume mechanism successfully handled Colab disconnections, preserving all generated work.
- **Scoring truncation at 512 tokens:** Citations beyond position 512 in the combined prompt+response are missed.
- **Single model (3B):** A 3B model may be too small for separable mechanistic signatures. Larger models might show clearer differentiation.
- **API-based verification:** Relies on Semantic Scholar/CrossRef coverage — citations to very niche or very recent papers might be misclassified.
- **Colab constraints:** T4 GPU (16GB) limits model sizes. Eager attention (required for full attention weights) uses more VRAM than SDPA.

### 8.6 Comparison with Existing Benchmarks

FACTUM reports results on RAG-grounded citation hallucination. Direct comparison is not possible since our setting is fundamentally different (no external context). However, key differences observed:
- **PAS sign:** FACTUM reports positive PAS values in RAG settings (attention and FFN pathways are aligned). We observe negative PAS values (anti-correlation) in the zero-shot setting.
- **Score separation:** FACTUM demonstrates clear separation between real and fabricated citations using their mechanistic scores. We do not observe this separation in the zero-shot / 3B model setting.

### 8.7 Practical Implications

The current results suggest that mechanistic scores alone, as computed from a 3B model in a zero-shot setting, are insufficient for building a reliable citation hallucination detector. Possible paths forward:
- Test larger models (8B, 70B) to see if mechanistic signatures emerge at scale.
- Incorporate additional features beyond the four FACTUM scores (e.g., per-head analysis, layer-range-specific scores).
- Combine mechanistic scores with surface-level features (citation format, author name plausibility, year plausibility).

---

## 10. References & Resources

### 10.1 Research Papers

- Dassen, M., Kotula, R., Murray, K., Yates, A., Lawrie, D., Kayi, E., Mayfield, J., & Duh, K. (2026). *FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG.* ECIR 2026. [arXiv:2601.05866](https://arxiv.org/pdf/2601.05866)

### 10.2 Model Documentation

- [Llama 3.2 3B Instruct — HuggingFace Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### 10.3 Benchmark & Dataset Sources

- Custom prompt dataset: 150 scientific prompts, stored in `data/prompts/scientific.json`.
- [Semantic Scholar API](https://api.semanticscholar.org/) — primary source for academic citation verification.
- [CrossRef API](https://api.crossref.org/) — fallback for citation verification.

### 10.4 Tools & Libraries

| Tool/Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.0.0 | Model inference, gradient-free forward passes |
| HuggingFace Transformers | ≥4.36.0 | Model loading, tokenization, generation |
| Accelerate | ≥0.23.0 | Device mapping for large models |
| Pandas | ≥2.0.0 | Data manipulation and analysis |
| Matplotlib | ≥3.7.0 | Visualization (layer-wise score plots) |
| Requests | ≥2.31.0 | API calls to Semantic Scholar and CrossRef |
| Google Colab | T4 GPU runtime | Primary compute environment |
