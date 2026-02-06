# Citation Hallucination Detection in the Wild

Mechanistic detection of citation hallucination in zero-shot LLM generation, adapting the [FACTUM framework](https://github.com/Maximeswd/citation-hallucination) for non-RAG settings.

## Overview

This project detects fabricated citations by analyzing internal model pathways when citations are generated. Unlike the original FACTUM which measures alignment with external source documents, we measure **Internal Consistency (ICS)** - alignment with the model's own generated essay context.

## Key Contributions

1. **ICS Score**: Adapted Context Alignment Score for zero-shot settings
2. **Multi-domain prompts**: Scientific, legal, and historical citation scenarios
3. **Lightweight guardrail**: Real-time trust scoring for citations

## Installation

```bash
# Clone repository
cd d:/code/sop

# Create environment
conda env create -f FACTUM-main/environment.yml
conda activate citation

# Install dependencies
pip install -r requirements.txt

# Install FACTUM's modified transformers (for full score extraction)
pip install -e FACTUM-main/transformers
```

## Usage

### 1. Generate Essays with Citations

```bash
python src/generate_essays.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompts data/prompts/scientific.json \
  --output data/generations/ \
  --temperature 0.7
```

### 2. Verify Citations

```bash
python src/verify_citations.py \
  --input data/generations/Llama-3.1-8B-Instruct_essays.json \
  --output data/annotations/verified_essays.json \
  --api-key YOUR_SEMANTIC_SCHOLAR_KEY  # Optional
```

### 3. Compute Mechanistic Scores

```bash
python src/zero_shot_scorer.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input data/annotations/verified_essays.json \
  --output data/activations/
```

### 4. Train Classifier

```bash
python src/train_classifier.py \
  --scores data/activations/*_scores.json \
  --output models/
```

### 5. Run Analysis

```bash
python experiments/run_analysis.py \
  --scores-dir data/activations/ \
  --output outputs/
```

## Project Structure

```
d:/code/sop/
├── FACTUM-main/           # Original FACTUM implementation
├── data/
│   ├── prompts/           # 150 prompts across 3 domains
│   ├── generations/       # Generated essays
│   ├── annotations/       # Verified citations
│   └── activations/       # Mechanistic scores
├── src/
│   ├── generate_essays.py     # Essay generation
│   ├── verify_citations.py    # Citation verification
│   ├── zero_shot_scorer.py    # Mechanistic scoring
│   └── train_classifier.py    # Guardrail training
├── experiments/
│   └── run_analysis.py        # Analysis experiments
├── models/                # Trained classifiers
└── outputs/               # Results and plots
```

## Mechanistic Scores

| Score | Description | Expected Signal |
|-------|-------------|-----------------|
| ICS | Internal Consistency with essay context | Lower for fabricated |
| POS | Pathway Orthogonality (attn↔FFN alignment) | Lower for fabricated |
| PFS | Parametric Force (FFN update magnitude) | Different distribution |
| BAS | BOS attention (synthesis indicator) | Higher for complex synthesis |

## Citation

Based on the FACTUM framework:
```bibtex
@inproceedings{dassen2026factum,
  author = {Dassen, Maxime and Kotula, Rebecca and Murray, Kenton and Yates, Andrew and Lawrie, Dawn and Kayi, Efsun and Mayfield, James and Duh, Kevin},
  title = {{FACTUM}: Mechanistic Detection of Citation Hallucination in Long-Form {RAG}},
  booktitle = {ECIR 2026},
  year = {2026}
}
```
