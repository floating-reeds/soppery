# Citation Hallucination Detection - Google Colab Setup

## Quick Start (Copy-Paste into Colab)

### Cell 1: Clone Repository
```python
!git clone https://github.com/floating-reeds/soppery.git
%cd soppery
```

### Cell 2: Install Dependencies
```python
!pip install -q torch transformers accelerate
!pip install -q numpy pandas scipy scikit-learn matplotlib seaborn
!pip install -q requests tqdm

# Optional: Install FACTUM's modified transformers for full score extraction
# !pip install -e FACTUM-main/transformers
```

### Cell 3: Verify GPU
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 4: Run Essay Generation (Example)
```python
!python src/generate_essays.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompts data/prompts/scientific.json \
  --output data/generations/ \
  --max-prompts 5
```

### Cell 5: Verify & Score Citations
```python
!python src/verify_citations.py \
  --input data/generations/*.json \
  --output data/annotations/verified.json

!python src/zero_shot_scorer.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --input data/annotations/verified.json \
  --output data/activations/
```

---

## Notes

- **GPU Runtime**: Go to Runtime → Change runtime type → T4 GPU
- **Llama Access**: You may need to accept terms at huggingface.co and login:
  ```python
  from huggingface_hub import login
  login()  # Enter your HF token
  ```
- **Memory**: For 8B+ models, use Colab Pro (A100) or enable CPU offload
