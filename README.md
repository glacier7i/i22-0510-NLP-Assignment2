# i22-0510-NLP-Assignment2

## CS-4063: Natural Language Processing — Assignment 2
Neural NLP Pipeline for BBC Urdu Corpus

### Environment
- **Python**: 3.11 (conda env: nlpassignment)
- **PyTorch**: 2.11.0+cu130
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)

### Reproduction
1. Activate conda environment: `conda activate nlpassignment`
2. Ensure `cleaned.txt`, `raw.txt`, and `metadata.json` are in the project root.
3. Open and run `i22-0510-assignment-2.ipynb` from top to bottom.
4. All outputs (embeddings, models, data files) will be generated automatically.

### Structure
- `embeddings/` — TF-IDF, PPMI, and Word2Vec embedding matrices
- `models/` — Trained BiLSTM and Transformer model checkpoints
- `data/` — Annotated POS and NER datasets in CoNLL format
