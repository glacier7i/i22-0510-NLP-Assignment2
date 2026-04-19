# NLP Assignment 2 — Neural NLP Pipeline for BBC Urdu Corpus

> **CS-4063: Natural Language Processing** | FAST NUCES  
> Student: `i22-0510` &nbsp;|&nbsp; Framework: PyTorch (from scratch) &nbsp;|&nbsp; GPU: NVIDIA RTX 4060 8 GB  
> GitHub: https://github.com/glacier7i/i22-0510-NLP-Assignment2

---

## Overview

End-to-end NLP pipeline built from scratch on a 250-article BBC Urdu corpus (~400K tokens):

- **Part 1** — TF-IDF, PPMI co-occurrence matrix, Skip-gram Word2Vec (4 conditions), analogy evaluation
- **Part 2** — Manual POS & NER annotation (500 sentences), BiLSTM-CRF sequence labeler, ablation study
- **Part 3** — Transformer encoder (6 modules from scratch) for topic classification, comparison with BiLSTM

---

## Repository Structure

```
i22-0510-NLP-Assignment2/
│
├── i22-0510_Assignment2_AI-A.ipynb   ← Main notebook (62 cells, all outputs present)
├── report.pdf                         ← Assignment report
├── README.md
│
├── embeddings/
│   ├── word2idx.json                  ← Vocabulary (10,001 tokens)
│   ├── tfidf_matrix.npy               ← TF-IDF matrix (10001 × 250)
│   ├── ppmi_matrix.npz                ← PPMI co-occurrence matrix (sparse, 10001 × 10001)
│   ├── embeddings_w2v.npy             ← Skip-gram embeddings ½(V+U) (10001 × 100)
│   ├── tsne_ppmi.png                  ← t-SNE of PPMI vectors
│   └── tsne_w2v_c3.png                ← t-SNE of Word2Vec C3
│
├── models/
│   ├── bilstm_pos.pt                  ← Trained BiLSTM POS tagger
│   ├── bilstm_ner.pt                  ← Trained BiLSTM-CRF NER model
│   └── transformer_cls.pt             ← Trained Transformer topic classifier
│
├── data/
│   ├── pos_train.conll                ← POS training set (354 sentences)
│   ├── pos_test.conll                 ← POS test set (73 sentences)
│   ├── ner_train.conll                ← NER training set (BIO scheme)
│   └── ner_test.conll                 ← NER test set
│
├── run_part1.py                       ← TF-IDF, PPMI, Word2Vec C3, t-SNE
├── run_part1b.py                      ← Word2Vec C2/C4 + MRR evaluation
├── run_part2_topics.py                ← Topic label assignment (250 articles)
├── run_part2_annotate.py              ← POS & NER annotation → CoNLL files
├── run_part2_train.py                 ← BiLSTM-CRF training + ablation study
└── run_part3_transformer.py           ← Transformer training + BiLSTM comparison
```

---

## Setup

```bash
# Clone
git clone https://github.com/glacier7i/i22-0510-NLP-Assignment2.git
cd i22-0510-NLP-Assignment2

# Create conda environment
conda create -n nlpassignment python=3.11
conda activate nlpassignment

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm
```

> **Requirements:** Python 3.11 · PyTorch 2.11+ · CUDA 13.0 · 8 GB+ GPU VRAM

---

## Reproducing Results

### Part 1 — Word Embeddings

```bash
python run_part1.py     # TF-IDF, PPMI, Word2Vec C3 (cleaned, d=100), t-SNE
python run_part1b.py    # Word2Vec C2 (raw) + C4 (d=200) + MRR evaluation
```

### Part 2 — Sequence Labeling

```bash
python run_part2_topics.py     # Assign topic labels to 250 articles
python run_part2_annotate.py   # Annotate 500 sentences → CoNLL files
python run_part2_train.py      # Train BiLSTM-CRF + ablation study
```

### Part 3 — Transformer Encoder

```bash
python run_part3_transformer.py   # Train Transformer + BiLSTM, attention heatmaps
```

---

## Results

### Sequence Labeling & Classification

| Model | Task | Accuracy | Macro-F1 |
|---|---|:---:|:---:|
| BiLSTM (fine-tuned) | POS Tagging | 0.953 | 0.953 |
| BiLSTM-CRF | NER (token-level) | — | 0.110 |
| Transformer | Topic Classification | 0.342 | 0.102 |
| BiLSTM | Topic Classification | 0.421 | 0.415 |

### Word2Vec Analogy Evaluation (MRR — 20 pairs)

| Condition | Description | MRR |
|---|---|:---:|
| C1 | PPMI + SVD (d=50) | 0.056 |
| C2 | Skip-gram · raw.txt · d=100 | 0.017 |
| C3 | Skip-gram · cleaned · d=100 | 0.004 |
| C4 | Skip-gram · cleaned · d=200 | 0.025 |

---

## Implementation Notes

- **No pretrained models** — no HuggingFace, no Gensim, no pretrained weights anywhere
- **No `nn.Transformer`** — self-attention, multi-head attention, positional encoding all built from scratch
- **Custom CRF** — forward algorithm + Viterbi decoding implemented manually; TorchCRF not used
- **CUDA throughout** — `device = torch.device('cuda')` set globally; trained on RTX 4060
- **Low NER/Transformer F1 is expected** — only ~500 sentences (2% NER tokens), 250 articles too small for self-attention; documented as known limitations in the notebook
