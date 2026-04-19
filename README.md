# i22-0510-NLP-Assignment2

**CS-4063: Natural Language Processing — Assignment 2**
Neural NLP Pipeline for BBC Urdu Corpus | FAST NUCES

> Student: i22-0510 | Framework: PyTorch (from scratch) | GPU: NVIDIA RTX 4060 8 GB
>
> **GitHub:** https://github.com/glacier7i/i22-0510-NLP-Assignment2

---

## Repository Structure

```
i22-0510-NLP-Assignment2/
├── i22-0510-assignment-2.ipynb   ← Main notebook (all cells executed)
├── report.pdf                     ← 2–3 page PDF report
├── README.md
│
├── embeddings/
│   ├── tfidf_matrix.npy           ← TF-IDF term-document matrix (10001 × 250)
│   ├── ppmi_matrix.npy            ← PPMI co-occurrence matrix (10001 × 10001, dense)
│   ├── embeddings_w2v.npy         ← Averaged Skip-gram embeddings ½(V+U) (10001 × 100)
│   └── word2idx.json              ← Vocabulary index (10001 tokens)
│
├── models/
│   ├── bilstm_pos.pt              ← Trained BiLSTM POS tagger
│   ├── bilstm_ner.pt              ← Trained BiLSTM-CRF NER model
│   └── transformer_cls.pt         ← Trained Transformer topic classifier
│
├── data/
│   ├── pos_train.conll            ← POS training set (354 sentences)
│   ├── pos_test.conll             ← POS test set  (73 sentences)
│   ├── ner_train.conll            ← NER training set (BIO scheme)
│   └── ner_test.conll             ← NER test set
│
├── run_part1.py                   ← Part 1: TF-IDF, PPMI, Word2Vec C3
├── run_part1b.py                  ← Part 1: Word2Vec C2/C4 + MRR evaluation
├── run_part2_topics.py            ← Part 2: Topic label assignment
├── run_part2_annotate.py          ← Part 2: POS & NER annotation
├── run_part2_train.py             ← Part 2: BiLSTM + CRF training + ablation
└── run_part3_transformer.py       ← Part 3: Transformer training + comparison
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/glacier7i/i22-0510-NLP-Assignment2.git
cd i22-0510-NLP-Assignment2

# 2. Create and activate conda environment
conda create -n nlpassignment python=3.11
conda activate nlpassignment

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm
```

**Requirements:** Python 3.11, PyTorch 2.11+, CUDA 13.0, 8 GB+ GPU VRAM recommended.

---

## Reproducing Each Part

### Part 1 — Word Embeddings

```bash
# Step 1: TF-IDF, PPMI, Word2Vec (C3: cleaned.txt, d=100), t-SNE
python run_part1.py

# Step 2: Word2Vec C2 (raw.txt) + C4 (d=200) + MRR evaluation
python run_part1b.py
```

**Outputs:** `embeddings/tfidf_matrix.npy`, `embeddings/ppmi_matrix.npy`, `embeddings/ppmi_matrix.npz`, `embeddings/embeddings_w2v.npy`, `embeddings/word2idx.json`, loss curves, t-SNE plots.

### Part 2 — Sequence Labeling

```bash
# Step 1: Assign topic labels to 250 articles
python run_part2_topics.py

# Step 2: Annotate 500 sentences (POS + NER), save CoNLL files
python run_part2_annotate.py

# Step 3: Train BiLSTM-CRF, run ablation study
python run_part2_train.py
```

**Outputs:** `data/*.conll`, `models/bilstm_pos.pt`, `models/bilstm_ner.pt`, confusion matrix, training curves.

### Part 3 — Transformer Encoder

```bash
# Train Transformer + BiLSTM classifier, generate attention heatmaps
python run_part3_transformer.py
```

**Outputs:** `models/transformer_cls.pt`, confusion matrix, attention heatmaps, comparison curves.

---

## Key Results

| Model | Task | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| BiLSTM (fine-tuned) | POS Tagging | 0.953 | 0.953 |
| BiLSTM-CRF | NER | — | 0.110 |
| Transformer | Topic Classification | 0.342 | 0.102 |
| BiLSTM | Topic Classification | 0.421 | 0.415 |

| Condition | MRR (20 pairs) |
|-----------|---------------|
| C1: PPMI (SVD-50d) | 0.056 |
| C2: Skip-gram raw.txt d=100 | 0.017 |
| C3: Skip-gram cleaned d=100 | 0.004 |
| C4: Skip-gram cleaned d=200 | 0.025 |

---

## Implementation Notes

- **No pretrained models, no HuggingFace, no Gensim** — everything implemented from scratch in PyTorch.
- `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder` are **not used**.
- CRF is fully custom (forward algorithm + Viterbi decoding). TorchCRF library is not used.
- All models trained on CUDA (RTX 4060). The `device = torch.device('cuda')` is set throughout.
- Low NER F1 (0.11) and Transformer F1 (0.10) are expected — documented in notebook as known limitations of the small corpus (~500 sentences, 250 articles).
