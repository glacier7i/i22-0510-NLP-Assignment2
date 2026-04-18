"""Run Part 1 cells as a script to validate code before notebook execution."""
import sys, os, re, json, math, random, warnings
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs('embeddings', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── A1 preprocessing ─────────────────────────────────────────────────────────
def remove_diacritics(text):
    return re.sub(r'[\u064B-\u065F]', '', text)

def remove_noise(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s\u06F0-\u06F9\u0660-\u0669\u060C\u061B\u061F\u06D4]', '', text)
    return text

def remove_non_urdu(text):
    return re.sub(r'[^\u0600-\u06FF\s\u06F0-\u06F9\u0660-\u0669]', ' ', text)

def segment_sentences(text):
    text = re.sub(r'([\u06D4\u061F!])', r'\1\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def urdu_tokenize(text):
    text = re.sub(r'[\u06F0-\u06F9]+', '<NUM>', text)
    text = re.sub(r'[0-9]+', '<NUM>', text)
    text = re.sub(r'[\u06D4\u061F!\u060C\u061B\u066A,;:\[\]{}()]', '', text)
    return [t.strip() for t in text.split() if t.strip()]

def urdu_lemmatize(word):
    if len(word) < 4: return word
    if word.endswith('\u06CC\u0627\u06BA'): return word[:-3] + '\u06CC\u0627'
    elif word.endswith('\u0626\u06CC\u06BA'): return word[:-3]
    elif word.endswith('\u06CC\u06BA') and len(word) > 4: return word[:-2]
    elif word.endswith('\u0648\u06BA') and len(word) > 4: return word[:-2]
    return word

def urdu_stem(word):
    if len(word) < 4: return word
    suffixes = ['\u06CC\u0648\u06BA', '\u0627\u0626\u06CC', '\u06CC\u0627\u06BA', '\u0626\u06CC\u06BA',
                '\u0646\u06D2', '\u06A9\u0648', '\u0633\u06D2', '\u067E\u0631',
                '\u062A\u0627', '\u062A\u06CC', '\u062A\u06D2', '\u06AF\u0627', '\u06AF\u06D2', '\u06AF\u06CC']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            if len(stem) >= 3: return stem
    return word

# ── 1.1 Data loading & vocab ──────────────────────────────────────────────────
print("\n=== 1.1 Data Loading ===")
with open('cleaned.txt', 'r', encoding='utf-8') as f:
    cleaned_text = f.read()

docs_raw = re.split(r'\n?\[(\d+)\]\n', cleaned_text)
documents = []
for i in range(1, len(docs_raw), 2):
    doc_id = int(docs_raw[i])
    doc_text = docs_raw[i+1].strip() if i+1 < len(docs_raw) else ''
    documents.append((doc_id, doc_text))

print(f"Parsed {len(documents)} documents")
tokenized_docs = [(doc_id, doc_text.split()) for doc_id, doc_text in documents]
all_tokens = [tok for _, tokens in tokenized_docs for tok in tokens]
token_counts = Counter(all_tokens)
vocab_tokens = [tok for tok, _ in token_counts.most_common(10000)]

word2idx = {'<UNK>': 0}
for i, tok in enumerate(vocab_tokens):
    word2idx[tok] = i + 1
idx2word = {v: k for k, v in word2idx.items()}

print(f"Vocab size: {len(word2idx)}, Total tokens: {len(all_tokens):,}")

with open('embeddings/word2idx.json', 'w', encoding='utf-8') as f:
    json.dump(word2idx, f, ensure_ascii=False)
print("Saved word2idx.json")

# ── 1.2 TF-IDF ────────────────────────────────────────────────────────────────
print("\n=== 1.2 TF-IDF ===")
V, N = len(word2idx), len(documents)
tfidf_matrix = np.zeros((V, N), dtype=np.float32)

for j, (doc_id, tokens) in enumerate(tokenized_docs):
    tf = Counter(tokens)
    for tok, cnt in tf.items():
        tfidf_matrix[word2idx.get(tok, 0), j] = cnt

df_vec = (tfidf_matrix > 0).sum(axis=1)
idf = np.log(N / (1 + df_vec))
tfidf_matrix = tfidf_matrix * idf[:, np.newaxis]
np.save('embeddings/tfidf_matrix.npy', tfidf_matrix)
print(f"TF-IDF shape: {tfidf_matrix.shape}, saved.")

# ── 1.3 PPMI ──────────────────────────────────────────────────────────────────
print("\n=== 1.3 PPMI Co-occurrence ===")
WINDOW = 5
cooc = sparse.lil_matrix((V, V), dtype=np.float32)

for _, tokens in tqdm(tokenized_docs, desc="Building co-occurrence"):
    ids = [word2idx.get(t, 0) for t in tokens]
    for pos, center in enumerate(ids):
        start, end = max(0, pos-WINDOW), min(len(ids), pos+WINDOW+1)
        for cp in range(start, end):
            if cp != pos:
                cooc[center, ids[cp]] += 1

cooc = cooc.tocsr()
print(f"Co-occurrence: {cooc.shape}, nnz={cooc.nnz:,}")

total = cooc.sum()
word_prob = np.array(cooc.sum(axis=1)).flatten() / total
cx = cooc.tocoo()
rows, cols, data = cx.row, cx.col, cx.data
pmi_data = np.log2((data/total) / (word_prob[rows]*word_prob[cols]+1e-10))
ppmi_data = np.maximum(0, pmi_data)
ppmi = sparse.csr_matrix((ppmi_data, (rows, cols)), shape=(V, V))
sparse.save_npz('embeddings/ppmi_matrix.npz', ppmi)
print(f"PPMI saved. nnz={ppmi.nnz:,}")

# Quick nearest-neighbor check via PPMI
from sklearn.metrics.pairwise import cosine_similarity

def ppmi_neighbors(word, top_n=5):
    idx = word2idx.get(word)
    if idx is None: return []
    vec = ppmi[idx, :].toarray()
    cand_ids = list(range(1, 500))
    cand_mat = ppmi[cand_ids, :].toarray()
    sims = cosine_similarity(vec, cand_mat)[0]
    if idx in cand_ids: sims[cand_ids.index(idx)] = -1
    top_ids = np.argsort(sims)[::-1][:top_n]
    return [(idx2word[cand_ids[i]], float(sims[i])) for i in top_ids]

for w in ['پاکستان', 'حکومت']:
    r = ppmi_neighbors(w)
    print(f"  PPMI neighbors '{w}': {[x[0] for x in r]}")

# ── 1.4-1.5 Skip-gram Word2Vec ────────────────────────────────────────────────
print("\n=== 1.4-1.5 Word2Vec (C3) ===")

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embeddings  = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)

    def forward(self, center, context, neg_samples):
        ce  = self.center_embeddings(center)
        cte = self.context_embeddings(context)
        ne  = self.context_embeddings(neg_samples)
        pos_loss = F.logsigmoid(torch.sum(ce * cte, dim=1))
        neg_loss = F.logsigmoid(-torch.bmm(ne, ce.unsqueeze(2)).squeeze(2)).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_docs, word2idx, window=5):
        self.pairs = []
        for _, tokens in tokenized_docs:
            ids = [word2idx.get(t, 0) for t in tokens]
            for pos, center in enumerate(ids):
                start, end = max(0, pos-window), min(len(ids), pos+window+1)
                for cp in range(start, end):
                    if cp != pos: self.pairs.append((center, ids[cp]))
        self.pairs = torch.tensor(self.pairs, dtype=torch.long)
        print(f"  {len(self.pairs):,} skip-gram pairs")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def build_noise_dist(token_counts, word2idx, power=0.75):
    freq = np.zeros(len(word2idx))
    for tok, cnt in token_counts.items():
        freq[word2idx.get(tok, 0)] += cnt
    freq = freq ** power
    return freq / freq.sum()

EMB_DIM, BATCH, EPOCHS, NEG_K, LR = 100, 512, 7, 10, 0.001

dataset_c3 = SkipGramDataset(tokenized_docs, word2idx)
loader_c3  = torch.utils.data.DataLoader(dataset_c3, batch_size=BATCH, shuffle=True)
noise_c3   = build_noise_dist(token_counts, word2idx)
model_c3   = SkipGramModel(len(word2idx), EMB_DIM).to(device)
opt_c3     = torch.optim.Adam(model_c3.parameters(), lr=LR)

losses_c3 = []
for ep in range(EPOCHS):
    total = 0
    for batch in tqdm(loader_c3, desc=f"C3 ep{ep+1}", leave=False):
        ctr = batch[:,0].to(device); ctx = batch[:,1].to(device)
        neg = torch.tensor(np.random.choice(len(word2idx),(len(batch),NEG_K),p=noise_c3),
                           dtype=torch.long, device=device)
        loss = model_c3(ctr, ctx, neg)
        opt_c3.zero_grad(); loss.backward(); opt_c3.step()
        total += loss.item()
    avg = total/len(loader_c3); losses_c3.append(avg)
    print(f"  C3 ep{ep+1}: {avg:.4f}")

emb_c3 = 0.5*(model_c3.center_embeddings.weight.data.cpu().numpy() +
               model_c3.context_embeddings.weight.data.cpu().numpy())
np.save('embeddings/embeddings_w2v.npy', emb_c3)
print(f"Saved embeddings_w2v.npy shape={emb_c3.shape}")

# ── C2: raw.txt ──────────────────────────────────────────────────────────────
print("\n=== C2: raw.txt ===")
with open('raw.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
raw_split = re.split(r'\n?\[(\d+)\]\n', raw_text)
raw_docs = [(int(raw_split[i]), raw_split[i+1].strip())
            for i in range(1, len(raw_split), 2) if i+1 < len(raw_split)]
def preproc_raw(t):
    t = remove_diacritics(t); t = remove_noise(t)
    t = remove_non_urdu(t); t = normalize_whitespace(t)
    return urdu_tokenize(t)
raw_tok = [(did, preproc_raw(txt)) for did, txt in raw_docs]
all_raw = [t for _, ts in raw_tok for t in ts]
raw_cnts = Counter(all_raw)
raw_vocab = [t for t,_ in raw_cnts.most_common(10000)]
raw_w2i = {'<UNK>': 0}
for i,t in enumerate(raw_vocab): raw_w2i[t] = i+1
raw_i2w = {v:k for k,v in raw_w2i.items()}
print(f"C2 vocab: {len(raw_w2i)}")

ds_c2 = SkipGramDataset(raw_tok, raw_w2i)
ld_c2 = torch.utils.data.DataLoader(ds_c2, batch_size=BATCH, shuffle=True)
nc2   = build_noise_dist(raw_cnts, raw_w2i)
m_c2  = SkipGramModel(len(raw_w2i), EMB_DIM).to(device)
oc2   = torch.optim.Adam(m_c2.parameters(), lr=LR)
losses_c2 = []
for ep in range(EPOCHS):
    total = 0
    for batch in tqdm(ld_c2, desc=f"C2 ep{ep+1}", leave=False):
        ctr = batch[:,0].to(device); ctx = batch[:,1].to(device)
        neg = torch.tensor(np.random.choice(len(raw_w2i),(len(batch),NEG_K),p=nc2),
                           dtype=torch.long, device=device)
        loss = m_c2(ctr, ctx, neg)
        oc2.zero_grad(); loss.backward(); oc2.step(); total += loss.item()
    avg = total/len(ld_c2); losses_c2.append(avg)
    print(f"  C2 ep{ep+1}: {avg:.4f}")
emb_c2 = 0.5*(m_c2.center_embeddings.weight.data.cpu().numpy() +
               m_c2.context_embeddings.weight.data.cpu().numpy())

# ── C4: d=200 ────────────────────────────────────────────────────────────────
print("\n=== C4: d=200 ===")
m_c4  = SkipGramModel(len(word2idx), 200).to(device)
oc4   = torch.optim.Adam(m_c4.parameters(), lr=LR)
losses_c4 = []
for ep in range(EPOCHS):
    total = 0
    for batch in tqdm(loader_c3, desc=f"C4 ep{ep+1}", leave=False):
        ctr = batch[:,0].to(device); ctx = batch[:,1].to(device)
        neg = torch.tensor(np.random.choice(len(word2idx),(len(batch),NEG_K),p=noise_c3),
                           dtype=torch.long, device=device)
        loss = m_c4(ctr, ctx, neg)
        oc4.zero_grad(); loss.backward(); oc4.step(); total += loss.item()
    avg = total/len(loader_c3); losses_c4.append(avg)
    print(f"  C4 ep{ep+1}: {avg:.4f}")
emb_c4 = 0.5*(m_c4.center_embeddings.weight.data.cpu().numpy() +
               m_c4.context_embeddings.weight.data.cpu().numpy())

# ── Evaluation ───────────────────────────────────────────────────────────────
print("\n=== 1.6 Evaluation ===")
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

def w2v_neighbors(word, emb, w2i, i2w, top_n=10):
    idx = w2i.get(word)
    if idx is None:
        for k in w2i:
            if len(word)>3 and word[:4] in k: idx = w2i[k]; word=k; break
    if idx is None: return []
    sims = cos_sim(emb[idx:idx+1], emb)[0]
    sims[idx] = -1
    return [(i2w[i], float(sims[i])) for i in np.argsort(sims)[::-1][:top_n]]

queries = ['پاکستان', 'حکومت', 'عدالت', 'معیشت', 'فوج', 'صحت', 'تعلیم', 'آبادی']
for w in queries:
    nbrs = w2v_neighbors(w, emb_c3, word2idx, idx2word, top_n=5)
    print(f"  {w}: {[n for n,_ in nbrs]}")

# ── MRR comparison ───────────────────────────────────────────────────────────
gold_pairs = [
    ('پاکستان','ملک'),('حکومت','وزیر'),('عدالت','جج'),('فوج','فوجی'),
    ('ملک','ملکی'),('کرکٹ','کھلاڑ'),('بینک','روپ'),('ڈاکٹر','مریض'),
    ('لاہور','پنجاب'),('سیاس','انتخاب'),('میچ','ٹیم'),('پارلیمن','ارکان'),
    ('تجارت','بجٹ'),('تعلیم','اسکول'),('آبادی','شہر'),
    ('مذہب','مسجد'),('سیلاب','بارش'),('کابل','افغانستان'),
    ('انتخاب','ووٹ'),('صحت','ہسپتال')
]

def compute_mrr(emb, w2i, i2w, gold):
    rr = 0
    for src, tgt in gold:
        if src not in w2i: continue
        nbrs = w2v_neighbors(src, emb, w2i, i2w, top_n=20)
        stem = tgt[:4]
        rank = next((i+1 for i,(w,_) in enumerate(nbrs) if stem in w), None)
        rr += (1/rank) if rank else 0
    return rr/len(gold)

from sklearn.decomposition import TruncatedSVD
top_n_ppmi = min(5000, V)
svd = TruncatedSVD(n_components=50, random_state=SEED)
ppmi_svd_full = svd.fit_transform(ppmi[:top_n_ppmi, :].toarray())
ppmi_emb_full = np.zeros((V, 50))
ppmi_emb_full[:top_n_ppmi] = ppmi_svd_full

mrr_c1 = compute_mrr(ppmi_emb_full, word2idx, idx2word, gold_pairs)
mrr_c2 = compute_mrr(emb_c2, raw_w2i, raw_i2w, gold_pairs)
mrr_c3 = compute_mrr(emb_c3, word2idx, idx2word, gold_pairs)
mrr_c4 = compute_mrr(emb_c4, word2idx, idx2word, gold_pairs)

print(f"\n  MRR — C1(PPMI):{mrr_c1:.3f}  C2(raw):{mrr_c2:.3f}  C3(cleaned):{mrr_c3:.3f}  C4(d=200):{mrr_c4:.3f}")

# ── Save training loss plots ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(range(1,EPOCHS+1), losses_c3, label='C3 cleaned d=100', marker='o')
ax.plot(range(1,EPOCHS+1), losses_c2, label='C2 raw d=100', marker='s')
ax.plot(range(1,EPOCHS+1), losses_c4, label='C4 cleaned d=200', marker='^')
ax.set_title('Word2Vec Training Loss Comparison')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('embeddings/w2v_loss_comparison.png', dpi=150)
print("Saved w2v_loss_comparison.png")

# ── t-SNE ────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE
top200_ids = list(range(1, 201))
ppmi_200 = np.array(ppmi[top200_ids, :].todense())
ppmi_svd200 = svd.transform(ppmi_200[:,:ppmi_svd_full.shape[1]])

# Use W2V embeddings for t-SNE (better signal)
emb200 = emb_c3[top200_ids]
tsne = TSNE(n_components=2, random_state=SEED, perplexity=20, max_iter=1000)
coords = tsne.fit_transform(emb200)

colors = ['red']*40 + ['blue']*40 + ['green']*40 + ['orange']*40 + ['purple']*40
cat_info = [('Top 1-40','red'),('Top 41-80','blue'),('Top 81-120','green'),
            ('Top 121-160','orange'),('Top 161-200','purple')]
fig, ax = plt.subplots(figsize=(14,10))
for lbl, col in cat_info:
    mask = [i for i,c in enumerate(colors) if c==col]
    ax.scatter(coords[mask,0], coords[mask,1], c=col, label=lbl, alpha=0.7, s=30)
for i in range(0, 200, 8):
    ax.annotate(idx2word[top200_ids[i]], (coords[i,0], coords[i,1]), fontsize=6, alpha=0.8)
ax.set_title('t-SNE of Top-200 Tokens (Word2Vec C3 Embeddings)')
ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2'); ax.legend()
plt.tight_layout()
plt.savefig('embeddings/tsne_w2v.png', dpi=150)
print("Saved tsne_w2v.png")

print("\n=== Part 1 complete! ===")
print("Files saved:", os.listdir('embeddings'))

# Save state for next parts
import pickle
state = {
    'word2idx': word2idx, 'idx2word': idx2word, 'token_counts': token_counts,
    'tokenized_docs': tokenized_docs, 'documents': documents,
    'emb_c3': emb_c3, 'emb_c4': emb_c4,
    'raw_w2i': raw_w2i, 'raw_i2w': raw_i2w, 'raw_tok': raw_tok,
    'losses_c3': losses_c3, 'losses_c2': losses_c2, 'losses_c4': losses_c4,
    'mrr': {'c1': mrr_c1, 'c2': mrr_c2, 'c3': mrr_c3, 'c4': mrr_c4}
}
with open('embeddings/state_part1.pkl', 'wb') as f:
    pickle.dump(state, f)
print("State saved to embeddings/state_part1.pkl")
