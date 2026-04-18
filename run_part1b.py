"""Part 1b: C2+C4 training + evaluation (C3 already done)."""
import sys, os, re, json, math, random, warnings, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load existing artifacts ───────────────────────────────────────────────────
word2idx = json.load(open('embeddings/word2idx.json', encoding='utf-8'))
idx2word = {v:k for k,v in word2idx.items()}
emb_c3   = np.load('embeddings/embeddings_w2v.npy')
ppmi     = sparse.load_npz('embeddings/ppmi_matrix.npz')
print(f"Loaded: vocab={len(word2idx)}, emb_c3={emb_c3.shape}, ppmi={ppmi.shape}")

# Rebuild tokenized_docs (needed for C3 dataloader reuse and C4)
with open('cleaned.txt','r',encoding='utf-8') as f: cleaned_text = f.read()
docs_raw = re.split(r'\n?\[(\d+)\]\n', cleaned_text)
documents = [(int(docs_raw[i]), docs_raw[i+1].strip()) for i in range(1,len(docs_raw),2) if i+1<len(docs_raw)]
tokenized_docs = [(did, txt.split()) for did,txt in documents]
all_tokens = [t for _,toks in tokenized_docs for t in toks]
token_counts = Counter(all_tokens)
print(f"Corpus: {len(documents)} docs, {len(all_tokens):,} tokens")

# ── A1 preprocessing (for C2) ────────────────────────────────────────────────
def remove_diacritics(t): return re.sub(r'[\u064B-\u065F]','',t)
def remove_noise(t):
    t = re.sub(r'http\S+|www\S+','',t)
    return re.sub(r'[^\u0600-\u06FF\s\u06F0-\u06F9\u0660-\u0669\u060C\u061B\u061F\u06D4]','',t)
def remove_non_urdu(t): return re.sub(r'[^\u0600-\u06FF\s\u06F0-\u06F9\u0660-\u0669]',' ',t)
def normalize_whitespace(t): return re.sub(r'\s+',' ',t).strip()
def urdu_tokenize(t):
    t = re.sub(r'[\u06F0-\u06F9]+','<NUM>',t)
    t = re.sub(r'[0-9]+','<NUM>',t)
    t = re.sub(r'[\u06D4\u061F!\u060C\u061B\u066A,;:\[\]{}()]','',t)
    return [x.strip() for x in t.split() if x.strip()]

# ── SkipGram classes ──────────────────────────────────────────────────────────
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.C = nn.Embedding(vocab_size, dim)
        self.U = nn.Embedding(vocab_size, dim)
        self.C.weight.data.uniform_(-0.5/dim, 0.5/dim)
        self.U.weight.data.uniform_(-0.5/dim, 0.5/dim)
    def forward(self, ctr, ctx, neg):
        ce=self.C(ctr); ue=self.U(ctx); ne=self.U(neg)
        pl=F.logsigmoid(torch.sum(ce*ue,1))
        nl=F.logsigmoid(-torch.bmm(ne,ce.unsqueeze(2)).squeeze(2)).sum(1)
        return -(pl+nl).mean()

class SGDataset(torch.utils.data.Dataset):
    def __init__(self, tok_docs, w2i, window=5):
        pairs=[]
        for _,toks in tok_docs:
            ids=[w2i.get(t,0) for t in toks]
            for p,c in enumerate(ids):
                for cp in range(max(0,p-window),min(len(ids),p+window+1)):
                    if cp!=p: pairs.append((c,ids[cp]))
        self.p=torch.tensor(pairs,dtype=torch.long)
        print(f"  {len(self.p):,} pairs")
    def __len__(self): return len(self.p)
    def __getitem__(self,i): return self.p[i]

def noise_dist(counts, w2i, pw=0.75):
    f=np.zeros(len(w2i))
    for t,c in counts.items(): f[w2i.get(t,0)]+=c
    f=f**pw; return f/f.sum()

def train_sg(ds, w2i, dim, epochs=7, batch=512, lr=0.001, tag=''):
    ld=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=True)
    nd=noise_dist(Counter([idx2word.get(i,'<UNK>') for i in range(len(w2i))]),w2i)
    # rebuild noise from actual counts
    if tag=='C2':
        all_raw_toks=[t for _,ts in raw_tok for t in ts]
        nd=noise_dist(Counter(all_raw_toks), w2i)
    else:
        nd=noise_dist(token_counts, word2idx)
    m=SkipGramModel(len(w2i),dim).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    losses=[]
    for ep in range(epochs):
        tot=0
        for b in tqdm(ld,desc=f"{tag} ep{ep+1}",leave=False):
            ctr=b[:,0].to(device); ctx=b[:,1].to(device)
            neg=torch.tensor(np.random.choice(len(w2i),(len(b),10),p=nd),dtype=torch.long,device=device)
            loss=m(ctr,ctx,neg); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        avg=tot/len(ld); losses.append(avg); print(f"  {tag} ep{ep+1}: {avg:.4f}")
    emb=0.5*(m.C.weight.data.cpu().numpy()+m.U.weight.data.cpu().numpy())
    return emb, losses

# ── C2: raw.txt ──────────────────────────────────────────────────────────────
print("\n=== C2: raw.txt preprocessing ===")
with open('raw.txt','r',encoding='utf-8') as f: raw_text=f.read()
raw_split=re.split(r'\n?\[(\d+)\]\n',raw_text)
raw_docs=[(int(raw_split[i]),raw_split[i+1].strip()) for i in range(1,len(raw_split),2) if i+1<len(raw_split)]
def preproc(t):
    t=remove_diacritics(t); t=remove_noise(t); t=remove_non_urdu(t)
    return urdu_tokenize(normalize_whitespace(t))
raw_tok=[(did,preproc(txt)) for did,txt in raw_docs]
all_raw=[t for _,ts in raw_tok for t in ts]
raw_cnts=Counter(all_raw)
raw_vocab=[t for t,_ in raw_cnts.most_common(10000)]
raw_w2i={'<UNK>':0}
for i,t in enumerate(raw_vocab): raw_w2i[t]=i+1
raw_i2w={v:k for k,v in raw_w2i.items()}
print(f"C2 vocab={len(raw_w2i)}, tokens={len(all_raw):,}")

print("\n=== C2 Training ===")
ds_c2=SGDataset(raw_tok, raw_w2i)
emb_c2, losses_c2 = train_sg(ds_c2, raw_w2i, 100, tag='C2')
print(f"C2 done: {emb_c2.shape}")

# ── C4: cleaned d=200 ────────────────────────────────────────────────────────
print("\n=== C4: cleaned d=200 ===")
ds_c3=SGDataset(tokenized_docs, word2idx)
emb_c4, losses_c4 = train_sg(ds_c3, word2idx, 200, tag='C4')
print(f"C4 done: {emb_c4.shape}")

# ── Evaluation ───────────────────────────────────────────────────────────────
print("\n=== Evaluation ===")

def neighbors(word, emb, w2i, i2w, n=10):
    idx=w2i.get(word)
    if idx is None:
        for k in w2i:
            if len(word)>3 and word[:4] in k: idx=w2i[k]; word=k; break
    if idx is None: return []
    sims=cos_sim(emb[idx:idx+1],emb)[0]; sims[idx]=-1
    return [(i2w[i],float(sims[i])) for i in np.argsort(sims)[::-1][:n]]

queries=['پاکستان','حکومت','عدالت','معیشت','فوج','صحت','تعلیم','آبادی']
print("\n--- C3 Top-10 Neighbors ---")
for w in queries:
    nbrs=neighbors(w,emb_c3,word2idx,idx2word,n=10)
    ns=[n for n,_ in nbrs]
    print(f"  {w}: {ns[:5]}")

# Analogy tests
def analogy(a,b,c,emb,w2i,i2w,n=3):
    for x in [a,b,c]:
        if x not in w2i: return []
    q=emb[w2i[b]]-emb[w2i[a]]+emb[w2i[c]]
    sims=cos_sim(q.reshape(1,-1),emb)[0]
    for x in [a,b,c]: sims[w2i[x]]=-1
    return [(i2w[i],float(sims[i])) for i in np.argsort(sims)[::-1][:n]]

analogy_tests=[
    ('پاکستان','اسلام','انڈیا'),
    ('مرد','عورت','لڑکا'),
    ('لاہور','پنجاب','کراچی'),
    ('وزیر','حکومت','جج'),
    ('فوج','جنگ','پولیس'),
    ('ملک','صدر','شہر'),
    ('بینک','روپ','اسکول'),
    ('کرکٹ','بیٹنگ','فٹبال'),
    ('ڈاکٹر','ہسپتال','استاد'),
    ('لندن','برطانی','واشنگٹن'),
]
print("\n--- Analogies (C3) ---")
for a,b,c in analogy_tests:
    r=analogy(a,b,c,emb_c3,word2idx,idx2word)
    ans=[x for x,_ in r]
    print(f"  {a}:{b}::{c}:? -> {ans}")

# MRR
gold_pairs=[
    ('پاکستان','ملک'),('حکومت','وزیر'),('عدالت','جج'),('فوج','فوجی'),
    ('ملک','ملکی'),('کرکٹ','کھلاڑ'),('بینک','روپ'),('ڈاکٹر','مریض'),
    ('لاہور','پنجاب'),('سیاس','انتخاب'),('میچ','ٹیم'),('پارلیمن','ارکان'),
    ('تجارت','بجٹ'),('تعلیم','اسکول'),('آبادی','شہر'),
    ('مذہب','مسجد'),('سیلاب','بارش'),('کابل','افغانستان'),
    ('انتخاب','ووٹ'),('صحت','ہسپتال')
]

def mrr(emb, w2i, i2w, gold):
    rr=0
    for src,tgt in gold:
        if src not in w2i: continue
        nbrs=neighbors(src,emb,w2i,i2w,n=20)
        stem=tgt[:4]
        rank=next((i+1 for i,(w,_) in enumerate(nbrs) if stem in w),None)
        rr+=(1/rank) if rank else 0
    return rr/len(gold)

# C1 PPMI (SVD to 50d)
svd=TruncatedSVD(n_components=50,random_state=SEED)
ppmi_arr=ppmi[:min(5000,len(word2idx)),:].toarray()
ppmi_svd=svd.fit_transform(ppmi_arr)
ppmi_emb=np.zeros((len(word2idx),50))
ppmi_emb[:ppmi_arr.shape[0]]=ppmi_svd

m1=mrr(ppmi_emb,word2idx,idx2word,gold_pairs)
m2=mrr(emb_c2,raw_w2i,raw_i2w,gold_pairs)
m3=mrr(emb_c3,word2idx,idx2word,gold_pairs)
m4=mrr(emb_c4,word2idx,idx2word,gold_pairs)
print(f"\nMRR — C1:{m1:.3f} C2:{m2:.3f} C3:{m3:.3f} C4:{m4:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
# Loss curves
losses_c3_dummy=[0.0]*7  # placeholder if needed
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(range(1,8),losses_c2,marker='s',label='C2 raw d=100')
ax.plot(range(1,8),losses_c4,marker='^',label='C4 cleaned d=200')
ax.set_title('Word2Vec Training Loss (C2 & C4)'); ax.set_xlabel('Epoch')
ax.set_ylabel('Loss'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('embeddings/w2v_loss_c2_c4.png',dpi=150)
print("Saved w2v_loss_c2_c4.png")

# t-SNE on C3
from sklearn.manifold import TSNE
top200=[i for i in range(1,201)]
tsne=TSNE(n_components=2,random_state=SEED,perplexity=20,max_iter=1000)
coords=tsne.fit_transform(emb_c3[top200])
cols=['red']*40+['blue']*40+['green']*40+['orange']*40+['purple']*40
cat=[('Top 1-40','red'),('Top 41-80','blue'),('Top 81-120','green'),
     ('Top 121-160','orange'),('Top 161-200','purple')]
fig,ax=plt.subplots(figsize=(14,10))
for lbl,col in cat:
    mask=[i for i,c in enumerate(cols) if c==col]
    ax.scatter(coords[mask,0],coords[mask,1],c=col,label=lbl,alpha=0.7,s=30)
for i in range(0,200,10):
    try: ax.annotate(idx2word[top200[i]],(coords[i,0],coords[i,1]),fontsize=6,alpha=0.8)
    except: pass
ax.set_title('t-SNE of Top-200 Tokens (Word2Vec C3)'); ax.set_xlabel('t-SNE Dim 1')
ax.set_ylabel('t-SNE Dim 2'); ax.legend(); plt.tight_layout()
plt.savefig('embeddings/tsne_w2v_c3.png',dpi=150); print("Saved tsne_w2v_c3.png")

# ── Save state ────────────────────────────────────────────────────────────────
state={
    'word2idx':word2idx,'idx2word':idx2word,'token_counts':dict(token_counts),
    'tokenized_docs':tokenized_docs,'documents':documents,
    'emb_c3':emb_c3,'emb_c4':emb_c4,'emb_c2':emb_c2,
    'raw_w2i':raw_w2i,'raw_i2w':raw_i2w,'raw_tok':raw_tok,
    'losses_c2':losses_c2,'losses_c4':losses_c4,
    'mrr':{'c1':m1,'c2':m2,'c3':m3,'c4':m4}
}
with open('embeddings/state_part1.pkl','wb') as f: pickle.dump(state,f)
print("State saved → embeddings/state_part1.pkl")
print("\n=== Part 1 COMPLETE ===")
print("embeddings/:", sorted(os.listdir('embeddings')))
