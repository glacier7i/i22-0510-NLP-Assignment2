"""Part 3: Transformer Encoder for topic classification + BiLSTM comparison."""
import sys, os, re, json, pickle, random, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SEED=42; random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load state ────────────────────────────────────────────────────────────────
with open('embeddings/state_part1.pkl','rb') as f: st=pickle.load(f)
word2idx=st['word2idx']; idx2word=st['idx2word']
emb_c3=st['emb_c3']
tokenized_docs=st['tokenized_docs']
topic_labels=st['topic_labels']
print(f"Vocab={len(word2idx)}, Docs={len(tokenized_docs)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
TOPIC_LIST=['سیاست','کھیل','معیشت','عالمی','صحت']
topic2idx={t:i for i,t in enumerate(TOPIC_LIST)}
idx2topic={v:k for k,v in topic2idx.items()}

MAX_LEN=256
PAD=0

class TopicDataset(Dataset):
    def __init__(self, doc_ids, tokenized_docs, topic_labels, word2idx, max_len=MAX_LEN):
        self.samples=[]
        doc_map={did:toks for did,toks in tokenized_docs}
        for did in doc_ids:
            toks=doc_map.get(did,[])
            ids=[word2idx.get(t,0) for t in toks[:max_len]]
            ids+=[PAD]*(max_len-len(ids))
            label=topic2idx.get(topic_labels.get(did,'سیاست'),0)
            length=min(len(toks),max_len)
            self.samples.append((ids,label,length))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        ids,label,ln=self.samples[i]
        return (torch.tensor(ids,dtype=torch.long),
                torch.tensor(label,dtype=torch.long),
                torch.tensor(ln,dtype=torch.long))

# Stratified split: 175 train / 37 val / 38 test
all_doc_ids=[did for did,_ in tokenized_docs]
by_topic={}
for did in all_doc_ids:
    t=topic_labels.get(did,'سیاست')
    by_topic.setdefault(t,[]).append(did)

train_ids,val_ids,test_ids=[],[],[]
for topic,ids in by_topic.items():
    random.shuffle(ids)
    n=len(ids)
    nt=max(1,round(n*0.15))
    nv=max(1,round(n*0.15))
    test_ids+=ids[:nt]
    val_ids+=ids[nt:nt+nv]
    train_ids+=ids[nt+nv:]

print(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
print("Train topic dist:", Counter(topic_labels.get(d,'?') for d in train_ids))

train_ds=TopicDataset(train_ids,tokenized_docs,topic_labels,word2idx)
val_ds  =TopicDataset(val_ids,  tokenized_docs,topic_labels,word2idx)
test_ds =TopicDataset(test_ids, tokenized_docs,topic_labels,word2idx)
train_ld=DataLoader(train_ds,batch_size=16,shuffle=True)
val_ld  =DataLoader(val_ds,  batch_size=16)
test_ld =DataLoader(test_ds, batch_size=16)

# ── Transformer modules (all from scratch) ────────────────────────────────────
class ScaledDotProductAttention(nn.Module):
    def forward(self,Q,K,V,mask=None):
        d_k=Q.size(-1)
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))
        attn=F.softmax(scores,dim=-1)
        attn=torch.nan_to_num(attn,nan=0.0)  # handle -inf→nan
        return torch.matmul(attn,V),attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model=128,num_heads=4):
        super().__init__()
        self.h=num_heads; self.dk=d_model//num_heads
        self.WQ=nn.Linear(d_model,d_model)
        self.WK=nn.Linear(d_model,d_model)
        self.WV=nn.Linear(d_model,d_model)
        self.WO=nn.Linear(d_model,d_model)
        self.attn=ScaledDotProductAttention()
    def forward(self,x,mask=None):
        B,T,_=x.size()
        Q=self.WQ(x).view(B,T,self.h,self.dk).transpose(1,2)
        K=self.WK(x).view(B,T,self.h,self.dk).transpose(1,2)
        V=self.WV(x).view(B,T,self.h,self.dk).transpose(1,2)
        out,aw=self.attn(Q,K,V,mask)
        out=out.transpose(1,2).contiguous().view(B,T,-1)
        return self.WO(out),aw

class PositionWiseFFN(nn.Module):
    def __init__(self,d_model=128,d_ff=512,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=512):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1),:]

class EncoderBlock(nn.Module):
    def __init__(self,d_model=128,num_heads=4,d_ff=512,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.mhsa=MultiHeadSelfAttention(d_model,num_heads)
        self.ffn=PositionWiseFFN(d_model,d_ff,dropout)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        n=self.norm1(x)
        a,aw=self.mhsa(n,mask)
        x=x+self.drop(a)
        x=x+self.drop(self.ffn(self.norm2(x)))
        return x,aw

class TransformerClassifier(nn.Module):
    def __init__(self,vocab_size,d_model=128,num_heads=4,d_ff=512,
                 num_layers=4,num_classes=5,max_len=257,dropout=0.1):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.cls_token=nn.Parameter(torch.randn(1,1,d_model))
        self.pos_enc=SinusoidalPositionalEncoding(d_model,max_len)
        self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([EncoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.classifier=nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,num_classes)
        )
    def forward(self,x,mask=None):
        B=x.size(0)
        emb=self.embedding(x)
        cls=self.cls_token.expand(B,1,-1)
        emb=torch.cat([cls,emb],dim=1)
        emb=self.pos_enc(emb)
        emb=self.drop(emb)
        all_attn=[]
        for blk in self.blocks:
            emb,aw=blk(emb,mask)
            all_attn.append(aw)
        return self.classifier(emb[:,0,:]),all_attn

# ── LR schedule ───────────────────────────────────────────────────────────────
def make_lr_lambda(total_steps,warmup=50):
    def fn(step):
        if step<warmup: return step/max(1,warmup)
        p=(step-warmup)/max(1,total_steps-warmup)
        return 0.5*(1+math.cos(math.pi*p))
    return fn

# ── Training ──────────────────────────────────────────────────────────────────
print("\n=== Training Transformer Classifier ===")
EPOCHS_T=30
model_t=TransformerClassifier(vocab_size=len(word2idx)).to(device)
opt_t=torch.optim.AdamW(model_t.parameters(),lr=5e-4,weight_decay=0.01)
total_steps=EPOCHS_T*len(train_ld)
sched=torch.optim.lr_scheduler.LambdaLR(opt_t,make_lr_lambda(total_steps))

def train_cls(model,loader,opt,sched=None):
    model.train(); tot=0
    for ids,labels,lens in loader:
        ids,labels=ids.to(device),labels.to(device)
        logits,_=model(ids)
        loss=F.cross_entropy(logits,labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        tot+=loss.item()
    return tot/len(loader)

def eval_cls(model,loader):
    model.eval(); preds,trues=[],[]
    with torch.no_grad():
        for ids,labels,lens in loader:
            ids=ids.to(device)
            logits,_=model(ids)
            preds+=logits.argmax(-1).cpu().tolist()
            trues+=labels.tolist()
    acc=(np.array(preds)==np.array(trues)).mean()
    f1=f1_score(trues,preds,average='macro',zero_division=0)
    return acc,f1,preds,trues

t_losses,t_val_acc,t_val_f1=[],[],[]
best_tf1=-1; wait=0
for ep in range(EPOCHS_T):
    loss=train_cls(model_t,train_ld,opt_t,sched)
    acc,f1,_,_=eval_cls(model_t,val_ld)
    t_losses.append(loss); t_val_acc.append(acc); t_val_f1.append(f1)
    print(f"  T ep{ep+1:02d}: loss={loss:.4f} val_acc={acc:.3f} val_f1={f1:.3f}")
    if f1>best_tf1:
        best_tf1=f1; torch.save(model_t.state_dict(),'models/transformer_cls.pt'); wait=0
    else:
        wait+=1
        if wait>=7: print(f"  Early stop ep{ep+1}"); break

model_t.load_state_dict(torch.load('models/transformer_cls.pt'))
t_acc,t_f1,t_preds,t_trues=eval_cls(model_t,test_ld)
print(f"\nTransformer Test: acc={t_acc:.4f} macro-F1={t_f1:.4f}")

# Confusion matrix
TOPIC_LABELS=['سیاست','کھیل','معیشت','عالمی','صحت']
cm=confusion_matrix(t_trues,t_preds,labels=list(range(5)))
fig,ax=plt.subplots(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=TOPIC_LABELS,yticklabels=TOPIC_LABELS,ax=ax,cmap='Oranges')
ax.set_title('Transformer Topic Classification Confusion Matrix')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig('models/transformer_confusion.png',dpi=150)
print("Saved transformer_confusion.png")

# ── Attention heatmaps ────────────────────────────────────────────────────────
print("\nGenerating attention heatmaps...")
model_t.eval()
# Find 3 correctly classified test examples
correct_examples=[]
with torch.no_grad():
    for ids,labels,lens in test_ld:
        ids_dev=ids.to(device)
        logits,all_attn=model_t(ids_dev)
        preds=logits.argmax(-1).cpu()
        for b in range(ids.size(0)):
            if preds[b]==labels[b] and len(correct_examples)<3:
                correct_examples.append({
                    'ids':ids[b].tolist(),
                    'label':labels[b].item(),
                    'attn':all_attn[-1][b].cpu().numpy(),  # last layer
                    'length':lens[b].item()
                })
        if len(correct_examples)>=3: break

for ci,ex in enumerate(correct_examples):
    toks=['[CLS]']+[idx2word.get(i,'<UNK>') for i in ex['ids'][:30]]
    attn=ex['attn'][:,:31,:31]  # 4 heads, 31 tokens
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for hi in range(2):
        sns.heatmap(attn[hi],ax=axes[hi],xticklabels=toks,yticklabels=toks,
                    cmap='viridis',cbar=True)
        axes[hi].set_title(f'Head {hi+1} | Topic: {TOPIC_LABELS[ex["label"]]}')
        axes[hi].tick_params(axis='x',rotation=90); axes[hi].tick_params(axis='y',rotation=0)
    plt.suptitle(f'Attention Heatmap — Example {ci+1}',fontsize=14)
    plt.tight_layout()
    plt.savefig(f'models/attention_heatmap_{ci+1}.png',dpi=120)
    print(f"  Saved attention_heatmap_{ci+1}.png")

# ── BiLSTM Classifier for comparison ─────────────────────────────────────────
print("\n=== BiLSTM Classifier (comparison) ===")

class BiLSTMClassifier(nn.Module):
    def __init__(self,vocab_size,emb_dim,hidden_dim,num_classes,pretrained=None):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.tensor(pretrained,dtype=torch.float32))
        self.lstm=nn.LSTM(emb_dim,hidden_dim,num_layers=2,bidirectional=True,
                          dropout=0.5,batch_first=True)
        self.clf=nn.Sequential(
            nn.Linear(2*hidden_dim,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,num_classes))
    def forward(self,x,lengths):
        emb=self.emb(x)
        packed=nn.utils.rnn.pack_padded_sequence(emb,lengths.cpu().clamp(min=1),
                                                  batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(packed)
        h=torch.cat([h[-2],h[-1]],dim=1)
        return self.clf(h)

model_b=BiLSTMClassifier(len(word2idx),100,128,5,pretrained=emb_c3).to(device)
opt_b=torch.optim.Adam(model_b.parameters(),lr=1e-3)

import time
b_losses,b_val_f1=[],[]
best_bf1=-1; t_start_b=time.time()
for ep in range(EPOCHS_T):
    model_b.train(); tot=0
    for ids,labels,lens in train_ld:
        ids,labels,lens=ids.to(device),labels.to(device),lens.to(device)
        logits=model_b(ids,lens)
        loss=F.cross_entropy(logits,labels)
        opt_b.zero_grad(); loss.backward(); opt_b.step(); tot+=loss.item()
    # val
    model_b.eval(); preds,trues=[],[]
    with torch.no_grad():
        for ids,labels,lens in val_ld:
            ids,lens=ids.to(device),lens.to(device)
            logits=model_b(ids,lens)
            preds+=logits.argmax(-1).cpu().tolist(); trues+=labels.tolist()
    f1=f1_score(trues,preds,average='macro',zero_division=0)
    b_losses.append(tot/len(train_ld)); b_val_f1.append(f1)
    print(f"  B ep{ep+1:02d}: loss={tot/len(train_ld):.4f} val_f1={f1:.3f}")
    if f1>best_bf1: best_bf1=f1; best_b_ep=ep+1

t_end_b=time.time(); b_time=t_end_b-t_start_b

# Test BiLSTM
model_b.eval(); preds_b,trues_b=[],[]
with torch.no_grad():
    for ids,labels,lens in test_ld:
        ids,lens=ids.to(device),lens.to(device)
        logits=model_b(ids,lens)
        preds_b+=logits.argmax(-1).cpu().tolist(); trues_b+=labels.tolist()
b_acc=(np.array(preds_b)==np.array(trues_b)).mean()
b_f1=f1_score(trues_b,preds_b,average='macro',zero_division=0)
print(f"BiLSTM Test: acc={b_acc:.4f} macro-F1={b_f1:.4f}")

# ── Training curves comparison ────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(t_losses,'b-',label='Transformer Loss')
axes[0].plot(b_losses,'r-',label='BiLSTM Loss')
axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(t_val_f1,'b-',label='Transformer Val F1')
axes[1].plot(b_val_f1,'r-',label='BiLSTM Val F1')
axes[1].set_title('Validation F1'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('models/comparison_curves.png',dpi=150)
print("Saved comparison_curves.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"""
=== BiLSTM vs Transformer Comparison ===
                Transformer    BiLSTM
Test Accuracy:  {t_acc:.4f}        {b_acc:.4f}
Macro-F1:       {t_f1:.4f}        {b_f1:.4f}
Converged (ep): {len(t_val_f1)}           {best_b_ep}
""")

# Full classification reports
print("Transformer:")
print(classification_report(t_trues,t_preds,target_names=TOPIC_LABELS,zero_division=0))
print("BiLSTM:")
print(classification_report(trues_b,preds_b,target_names=TOPIC_LABELS,zero_division=0))

# ── Save state ────────────────────────────────────────────────────────────────
st['transformer_results']={'acc':t_acc,'f1':t_f1,'preds':t_preds,'trues':t_trues}
st['bilstm_cls_results']={'acc':b_acc,'f1':b_f1,'preds':preds_b,'trues':trues_b}
st['t_losses']=t_losses; st['t_val_f1']=t_val_f1
st['b_losses']=b_losses; st['b_val_f1']=b_val_f1
with open('embeddings/state_part1.pkl','wb') as f: pickle.dump(st,f)
print("\nState saved. Part 3 COMPLETE!")
print("Models:", os.listdir('models'))
