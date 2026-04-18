"""Part 2.2-2.3: BiLSTM (POS) + BiLSTM-CRF (NER) training & evaluation."""
import sys, os, re, json, pickle, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

SEED=42; random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load state ────────────────────────────────────────────────────────────────
with open('embeddings/state_part1.pkl','rb') as f: st=pickle.load(f)
word2idx=st['word2idx']; idx2word=st['idx2word']
emb_c3=st['emb_c3']
annotated=st['annotated']
train_ids=st['train_ids']; val_ids=st['val_ids']; test_ids=st['test_ids']
print(f"Loaded: {len(annotated)} sentences, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

# ── Build tag vocabularies ────────────────────────────────────────────────────
POS_TAGS = ['NOUN','VERB','ADJ','ADV','PRON','DET','CONJ','POST','NUM','UNK']
NER_TAGS = ['O','B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC']

pos2idx = {t:i for i,t in enumerate(POS_TAGS)}
ner2idx = {t:i for i,t in enumerate(NER_TAGS)}
idx2pos = {v:k for k,v in pos2idx.items()}
idx2ner = {v:k for k,v in ner2idx.items()}

PAD_IDX = 0  # <UNK> as padding
MAX_LEN  = 40

def encode_sentence(sample, pos2idx, ner2idx, word2idx, max_len=MAX_LEN):
    toks = sample['tokens'][:max_len]
    pos  = sample['pos'][:max_len]
    ner  = sample['ner'][:max_len]
    length = len(toks)
    ids  = [word2idx.get(t, 0) for t in toks]
    pos_ids = [pos2idx.get(p, pos2idx['UNK']) for p in pos]
    ner_ids = [ner2idx.get(n, ner2idx['O']) for n in ner]
    # Pad
    pad_len = max_len - length
    ids     += [PAD_IDX]*pad_len
    pos_ids += [0]*pad_len
    ner_ids += [0]*pad_len
    return ids, pos_ids, ner_ids, length

class SeqDataset(Dataset):
    def __init__(self, ids, data):
        self.samples=[(encode_sentence(data[i],pos2idx,ner2idx,word2idx)) for i in ids]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        ids,pos,ner,ln=self.samples[i]
        return (torch.tensor(ids,dtype=torch.long),
                torch.tensor(pos,dtype=torch.long),
                torch.tensor(ner,dtype=torch.long),
                torch.tensor(ln,dtype=torch.long))

train_ds=SeqDataset(train_ids,annotated)
val_ds  =SeqDataset(val_ids,  annotated)
test_ds =SeqDataset(test_ids, annotated)
train_ld=DataLoader(train_ds,batch_size=32,shuffle=True)
val_ld  =DataLoader(val_ds,  batch_size=32)
test_ld =DataLoader(test_ds, batch_size=32)

# ── BiLSTM Tagger (POS) ───────────────────────────────────────────────────────
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained=None, freeze=False):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained,dtype=torch.float32))
        if freeze: self.embedding.weight.requires_grad=False
        self.lstm=nn.LSTM(emb_dim,hidden_dim,num_layers=2,bidirectional=True,
                          dropout=0.5,batch_first=True)
        self.fc=nn.Linear(2*hidden_dim,num_tags)
        self.dropout=nn.Dropout(0.5)

    def forward(self, x, lengths):
        emb=self.dropout(self.embedding(x))
        packed=nn.utils.rnn.pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed)
        out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.size(1))
        return self.fc(out)

# ── CRF from scratch ──────────────────────────────────────────────────────────
class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags=num_tags
        self.transitions=nn.Parameter(torch.randn(num_tags,num_tags))
        # Disallow invalid transitions to/from O arbitrarily — let model learn

    def forward_algorithm(self, emissions, mask):
        B,T,K=emissions.size()
        alpha=emissions[:,0,:]
        for t in range(1,T):
            emit=emissions[:,t,:].unsqueeze(1)           # B,1,K
            trans=self.transitions.unsqueeze(0)           # 1,K,K
            score=alpha.unsqueeze(2)+trans               # B,K,K
            score=score+emit                              # B,K,K
            alpha_new=torch.logsumexp(score,dim=1)        # B,K
            m=mask[:,t].unsqueeze(1).float()
            alpha=alpha_new*m+alpha*(1-m)
        return torch.logsumexp(alpha,dim=1)               # B

    def score_sentence(self, emissions, tags, mask):
        B,T,_=emissions.size()
        score=torch.zeros(B,device=emissions.device)
        for t in range(T):
            m=mask[:,t].float()
            es=emissions[:,t,:].gather(1,tags[:,t].unsqueeze(1)).squeeze(1)
            if t>0:
                ts=self.transitions[tags[:,t-1],tags[:,t]]
                score+=(es+ts)*m
            else:
                score+=es*m
        return score

    def neg_log_likelihood(self, emissions, tags, mask):
        return (self.forward_algorithm(emissions,mask)-self.score_sentence(emissions,tags,mask)).mean()

    def viterbi_decode(self, emissions, mask):
        B,T,K=emissions.size()
        vit=emissions[:,0,:]
        bps=[]
        for t in range(1,T):
            v2=vit.unsqueeze(2)+self.transitions    # B,K,K
            best_scores,best_tags=v2.max(dim=1)     # B,K
            bps.append(best_tags)
            m=mask[:,t].unsqueeze(1).float()
            vit=(best_scores+emissions[:,t,:])*m+vit*(1-m)
        best_last=vit.argmax(dim=1)
        paths=[best_last.unsqueeze(1)]
        for bp in reversed(bps):
            best_last=bp.gather(1,best_last.unsqueeze(1)).squeeze(1)
            paths.append(best_last.unsqueeze(1))
        return torch.cat(paths[::-1],dim=1)

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained=None):
        super().__init__()
        self.bilstm=BiLSTMTagger(vocab_size,emb_dim,hidden_dim,num_tags,pretrained)
        self.crf=CRF(num_tags)

    def forward(self, x, lengths, tags=None, mask=None):
        emissions=self.bilstm(x,lengths)
        if tags is not None:
            return self.crf.neg_log_likelihood(emissions,tags,mask)
        return self.crf.viterbi_decode(emissions,mask)

# ── Training helpers ──────────────────────────────────────────────────────────
def make_mask(x):
    return (x != PAD_IDX).float()

def train_pos(model, loader, optimizer):
    model.train(); total=0
    for ids,pos,ner,lens in loader:
        ids,pos,lens=ids.to(device),pos.to(device),lens.to(device)
        logits=model(ids,lens)           # B,T,num_tags
        mask=make_mask(ids).bool()
        loss=F.cross_entropy(logits[mask],pos[mask.cpu()].to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total+=loss.item()
    return total/len(loader)

def eval_pos(model, loader):
    model.eval(); all_pred,all_true=[],[]
    with torch.no_grad():
        for ids,pos,ner,lens in loader:
            ids,pos,lens=ids.to(device),pos.to(device),lens.to(device)
            logits=model(ids,lens)
            mask=make_mask(ids).bool()
            pred=logits.argmax(-1)
            all_pred+=pred[mask].cpu().tolist()
            all_true+=pos[mask.cpu()].tolist()
    acc=(np.array(all_pred)==np.array(all_true)).mean()
    from sklearn.metrics import f1_score
    f1=f1_score(all_true,all_pred,average='macro',zero_division=0)
    return acc,f1,all_pred,all_true

def train_ner(model, loader, optimizer):
    model.train(); total=0
    for ids,pos,ner,lens in loader:
        ids,ner,lens=ids.to(device),ner.to(device),lens.to(device)
        mask=make_mask(ids).to(device)
        loss=model(ids,lens,tags=ner,mask=mask)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total+=loss.item()
    return total/len(loader)

def eval_ner(model, loader):
    model.eval(); all_pred,all_true=[],[]
    with torch.no_grad():
        for ids,pos,ner,lens in loader:
            ids,ner,lens=ids.to(device),ner.to(device),lens.to(device)
            mask=make_mask(ids).to(device)
            pred=model(ids,lens,mask=mask)    # B,T
            for b in range(ids.size(0)):
                l=lens[b].item()
                all_pred+=pred[b,:l].cpu().tolist()
                all_true+=ner[b,:l].cpu().tolist()
    from sklearn.metrics import f1_score
    f1=f1_score(all_true,all_pred,average='macro',zero_division=0)
    return f1,all_pred,all_true

# ── 1. POS BiLSTM (fine-tuned embeddings) ────────────────────────────────────
print("\n=== Training BiLSTM POS Tagger (fine-tuned) ===")
pos_model_ft=BiLSTMTagger(len(word2idx),100,128,len(POS_TAGS),pretrained=emb_c3,freeze=False).to(device)
opt_pos_ft=torch.optim.Adam(pos_model_ft.parameters(),lr=1e-3,weight_decay=1e-4)

best_val_f1=-1; patience=5; wait=0
pos_train_losses,pos_val_f1s=[],[]
EPOCHS=30
for ep in range(EPOCHS):
    loss=train_pos(pos_model_ft,train_ld,opt_pos_ft)
    acc,f1,_,_=eval_pos(pos_model_ft,val_ld)
    pos_train_losses.append(loss)
    pos_val_f1s.append(f1)
    print(f"  POS ep{ep+1:02d}: loss={loss:.4f} val_acc={acc:.3f} val_f1={f1:.3f}")
    if f1>best_val_f1:
        best_val_f1=f1; torch.save(pos_model_ft.state_dict(),'models/bilstm_pos.pt'); wait=0
    else:
        wait+=1
        if wait>=patience:
            print(f"  Early stop at ep {ep+1}")
            break

pos_model_ft.load_state_dict(torch.load('models/bilstm_pos.pt'))
test_acc,test_f1,pos_pred,pos_true=eval_pos(pos_model_ft,test_ld)
print(f"\nPOS Test: acc={test_acc:.4f}  macro-F1={test_f1:.4f}")

# Confusion matrix
cm=confusion_matrix(pos_true,pos_pred,labels=list(range(len(POS_TAGS))))
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=POS_TAGS,yticklabels=POS_TAGS,ax=ax,cmap='Blues')
ax.set_title('POS Tagging Confusion Matrix (BiLSTM Fine-tuned)')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig('models/pos_confusion_matrix.png',dpi=150)
print("Saved pos_confusion_matrix.png")

# Classification report
print("\nPOS Classification Report:")
print(classification_report(pos_true,pos_pred,target_names=POS_TAGS,zero_division=0))

# ── 2. POS BiLSTM (frozen embeddings) ────────────────────────────────────────
print("\n=== BiLSTM POS (frozen embeddings) ===")
pos_model_fr=BiLSTMTagger(len(word2idx),100,128,len(POS_TAGS),pretrained=emb_c3,freeze=True).to(device)
opt_pos_fr=torch.optim.Adam(filter(lambda p:p.requires_grad,pos_model_fr.parameters()),lr=1e-3)
best_fr=-1
for ep in range(20):
    loss=train_pos(pos_model_fr,train_ld,opt_pos_fr)
    acc,f1,_,_=eval_pos(pos_model_fr,val_ld)
    if f1>best_fr: best_fr=f1
    if ep>=9 and f1<0.05: break
acc_fr,f1_fr,_,_=eval_pos(pos_model_fr,test_ld)
print(f"POS Frozen: test_acc={acc_fr:.4f} macro-F1={f1_fr:.4f}")
print(f"Comparison — Fine-tuned: {test_f1:.4f} vs Frozen: {f1_fr:.4f}")

# ── 3. NER BiLSTM-CRF ────────────────────────────────────────────────────────
print("\n=== Training BiLSTM-CRF NER ===")
ner_model=BiLSTM_CRF(len(word2idx),100,128,len(NER_TAGS),pretrained=emb_c3).to(device)
opt_ner=torch.optim.Adam(ner_model.parameters(),lr=1e-3,weight_decay=1e-4)

best_ner_f1=-1; ner_losses,ner_val_f1s=[],[]
for ep in range(30):
    loss=train_ner(ner_model,train_ld,opt_ner)
    f1,_,_=eval_ner(ner_model,val_ld)
    ner_losses.append(loss); ner_val_f1s.append(f1)
    print(f"  NER ep{ep+1:02d}: loss={loss:.4f} val_f1={f1:.4f}")
    if f1>best_ner_f1:
        best_ner_f1=f1; torch.save(ner_model.state_dict(),'models/bilstm_ner.pt'); wait=0
    else:
        wait+=1
        if wait>=5: print(f"  Early stop at ep {ep+1}"); break

ner_model.load_state_dict(torch.load('models/bilstm_ner.pt'))
ner_f1,ner_pred,ner_true=eval_ner(ner_model,test_ld)
print(f"\nNER Test macro-F1={ner_f1:.4f}")
print("\nNER Report:")
print(classification_report(ner_true,ner_pred,target_names=NER_TAGS,zero_division=0))

# ── 4. NER without CRF (softmax baseline) ───────────────────────────────────
print("\n=== NER without CRF (softmax) ===")
ner_nocrf=BiLSTMTagger(len(word2idx),100,128,len(NER_TAGS),pretrained=emb_c3).to(device)
opt_nocrf=torch.optim.Adam(ner_nocrf.parameters(),lr=1e-3)
best_nocrf=-1
for ep in range(20):
    ner_nocrf.train(); tot=0
    for ids,pos,ner,lens in train_ld:
        ids,ner,lens=ids.to(device),ner.to(device),lens.to(device)
        logits=ner_nocrf(ids,lens)
        mask=make_mask(ids).bool()
        loss=F.cross_entropy(logits[mask],ner[mask.cpu()].to(device))
        opt_nocrf.zero_grad(); loss.backward(); opt_nocrf.step(); tot+=loss.item()
    # eval
    ner_nocrf.eval(); pred2,true2=[],[]
    with torch.no_grad():
        for ids,pos,ner,lens in val_ld:
            ids,ner,lens=ids.to(device),ner.to(device),lens.to(device)
            logits=ner_nocrf(ids,lens)
            mask=make_mask(ids).bool()
            pred2+=logits.argmax(-1)[mask].cpu().tolist()
            true2+=ner[mask.cpu()].tolist()
    from sklearn.metrics import f1_score
    f1v=f1_score(true2,pred2,average='macro',zero_division=0)
    if f1v>best_nocrf: best_nocrf=f1v

# test nocrf
ner_nocrf.eval(); pred3,true3=[],[]
with torch.no_grad():
    for ids,pos,ner,lens in test_ld:
        ids,ner,lens=ids.to(device),ner.to(device),lens.to(device)
        logits=ner_nocrf(ids,lens)
        mask=make_mask(ids).bool()
        pred3+=logits.argmax(-1)[mask].cpu().tolist()
        true3+=ner[mask.cpu()].tolist()
from sklearn.metrics import f1_score
f1_nocrf=f1_score(true3,pred3,average='macro',zero_division=0)
print(f"NER with CRF: {ner_f1:.4f}  without CRF: {f1_nocrf:.4f}")

# ── Training curves ───────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(pos_train_losses,label='POS Loss'); axes[0].plot(pos_val_f1s,label='POS Val F1')
axes[0].set_title('BiLSTM POS Training'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ner_losses,label='NER Loss'); axes[1].plot(ner_val_f1s,label='NER Val F1')
axes[1].set_title('BiLSTM-CRF NER Training'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('models/bilstm_training_curves.png',dpi=150)
print("Saved bilstm_training_curves.png")

# ── Ablation study ────────────────────────────────────────────────────────────
print("\n=== Ablation Study ===")
ablation_results = {}

# A1: Unidirectional LSTM
class UniLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained=None):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained,dtype=torch.float32))
        self.lstm=nn.LSTM(emb_dim,hidden_dim,num_layers=2,bidirectional=False,
                          dropout=0.3,batch_first=True)
        self.fc=nn.Linear(hidden_dim,num_tags)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x,lengths):
        emb=self.dropout(self.embedding(x))
        packed=nn.utils.rnn.pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed)
        out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.size(1))
        return self.fc(out)

a1_model=UniLSTMTagger(len(word2idx),100,128,len(POS_TAGS),pretrained=emb_c3).to(device)
opt_a1=torch.optim.Adam(a1_model.parameters(),lr=1e-3)
for ep in range(15):
    train_pos(a1_model,train_ld,opt_a1)
_,a1_f1,_,_=eval_pos(a1_model,test_ld)
ablation_results['A1: Uni-LSTM']=a1_f1
print(f"A1 (uni-LSTM) POS F1={a1_f1:.4f}")

# A2: No dropout
class BiLSTMNoDropout(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained=None):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained,dtype=torch.float32))
        self.lstm=nn.LSTM(emb_dim,hidden_dim,num_layers=2,bidirectional=True,batch_first=True)
        self.fc=nn.Linear(2*hidden_dim,num_tags)
    def forward(self,x,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(x),lengths.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed)
        out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.size(1))
        return self.fc(out)

a2_model=BiLSTMNoDropout(len(word2idx),100,128,len(POS_TAGS),pretrained=emb_c3).to(device)
opt_a2=torch.optim.Adam(a2_model.parameters(),lr=1e-3)
for ep in range(15):
    train_pos(a2_model,train_ld,opt_a2)
_,a2_f1,_,_=eval_pos(a2_model,test_ld)
ablation_results['A2: No Dropout']=a2_f1
print(f"A2 (no dropout) POS F1={a2_f1:.4f}")

# A3: Random embeddings
a3_model=BiLSTMTagger(len(word2idx),100,128,len(POS_TAGS),pretrained=None).to(device)
opt_a3=torch.optim.Adam(a3_model.parameters(),lr=1e-3)
for ep in range(15):
    train_pos(a3_model,train_ld,opt_a3)
_,a3_f1,_,_=eval_pos(a3_model,test_ld)
ablation_results['A3: Random Emb']=a3_f1
print(f"A3 (random emb) POS F1={a3_f1:.4f}")

# A4: Softmax NER
ablation_results['A4: NER Softmax']=f1_nocrf
ablation_results['BiLSTM+CRF NER']=ner_f1
ablation_results['BiLSTM POS (ft)']=test_f1
ablation_results['BiLSTM POS (frozen)']=f1_fr

print("\nAblation Summary:")
for k,v in ablation_results.items():
    print(f"  {k}: {v:.4f}")

# Save state
st['pos_pred']=pos_pred; st['pos_true']=pos_true
st['ner_pred']=ner_pred; st['ner_true']=ner_true
st['ablation']=ablation_results
st['pos_train_losses']=pos_train_losses; st['pos_val_f1s']=pos_val_f1s
st['ner_losses']=ner_losses; st['ner_val_f1s']=ner_val_f1s
with open('embeddings/state_part1.pkl','wb') as f: pickle.dump(st,f)
print("\nState saved. Part 2 training COMPLETE!")
print("Models saved:", os.listdir('models'))
