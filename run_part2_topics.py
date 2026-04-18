"""Part 3.1 first (per plan): Assign topic labels to 250 articles."""
import sys, os, re, json, pickle, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import Counter

SEED=42; random.seed(SEED); np.random.seed(SEED)

# Load state
with open('embeddings/state_part1.pkl','rb') as f: st=pickle.load(f)
word2idx=st['word2idx']; idx2word=st['idx2word']
tokenized_docs=st['tokenized_docs']; documents=st['documents']

with open('metadata.json','r',encoding='utf-8') as f: meta=json.load(f)

# ── Keyword-based topic classifier ───────────────────────────────────────────
# Keywords are stemmed/lemmatized forms that appear in cleaned.txt
TOPICS = {
    'سیاست': ['انتخاب','حکوم','وزیر','پارلیمن','سیاس','جماعت','ووٹ','ارکان','حزب',
               'اسمبل','صدر','وزارت','سینیٹ','ایم','رہنما','قائد','نمائند','اپوزیش',
               'اقتدار','الیکش','بجٹ','ریاست','وفاق','صوب'],
    'کھیل':  ['کرکٹ','میچ','ٹیم','کھلاڑ','سکور','ورلڈ','کپ','پی ایس ایل','بیٹنگ','باؤل',
               'ٹیسٹ','وکٹ','رن','اووز','چیمپین','ٹورنامنٹ','کوچ','فٹبال','کھیل',
               'اتھلیٹ','اولمپک','گول','ریس','انٹرنیشنل','ایشیا'],
    'معیشت': ['مہنگائ','تجارت','بینک','بجٹ','بجل','معیش','روپ','ڈالر','سرمای',
               'ٹیکس','افراط','کاروبار','مارکیٹ','برآمد','درآمد','قرض','آئی ایم',
               'توانائ','گیس','پٹرول','فیکٹر','صنعت','زراعت','سولر','ترقی'],
    'عالمی': ['اقوام','معاہد','تنازع','ایران','امریک','روس','یوکرین','چین','افغانستان',
               'جنگ','عالمی','بین','سفارت','ناٹو','اقتصادی','یورپ','مشرق','وسط',
               'اسرائیل','فلسطین','برطانی','فرانس','جاپان','جرمن','ترک','عرب'],
    'صحت':   ['ہسپتال','بیمار','ویکسین','سیلاب','تعلیم','کینسر','صحت','ماحول',
               'خواتین','آب','سیلاب','زلزل','قدرتی','آفت','معاشر','انسانی','بچ',
               'عورت','خاندان','غریب','مدرس','اسکول','یونیورس','ڈاکٹر','نرس']
}

def classify_article(doc_id, tokens):
    """Assign topic based on keyword hits in title + content."""
    # Get title keywords
    title = meta.get(str(doc_id), {}).get('title', '')

    scores = {t: 0 for t in TOPICS}
    # Check content tokens
    tok_set = tokens[:200]  # first 200 tokens
    for topic, kws in TOPICS.items():
        for kw in kws:
            # Check in tokens (partial match for stemmed forms)
            for tok in tok_set:
                if kw in tok or tok in kw:
                    scores[topic] += 1
            # Check in title
            if kw in title:
                scores[topic] += 3  # title hit worth more

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        # Fallback: assign based on metadata title
        for topic, kws in TOPICS.items():
            for kw in kws:
                if kw in title:
                    return topic
        return 'سیاست'  # default fallback
    return best

# Classify all 250 articles
labels = {}
for doc_id, doc_text in documents:
    tokens = doc_text.split()
    labels[doc_id] = classify_article(doc_id, tokens)

label_counts = Counter(labels.values())
print("Topic distribution:")
for topic, cnt in label_counts.most_common():
    print(f"  {topic}: {cnt}")

# Balance: if any class < 30, reassign borderline articles
def rebalance(labels, tokenized_docs, min_count=35):
    counts = Counter(labels.values())
    # Find underrepresented topics
    under = [t for t,c in counts.items() if c < min_count]
    if not under:
        return labels

    # For borderline articles (score difference < 3), allow reassignment
    new_labels = dict(labels)
    for doc_id, doc_text in documents:
        tokens = doc_text.split()
        scores = {t: 0 for t in TOPICS}
        for topic, kws in TOPICS.items():
            for kw in kws:
                for tok in tokens[:200]:
                    if kw in tok or tok in kw: scores[topic] += 1

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) >= 2:
            top1, top2 = sorted_scores[0], sorted_scores[1]
            # If top2 is underrepresented and score close to top1
            if top2[0] in under and top1[1] - top2[1] < 5:
                current = new_labels[doc_id]
                if counts.get(current,0) > min_count + 10:
                    new_labels[doc_id] = top2[0]
                    counts[current] -= 1
                    counts[top2[0]] = counts.get(top2[0],0) + 1
    return new_labels

labels = rebalance(labels, tokenized_docs, min_count=35)
label_counts = Counter(labels.values())
print("\nAfter rebalancing:")
for topic, cnt in label_counts.most_common():
    print(f"  {topic}: {cnt}")

# Save labels
with open('embeddings/topic_labels.json','w',encoding='utf-8') as f:
    json.dump({str(k): v for k,v in labels.items()}, f, ensure_ascii=False, indent=2)
print("Saved embeddings/topic_labels.json")

# Save extended state
st['topic_labels'] = labels
st['label_counts'] = dict(label_counts)
with open('embeddings/state_part1.pkl','wb') as f: pickle.dump(st,f)
print("Updated state_part1.pkl with topic labels")

# Show sample articles per topic
print("\nSample articles per topic:")
for topic in TOPICS:
    arts = [did for did, lbl in labels.items() if lbl == topic][:3]
    for did in arts:
        title = meta.get(str(did), {}).get('title', '')[:40]
        print(f"  [{topic}] Doc {did}: {title}")
