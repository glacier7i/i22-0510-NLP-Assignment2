"""Part 2.1: Select 500 sentences, annotate POS + NER, save CoNLL files."""
import sys, os, re, json, pickle, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import Counter, defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED)

with open('embeddings/state_part1.pkl','rb') as f: st=pickle.load(f)
word2idx=st['word2idx']; idx2word=st['idx2word']
topic_labels=st['topic_labels']
with open('metadata.json','r',encoding='utf-8') as f: meta=json.load(f)

# ── Sentence segmentation ────────────────────────────────────────────────────
def segment_sentences(text):
    text = re.sub(r'([\u06D4\u061F!])', r'\1\n', text)
    return [s.strip() for s in text.split('\n') if s.strip() and len(s.strip()) > 10]

def urdu_tokenize_raw(text):
    text = re.sub(r'[\u06F0-\u06F9]+', '<NUM>', text)
    text = re.sub(r'[0-9]+', '<NUM>', text)
    text = re.sub(r'[\u06D4\u061F!\u060C\u061B\u066A,;:\[\]{}()]', '', text)
    return [t.strip() for t in text.split() if t.strip()]

# Parse raw.txt for sentences
with open('raw.txt','r',encoding='utf-8') as f: raw_text=f.read()
raw_split = re.split(r'\n?\[(\d+)\]\n', raw_text)
raw_docs = {}
for i in range(1, len(raw_split), 2):
    if i+1 < len(raw_split):
        raw_docs[int(raw_split[i])] = raw_split[i+1].strip()

# Collect sentences per topic
sentences_by_topic = defaultdict(list)
for doc_id, article_text in raw_docs.items():
    topic = topic_labels.get(doc_id, 'سیاست')
    sents = segment_sentences(article_text)
    for s in sents:
        toks = urdu_tokenize_raw(s)
        if 5 <= len(toks) <= 40:  # reasonable length
            sentences_by_topic[topic].append((doc_id, toks))

print("Sentences per topic (before selection):")
for t,ss in sentences_by_topic.items():
    print(f"  {t}: {len(ss)}")

# Select 500 sentences: at least 100 from 3 main topics
# Strategy: 120 سیاست, 110 کھیل, 110 عالمی, 90 صحت, 70 معیشت = 500
target = {'سیاست':120, 'کھیل':110, 'عالمی':110, 'صحت':90, 'معیشت':70}
selected = []
for topic, n in target.items():
    pool = sentences_by_topic[topic]
    random.shuffle(pool)
    selected_n = pool[:n]
    for doc_id, toks in selected_n:
        selected.append((doc_id, topic, toks))

random.shuffle(selected)
print(f"\nSelected {len(selected)} sentences")
topic_dist = Counter(t for _,t,_ in selected)
print("Distribution:", dict(topic_dist))

# ── POS Lexicon (≥200 entries per major class) ────────────────────────────────
pos_lexicon = {}

# NOUN (≥200)
nouns = [
    'پاکستان','حکومت','ملک','عدالت','شہر','لوگ','پانی','بچے','وقت','سال',
    'طرف','حکام','صوبہ','ریاست','وزارت','اسمبلی','پارلیمنٹ','سینیٹ',
    'ووٹ','انتخاب','الیکشن','جماعت','پارٹی','رہنما','قائد','نمائندہ',
    'صدر','وزیر','وزیراعظم','گورنر','چیف','جسٹس','جج','جیل','عدالت',
    'فیصلہ','مقدمہ','تحقیقات','رپورٹ','خبر','میڈیا','چینل','اخبار',
    'انٹرویو','بیان','اعلان','تقریر','پریس','کانفرنس','اجلاس','اجتماع',
    'میٹنگ','بیٹھک','مذاکرات','معاہدہ','سمجھوتہ','امن','جنگ','لڑائی',
    'فوج','فوجی','آرمی','نیوی','پولیس','ادارہ','محکمہ','وزارت','بیورو',
    'کرکٹ','کھیل','میچ','ٹیم','کھلاڑی','کپتان','کوچ','سکور','وکٹ',
    'رن','اوور','باؤنڈری','سینچری','ٹیسٹ','ٹورنامنٹ','سیریز','لیگ',
    'پی ایس ایل','ورلڈ کپ','چیمپین','ٹرافی','تمغہ','ایوارڈ','ریکارڈ',
    'اسپتال','ڈاکٹر','مریض','علاج','دوا','ویکسین','بیماری','کینسر',
    'ذیابیطس','بخار','آپریشن','سرجری','ڈینگی','ملیریا','وبا','صحت',
    'اسکول','کالج','یونیورسٹی','طالب علم','استاد','تعلیم','نصاب',
    'کتاب','امتحان','ڈگری','اسکالرشپ','وظیفہ','تدریس','مدرسہ',
    'بینک','روپیہ','ڈالر','پیسہ','قرض','سود','سرمایہ','تجارت',
    'بجٹ','ٹیکس','محصول','مہنگائی','افراط زر','برآمد','درآمد',
    'گیس','بجلی','پٹرول','تیل','توانائی','سولر','پنل','لوڈ شیڈنگ',
    'ماحول','آلودگی','پانی','سیلاب','زلزلہ','بارش','خشک سالی',
    'خاندان','والدین','ماں','باپ','بھائی','بہن','بچہ','بیٹا','بیٹی',
    'شوہر','بیوی','دادا','نانا','چچا','ماموں','خالہ','پھوپھی',
    'گھر','مکان','کمرہ','دروازہ','کھڑکی','چھت','دیوار','باورچی خانہ',
    'سڑک','راستہ','گلی','محلہ','علاقہ','ضلع','تحصیل','موزہ','دیہات',
    'کابل','کراچی','لاہور','اسلام آباد','پشاور','کوئٹہ','ملتان',
    'دہلی','ممبئی','لندن','واشنگٹن','بیجنگ','ماسکو','تہران',
    'افغانستان','ایران','امریکہ','برطانیہ','روس','چین','بھارت',
    'ترکی','عرب','یورپ','افریقہ','ایشیا','مشرق وسطی',
    'دن','رات','ہفتہ','مہینہ','صبح','شام','سال','عمر','تاریخ',
    'نام','کام','دفتر','ملازمت','نوکری','تنخواہ','ہڑتال','اجرت',
    'طرز','انداز','خیال','رائے','مسئلہ','حل','فیصلہ','جواب',
    'سوال','بات','گفتگو','تبادلہ','اظہار','نظریہ','آواز',
    'دوست','دشمن','ساتھی','حریف','الزام','الزامات','تنقید',
]
for n in nouns: pos_lexicon[n] = 'NOUN'

# VERB (≥200)
verbs = [
    'ہے','ہیں','ہو','ہوں','تھا','تھی','تھے','تھیں','ہوگا','ہوگی','ہوگے',
    'کہا','کہی','کہے','کہتا','کہتی','کہتے','کہنا','کہیں','کہوں','کہو',
    'بتایا','بتائی','بتائے','بتاتا','بتاتی','بتانا','بتائیں',
    'کیا','کی','کئے','کرتا','کرتی','کرتے','کرنا','کریں','کرو','کروں',
    'ہوا','ہوئی','ہوئے','ہونا','ہوتا','ہوتی','ہوتے','ہوجاتا',
    'دیا','دی','دیئے','دینا','دیتا','دیتی','دیتے','دیں','دو',
    'لیا','لی','لئے','لینا','لیتا','لیتی','لیتے','لیں','لو',
    'آیا','آئی','آئے','آنا','آتا','آتی','آتے','آئیں',
    'گیا','گئی','گئے','جانا','جاتا','جاتی','جاتے','جائیں','جاؤ',
    'آرہا','آرہی','آرہے','جارہا','جارہی','جارہے',
    'چاہا','چاہی','چاہئے','چاہتا','چاہتی','چاہنا',
    'سکتا','سکتی','سکتے','سکیں','سکو',
    'پڑا','پڑی','پڑے','پڑنا','پڑتا','پڑتی','پڑتے',
    'نکلا','نکلی','نکلے','نکلنا','نکلتا','نکلنا',
    'ملا','ملی','ملے','ملنا','ملتا','ملتی','ملتے','ملیں',
    'بنا','بنی','بنے','بنانا','بناتا','بنتا','بنتی',
    'لگا','لگی','لگے','لگنا','لگتا','لگتی','لگتے',
    'رہا','رہی','رہے','رہنا','رہتا','رہتی','رہتے',
    'مانا','مانی','مانے','ماننا','مانتا','مانتی',
    'پڑھا','پڑھی','پڑھے','پڑھنا','پڑھتا','پڑھتی',
    'سنا','سنی','سنے','سننا','سنتا','سنتی',
    'دیکھا','دیکھی','دیکھے','دیکھنا','دیکھتا','دیکھتی',
    'اٹھایا','اٹھائی','اٹھانا','اٹھتا','اٹھتی',
    'چھوڑا','چھوڑی','چھوڑنا','چھوڑتا','چھوڑتی',
    'شروع','ختم','بند','کھول','چلایا','چلائی','چلانا',
    'بڑھا','بڑھی','بڑھنا','بڑھتا','بڑھتی',
    'گرا','گری','گرنا','گرتا','گرتی',
    'اعلان کیا','بیان دیا','مطالبہ کیا','احتجاج کیا',
    'منظور','رد','برخاست','تعینات','گرفتار','رہا',
    'جیتا','جیتی','جیتے','ہارا','ہاری','ہارے',
    'کھیلا','کھیلی','کھیلے','کھیلنا','کھیلتا',
    'مارا','ماری','مارنا','مارتا','مارتی',
    'پہنچا','پہنچی','پہنچے','پہنچنا','پہنچتا',
    'واپس','پلٹا','پلٹی','پلٹنا',
    'اٹھا','بیٹھا','کھڑا','لیٹا','چلا','بھاگا',
]
for v in verbs: pos_lexicon[v] = 'VERB'

# ADJ (≥200)
adjs = [
    'بڑا','بڑی','بڑے','چھوٹا','چھوٹی','چھوٹے',
    'نیا','نئی','نئے','پرانا','پرانی','پرانے',
    'اچھا','اچھی','اچھے','برا','بری','برے',
    'پہلا','پہلی','پہلے','آخری','دوسرا','دوسری','دوسرے',
    'اہم','خاص','عام','مشہور','معروف','نامور',
    'ممکن','ناممکن','مشکل','آسان','مناسب','صحیح','غلط',
    'مضبوط','کمزور','سخت','نرم','لمبا','چھوٹا',
    'گرم','ٹھنڈا','خشک','تر','صاف','گندا',
    'تیز','سست','پرانا','جدید','ترقی یافتہ','پسماندہ',
    'امیر','غریب','مالدار','کنگال','خوش','ناخوش',
    'کامیاب','ناکام','مشکوک','بے گناہ','مجرم',
    'مختلف','ایک جیسا','یکساں','متعدد','کئی',
    'کل','کچھ','زیادہ','کم','ہر','تمام','سب','کوئی',
    'ایسا','ویسا','ایسی','ویسی','اتنا','اتنی','جتنا',
    'خوبصورت','بدصورت','سادہ','پیچیدہ','واضح','مبہم',
    'سرکاری','نجی','وفاقی','صوبائی','مقامی','غیر ملکی',
    'قانونی','غیر قانونی','آئینی','غیر آئینی',
    'ملکی','بین الاقوامی','علاقائی','عالمی','قومی',
    'انسانی','سیاسی','معاشی','سماجی','مذہبی','ثقافتی',
    'تعلیمی','صحت','فوجی','عدالتی','قانونی',
    'موجودہ','سابق','آئندہ','مستقبل','گزشتہ','ماضی',
    'مکمل','جزوی','عارضی','مستقل','فوری','طویل المیعاد',
    'اعلیٰ','ادنیٰ','اعلی','کمتر','بہتر','بدتر',
    'واحد','متحد','الگ','مشترک','مشترکہ',
    'تمام','کل','پوری','سارا','سب',
    'ہزار','لاکھ','کروڑ','ارب','ملین','بلین',
    'بیشتر','اکثر','شاید','ضرور','واقعی','اصل',
    'بے حد','انتہائی','نہایت','بالکل','قطعی',
    'درست','صحیح','حقیقی','جھوٹا','سچا','خالص',
    'مسلح','غیر مسلح','مرکزی','مرکزیت','مقبول',
]
for a in adjs: pos_lexicon[a] = 'ADJ'

# PRON
prons = ['وہ','یہ','اس','ان','میں','تم','آپ','ہم','وہاں','یہاں',
         'جو','کون','کیا','کسی','کچھ','خود','اپنا','اپنی','اپنے',
         'اسے','انہیں','مجھے','تمہیں','ہمیں','انہوں','اسنے','اس نے']
for p in prons: pos_lexicon[p] = 'PRON'

# DET
dets = ['ایک','کئی','بہت','چند','تھوڑا','کافی','تمام','ہر','کچھ',
        'دونوں','سارے','اکثر','بعض','اگلا','اگلی','پچھلا','پچھلی',
        'اسی','وہی','یہی','ان','ان کا','اس کا']
for d in dets: pos_lexicon[d] = 'DET'

# CONJ
conjs = ['اور','لیکن','مگر','یا','بلکہ','کیونکہ','جب','تو','اگر',
         'تاکہ','کہ','جبکہ','حالانکہ','گرچہ','چاہے','نہ','نہیں',
         'ورنہ','اس لیے','لہذا','چنانچہ','پھر','پھر بھی']
for c in conjs: pos_lexicon[c] = 'CONJ'

# POST (postpositions)
posts = ['میں','کو','سے','پر','کا','کی','کے','نے','تک','تلک',
         'کے لیے','کی طرف','کے بعد','کے ساتھ','کے بغیر',
         'کے خلاف','کے تحت','کے علاوہ','کے بارے','کے سامنے',
         'کے درمیان','کے اندر','کے باہر','کے اوپر','کے نیچے']
for p in posts: pos_lexicon[p] = 'POST'

# ADV
advs = ['بھی','پھر','اب','کب','کیسے','کیوں','کہاں','کتنا','اتنا',
        'جلدی','آہستہ','یقیناً','شاید','ضرور','واقعی','سچ میں',
        'عام طور','بالکل','بالخصوص','خاص طور','عموماً','اکثر']
for a in advs: pos_lexicon[a] = 'ADV'

print(f"POS lexicon size: {len(pos_lexicon)} entries")
poscounts = Counter(pos_lexicon.values())
print("Counts:", dict(poscounts))

# ── Rule-based POS tagger ─────────────────────────────────────────────────────
VERB_SUFFIXES = ['تا','تی','تے','گا','گے','گی','یا','ئی','ئے']
ADJ_SUFFIXES  = ['نہ','انہ','وار','گار','کار','آور','پذیر']
NOUN_SUFFIXES = ['ی','ائی','اوٹ','اہٹ','پن','گری','داری','مندی']

def pos_tag(tokens):
    tags = []
    prev_tag = None
    for i, tok in enumerate(tokens):
        if tok == '<NUM>':
            tag = 'NUM'
        elif tok in pos_lexicon:
            tag = pos_lexicon[tok]
        else:
            # Suffix rules
            tag = None
            for suf in VERB_SUFFIXES:
                if tok.endswith(suf) and len(tok) > len(suf)+2:
                    tag = 'VERB'; break
            if tag is None:
                for suf in ADJ_SUFFIXES:
                    if tok.endswith(suf) and len(tok) > len(suf)+2:
                        tag = 'ADJ'; break
            if tag is None:
                # Context: word after POST/DET often NOUN
                if prev_tag in ('POST','DET'):
                    tag = 'NOUN'
                elif len(tok) >= 3:
                    tag = 'NOUN'  # default
                else:
                    tag = 'UNK'
        tags.append(tag)
        prev_tag = tag
    return tags

# ── NER Gazetteer ─────────────────────────────────────────────────────────────
ner_gazetteer = {}

# PERSONS (≥50)
persons = [
    'عمران خان','شہباز شریف','آصف زرداری','نواز شریف','بلاول بھٹو',
    'مریم نواز','بابر اعظم','شاہد آفریدی','ملالہ یوسفزئی','ڈونلڈ ٹرمپ',
    'جو بائیڈن','پوتن','مودی','شی جن پنگ','یحییٰ سنوار','نتن یاہو',
    'رضا پہلوی','خامنہ ای','اردوان','بن سلمان','سلمان خان',
    'علی خامنہ','حسن روحانی','ابراہیم رئیسی','محمود عباس',
    'انوار ابراہیم','سوریا ابراہیم','بشار الاسد','یاسر عرفات',
    'احمد شہزاد','محمد رضوان','شان مسعود','ناصر حسین','سرفراز احمد',
    'فضل الرحمن','اختر مینگل','محسن نقوی','عمر ایوب','صلاح الدین',
    'محمد یوسف','یونس خان','وسیم اکرم','جاوید میانداد','عمران فاروق',
    'طارق رحمان','خالدہ ضیا','شیخ حسینہ','حسن ناصر','نثار مرزا',
    'اسحاق ڈار','مفتاح اسماعیل','حمزہ شہباز','سعد رفیق',
    'رانا ثنا اللہ','پرویز خٹک','علی امین گنڈاپور','محسن نقوی',
    'قمر جاوید باجوہ','عاصم منیر','فیض حمید','ظہیر الاسلام',
]
for name in persons:
    ner_gazetteer[name] = 'PER'

# LOCATIONS (≥50)
locations = [
    'پاکستان','انڈیا','بھارت','افغانستان','ایران','بنگلادیش','سریلنکا',
    'چین','روس','امریکہ','برطانیہ','فرانس','جرمنی','جاپان','ترکی',
    'اسرائیل','فلسطین','عراق','شام','لبنان','مصر','سعودی عرب',
    'اردن','متحدہ عرب امارات','قطر','کویت','بحرین','عمان',
    'کراچی','لاہور','اسلام آباد','راولپنڈی','پشاور','کوئٹہ',
    'ملتان','فیصل آباد','حیدرآباد','سکھر','لاڑکانہ','مظفرآباد',
    'دہلی','ممبئی','کولکاتہ','چنئی','بنگلور','احمدآباد',
    'کابل','تہران','ریاض','دبئی','ابوظہبی','واشنگٹن',
    'لندن','پیرس','برلن','ماسکو','بیجنگ','طوکیو',
    'اقوام متحدہ','جنیوا','ہیگ','برسلز',
    'پنجاب','سندھ','بلوچستان','خیبر پختونخوا','آزاد کشمیر','گلگت',
    'وزیرستان','سوات','دیر','چترال','ہنزہ','مہمند','باجوڑ',
    'مقبوضہ کشمیر','لائن آف کنٹرول','آبنائے ہرمز','بحیرہ احمر',
    'یورپ','افریقہ','ایشیا','مشرق وسطی','جنوبی ایشیا',
]
for loc in locations:
    ner_gazetteer[loc] = 'LOC'

# ORGANIZATIONS (≥30)
orgs = [
    'بی بی سی','تحریک انصاف','مسلم لیگ ن','مسلم لیگ','پیپلز پارٹی',
    'جماعت اسلامی','متحدہ قومی موومنٹ','عوامی نیشنل پارٹی',
    'سپریم کورٹ','ہائی کورٹ','اسلامی عدالت','فوجی عدالت',
    'اقوام متحدہ','عالمی ادارہ صحت','یونیسیف','یو این ایچ سی آر',
    'آئی ایم ایف','عالمی بینک','ایشیائی ترقیاتی بینک',
    'پی سی بی','بی سی سی آئی','آئی سی سی','ایشیا کپ',
    'ایف آئی اے','نیب','پی ٹی اے','اوگرا','نیپرا',
    'ناٹو','شنگھائی تعاون','سارک','آسیان','اوپیک',
    'آئی ایس آئی','ایم آئی','رینجرز','ایف سی',
    'حماس','حزب اللہ','طالبان','داعش','القاعدہ',
    'جیو','اے آر وائی','ڈان','دنیا نیوز','سما',
    'ریاست بھر','ریاستی','مرکزی حکومت','صوبائی حکومت',
]
for org in orgs:
    ner_gazetteer[org] = 'ORG'

# MISC
miscs = [
    'رمضان','عید الفطر','عید الاضحی','محرم','عاشورہ','بقر عید',
    'کرکٹ ورلڈ کپ','پی ایس ایل','ایشیا کپ','ٹی ٹوئنٹی',
    'نوبل','آسکر','گولڈن گلوب','بوکر','پلٹزر',
    'کووڈ','کورونا','ایچ آئی وی','ایڈز','ڈینگی',
]
for m in miscs:
    ner_gazetteer[m] = 'MISC'

print(f"NER gazetteer: {len(ner_gazetteer)} entries")

# Context triggers for NER
ner_pre_triggers = {
    'PER': ['صدر','وزیراعظم','وزیر','وزیراعلیٰ','چیئرمین','جنرل','کرنل',
            'ایڈمرل','جناب','محترمہ','ڈاکٹر','پروفیسر','انجینئر'],
    'LOC': ['صوبہ','شہر','ملک','ریاست','علاقہ','ضلع','تحصیل','گاؤں'],
    'ORG': ['پارٹی','جماعت','ادارہ','تنظیم','کمپنی','وزارت','محکمہ'],
}

def ner_tag(tokens, pos_tags):
    """BIO NER tagging using gazetteer + context rules."""
    n = len(tokens)
    bio = ['O'] * n

    # Multi-word entity matching (check bigrams, trigrams)
    i = 0
    while i < n:
        matched = False
        # Try 3-gram
        if i+2 < n:
            phrase3 = tokens[i]+' '+tokens[i+1]+' '+tokens[i+2]
            if phrase3 in ner_gazetteer:
                etype = ner_gazetteer[phrase3]
                bio[i] = f'B-{etype}'
                bio[i+1] = f'I-{etype}'
                bio[i+2] = f'I-{etype}'
                i += 3; matched = True
        # Try 2-gram
        if not matched and i+1 < n:
            phrase2 = tokens[i]+' '+tokens[i+1]
            if phrase2 in ner_gazetteer:
                etype = ner_gazetteer[phrase2]
                bio[i] = f'B-{etype}'
                bio[i+1] = f'I-{etype}'
                i += 2; matched = True
        # Try 1-gram
        if not matched:
            tok = tokens[i]
            if tok in ner_gazetteer:
                etype = ner_gazetteer[tok]
                bio[i] = f'B-{etype}'
            # Context: word after PER trigger → B-PER
            elif i > 0 and tokens[i-1] in ner_pre_triggers['PER'] and pos_tags[i] == 'NOUN':
                bio[i] = 'B-PER'
            elif i > 0 and tokens[i-1] in ner_pre_triggers['LOC'] and pos_tags[i] == 'NOUN':
                bio[i] = 'B-LOC'
            i += 1

    return bio

# ── Annotate 500 sentences ────────────────────────────────────────────────────
print("\nAnnotating 500 sentences...")
annotated = []
for doc_id, topic, tokens in selected:
    pos_tags = pos_tag(tokens)
    ner_tags = ner_tag(tokens, pos_tags)
    annotated.append({
        'doc_id': doc_id,
        'topic': topic,
        'tokens': tokens,
        'pos': pos_tags,
        'ner': ner_tags
    })

print(f"Annotated {len(annotated)} sentences")

# POS distribution
all_pos = [tag for s in annotated for tag in s['pos']]
print("\nPOS distribution:")
for tag, cnt in Counter(all_pos).most_common():
    print(f"  {tag}: {cnt}")

# NER distribution
all_ner = [tag for s in annotated for tag in s['ner']]
ner_counts = Counter(all_ner)
print("\nNER distribution:")
for tag, cnt in ner_counts.most_common():
    print(f"  {tag}: {cnt}")

# ── Split: 350 train / 75 val / 75 test ──────────────────────────────────────
# Stratified by topic
from collections import defaultdict
by_topic = defaultdict(list)
for i, sample in enumerate(annotated):
    by_topic[sample['topic']].append(i)

train_ids, val_ids, test_ids = [], [], []
for topic, ids in by_topic.items():
    random.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.15))
    test_ids.extend(ids[:n_test])
    val_ids.extend(ids[n_test:n_test+n_val])
    train_ids.extend(ids[n_test+n_val:])

print(f"\nSplit: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

# ── Save CoNLL files ──────────────────────────────────────────────────────────
def save_conll(path, sentence_ids, data, field):
    """field: 'pos' or 'ner'"""
    with open(path, 'w', encoding='utf-8') as f:
        for idx in sentence_ids:
            s = data[idx]
            for tok, tag in zip(s['tokens'], s[field]):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")
    print(f"Saved {path} ({len(sentence_ids)} sentences)")

save_conll('data/pos_train.conll', train_ids, annotated, 'pos')
save_conll('data/pos_test.conll',  test_ids,  annotated, 'pos')
save_conll('data/ner_train.conll', train_ids, annotated, 'ner')
save_conll('data/ner_test.conll',  test_ids,  annotated, 'ner')

# Also save val splits
save_conll('data/pos_val.conll', val_ids, annotated, 'pos')
save_conll('data/ner_val.conll', val_ids, annotated, 'ner')

# Save annotated data to state
st['annotated'] = annotated
st['train_ids'] = train_ids
st['val_ids']   = val_ids
st['test_ids']  = test_ids
with open('embeddings/state_part1.pkl','wb') as f: pickle.dump(st, f)
print("\nState saved. Part 2 annotation complete!")
