"""
nlp_core.py — Shared NLP logic
Imported by both app.py (Flask) and streamlit_app.py (HF Spaces).
"""

import spacy
import math
from collections import Counter

# ── spaCy — loaded once ───────────────────────────────────────────────────────
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy ready.")

# ── HuggingFace pipelines — lazy loaded on first call ─────────────────────────
_summarizer = None
_qa_pipeline = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        print("Loading BART summarization model...")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        print("Summarization model ready.")
    return _summarizer

def get_qa():
    global _qa_pipeline
    if _qa_pipeline is None:
        from transformers import pipeline
        print("Loading RoBERTa QA model...")
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
        print("QA model ready.")
    return _qa_pipeline


# ── Core functions ─────────────────────────────────────────────────────────────

def analyse_sentiment(text: str) -> dict:
    from textblob import TextBlob
    blob = TextBlob(text)
    polarity     = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > 0.1:
        label, color = "Positive", "#22c55e"
    elif polarity < -0.1:
        label, color = "Negative", "#ef4444"
    else:
        label, color = "Neutral", "#94a3b8"

    confidence = round(min(abs(polarity) * 2, 1.0) * 100, 1)

    sentences = []
    for sent in blob.sentences:
        p = round(sent.sentiment.polarity, 3)
        sentences.append({
            "text":     str(sent),
            "polarity": p,
            "label":    "Positive" if p > 0.1 else "Negative" if p < -0.1 else "Neutral"
        })

    return {
        "label":        label,
        "color":        color,
        "polarity":     polarity,
        "subjectivity": subjectivity,
        "confidence":   confidence,
        "sentences":    sentences
    }


def summarize_text(text: str, num_sentences: int = 3) -> dict:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer

    word_count = len(text.split())

    if word_count < 100:
        parser    = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary   = " ".join(str(s) for s in summarizer(parser.document, num_sentences))
        method    = "extractive (LSA)"
    else:
        max_len = min(150, max(50, word_count // 3))
        min_len = max(30, max_len // 3)
        result  = get_summarizer()(text, max_length=max_len, min_length=min_len, do_sample=False)
        summary = result[0]["summary_text"]
        method  = "abstractive (BART)"

    return {
        "summary":          summary,
        "method":           method,
        "original_words":   word_count,
        "summary_words":    len(summary.split()),
        "compression_rate": round((1 - len(summary.split()) / word_count) * 100, 1)
    }


def run_ner(text: str) -> dict:
    from spacy import displacy
    doc      = nlp(text)
    entities = [
        {
            "text":  ent.text,
            "label": ent.label_,
            "desc":  spacy.explain(ent.label_) or ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char
        }
        for ent in doc.ents
    ]
    html         = displacy.render(doc, style="ent", page=False, minify=True)
    type_counts  = Counter(e["label"] for e in entities)
    breakdown    = [{"label": k, "count": v} for k, v in type_counts.most_common()]
    return {"entities": entities, "html": html, "breakdown": breakdown, "total": len(entities)}


def extract_keywords(text: str, top_n: int = 15) -> dict:
    doc        = nlp(text)
    candidates = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ")
        and not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    if not candidates:
        return {"keywords": [], "total_words": len(doc)}

    total     = len(candidates)
    tf        = Counter(candidates)
    sentences = list(doc.sents)
    num_docs  = max(len(sentences), 1)
    idf       = {
        word: math.log((num_docs + 1) / (sum(1 for s in sentences if word in s.text.lower()) + 1)) + 1
        for word in set(candidates)
    }
    scored = sorted(
        [{"word": w, "score": round((c / total) * idf.get(w, 1), 4), "count": c} for w, c in tf.items()],
        key=lambda x: x["score"], reverse=True
    )
    return {"keywords": scored[:top_n], "total_words": len([t for t in doc if not t.is_space])}


def answer_question(text: str, question: str) -> dict:
    result = get_qa()(question=question, context=text)
    return {
        "answer":     result["answer"],
        "score":      round(result["score"], 4),
        "confidence": round(result["score"] * 100, 1),
        "question":   question
    }


def dependency_parse(text: str) -> dict:
    from spacy import displacy
    doc        = nlp(text)
    first_sent = list(doc.sents)[0]
    sent_doc   = first_sent.as_doc()
    svg        = displacy.render(
        sent_doc, style="dep", page=False, minify=True,
        options={"compact": True, "bg": "#0d1117", "color": "#e2e8f0", "font": "DM Sans"}
    )
    return {
        "svg":      svg,
        "sentence": first_sent.text,
        "tokens":   [
            {"text": t.text, "pos": t.pos_, "dep": t.dep_, "head": t.head.text}
            for t in first_sent
        ]
    }
