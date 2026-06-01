"""
NLP Multi-Feature Analyzer — Flask API Backend
Routes: /api/sentiment, /api/summarize, /api/ner, /api/keywords, /api/qa, /api/dependency
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter
import math
import re

app = Flask(__name__)
CORS(app)

# ── Load spaCy model once at startup ──────────────────────────────────────────
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy ready.")

# ── HuggingFace pipelines — lazy loaded on first use ─────────────────────────
_summarizer = None
_qa_pipeline = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        print("Loading summarization model (first use)...")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU; change to 0 for GPU
        )
        print("Summarization model ready.")
    return _summarizer

def get_qa():
    global _qa_pipeline
    if _qa_pipeline is None:
        from transformers import pipeline
        print("Loading QA model (first use)...")
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
        print("QA model ready.")
    return _qa_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_text(data, min_words=5):
    """Extract and validate text from request JSON."""
    text = (data.get("text") or "").strip()
    if not text:
        return None, "No text provided."
    words = text.split()
    if len(words) < min_words:
        return None, f"Please provide at least {min_words} words."
    return text, None


def compute_tfidf_keywords(doc, top_n=15):
    """
    Extract keywords using TF-IDF on noun chunks + important tokens.
    Returns list of {word, score, count} dicts sorted by score descending.
    """
    # Candidate tokens: nouns, proper nouns, adjectives (not stopwords)
    candidates = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ")
        and not token.is_stop
        and not token.is_punct
        and len(token.text) > 2
    ]
    if not candidates:
        return []

    # TF
    total = len(candidates)
    tf = Counter(candidates)

    # IDF approximation: use sentence count as "documents"
    sentences = list(doc.sents)
    num_docs = max(len(sentences), 1)
    idf = {}
    for word in set(candidates):
        docs_with_word = sum(
            1 for sent in sentences
            if word in sent.text.lower()
        )
        idf[word] = math.log((num_docs + 1) / (docs_with_word + 1)) + 1

    scored = [
        {
            "word": word,
            "score": round((count / total) * idf.get(word, 1), 4),
            "count": count
        }
        for word, count in tf.items()
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "NLP Analyzer API is running."})


@app.route("/api/sentiment", methods=["POST"])
def sentiment():
    """
    Analyse sentiment using TextBlob polarity + subjectivity.
    Returns: label, polarity (-1 to 1), subjectivity (0 to 1),
             sentence-level breakdown.
    """
    text, err = validate_text(request.json or {}, min_words=3)
    if err:
        return jsonify({"error": err}), 400

    blob = TextBlob(text)
    polarity    = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > 0.1:
        label, color = "Positive", "#22c55e"
    elif polarity < -0.1:
        label, color = "Negative", "#ef4444"
    else:
        label, color = "Neutral", "#94a3b8"

    # Confidence: map polarity to 0–100 range
    confidence = round(min(abs(polarity) * 2, 1.0) * 100, 1)

    # Sentence-level
    sentences = []
    for sent in blob.sentences:
        p = round(sent.sentiment.polarity, 3)
        if p > 0.1:
            s_label = "Positive"
        elif p < -0.1:
            s_label = "Negative"
        else:
            s_label = "Neutral"
        sentences.append({
            "text": str(sent),
            "polarity": p,
            "label": s_label
        })

    return jsonify({
        "label":        label,
        "color":        color,
        "polarity":     polarity,
        "subjectivity": subjectivity,
        "confidence":   confidence,
        "sentences":    sentences
    })


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """
    Summarize text.
    - Short texts (< 100 words): LSA extractive via sumy (fast, no model load)
    - Long texts (>= 100 words): BART abstractive via HuggingFace
    """
    text, err = validate_text(request.json or {}, min_words=30)
    if err:
        return jsonify({"error": err}), 400

    word_count  = len(text.split())
    num_sentences = request.json.get("sentences", 3)  # user-configurable

    if word_count < 100:
        # Extractive (sumy LSA) — fast
        parser    = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sents = summarizer(parser.document, num_sentences)
        summary   = " ".join(str(s) for s in summary_sents)
        method    = "extractive"
    else:
        # Abstractive (BART) — quality
        max_len = min(150, max(50, word_count // 3))
        min_len = max(30, max_len // 3)
        result  = get_summarizer()(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        summary = result[0]["summary_text"]
        method  = "abstractive (BART)"

    compression = round((1 - len(summary.split()) / word_count) * 100, 1)

    return jsonify({
        "summary":          summary,
        "method":           method,
        "original_words":   word_count,
        "summary_words":    len(summary.split()),
        "compression_rate": compression
    })


@app.route("/api/ner", methods=["POST"])
def ner():
    """
    Named Entity Recognition using spaCy.
    Returns: entities list, displaCy HTML, entity type breakdown.
    """
    text, err = validate_text(request.json or {}, min_words=5)
    if err:
        return jsonify({"error": err}), 400

    doc = nlp(text)

    # Entity list
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

    # displaCy HTML (entity visualizer)
    from spacy import displacy
    html = displacy.render(doc, style="ent", page=False, minify=True)

    # Type breakdown for chart
    type_counts = Counter(e["label"] for e in entities)
    breakdown   = [{"label": k, "count": v} for k, v in type_counts.most_common()]

    return jsonify({
        "entities":  entities,
        "html":      html,
        "breakdown": breakdown,
        "total":     len(entities)
    })


@app.route("/api/keywords", methods=["POST"])
def keywords():
    """
    Extract keywords using TF-IDF scoring over spaCy POS tags.
    Returns top N keywords with scores.
    """
    data = request.json or {}
    text, err = validate_text(data, min_words=10)
    if err:
        return jsonify({"error": err}), 400

    top_n = int(data.get("top_n", 15))
    doc   = nlp(text)
    kws   = compute_tfidf_keywords(doc, top_n=top_n)

    return jsonify({
        "keywords":    kws,
        "total_words": len([t for t in doc if not t.is_space])
    })


@app.route("/api/qa", methods=["POST"])
def question_answer():
    """
    Extractive QA using RoBERTa (deepset/roberta-base-squad2).
    Requires: context (text) and question.
    """
    data     = request.json or {}
    context  = (data.get("text") or "").strip()
    question = (data.get("question") or "").strip()

    if not context:
        return jsonify({"error": "No context text provided."}), 400
    if not question:
        return jsonify({"error": "No question provided."}), 400
    if len(context.split()) < 10:
        return jsonify({"error": "Context too short. Provide more text."}), 400

    result = get_qa()(question=question, context=context)

    return jsonify({
        "answer":     result["answer"],
        "score":      round(result["score"], 4),
        "confidence": round(result["score"] * 100, 1),
        "question":   question
    })


@app.route("/api/dependency", methods=["POST"])
def dependency():
    """
    Generate displaCy dependency parse SVG for a sentence.
    Uses only the first sentence if text is long.
    """
    text, err = validate_text(request.json or {}, min_words=3)
    if err:
        return jsonify({"error": err}), 400

    doc = nlp(text)

    # Use first sentence only (dep tree is unreadable for long texts)
    first_sent = list(doc.sents)[0]
    sent_doc   = first_sent.as_doc()

    from spacy import displacy
    svg = displacy.render(
        sent_doc,
        style="dep",
        page=False,
        minify=True,
        options={"compact": True, "bg": "#0d1117", "color": "#e2e8f0", "font": "DM Sans"}
    )

    return jsonify({
        "svg":      svg,
        "sentence": first_sent.text,
        "tokens":   [
            {"text": t.text, "pos": t.pos_, "dep": t.dep_, "head": t.head.text}
            for t in first_sent
        ]
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
