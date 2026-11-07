"""Explicit TF-IDF and cosine retrieval. Fit on TRAIN, rank TEST only."""
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
METHODS = {"tfidf": "TF-IDF", "cbow": "TF-IDF × CBOW", "sg": "TF-IDF × Skip-Gram"}
# Deliberately retain negation words (no, not, without).
STOPWORDS = frozenset("a an the and or of to in on at by for from with as is are was were be been being it its this that these those he she they we you i his her their our your said reuter reuters will would could should has have had do does did than then also about into which who whom up out over under".split())

def tokenize(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]

def normalize_dense(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)

def fit_tfidf(documents):
    df = Counter(w for tokens in documents for w in set(tokens))
    words = sorted(df)
    vocab = {w: i for i, w in enumerate(words)}
    dfs = np.array([df[w] for w in words], dtype=np.int32)
    idf = np.log((1 + len(documents)) / (1 + dfs)) + 1
    return vocab, dfs, idf.astype(np.float32)

def transform(documents, vocab, idf):
    rows, cols, values = [], [], []
    for row, tokens in enumerate(documents):
        for word, count in Counter(tokens).items():
            if word in vocab:
                col = vocab[word]
                rows.append(row); cols.append(col)
                values.append((1 + np.log(count)) * idf[col])
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(documents), len(vocab)), dtype=np.float32)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    inverse = np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)
    return sparse.diags(inverse).dot(matrix).tocsr()

def rank(scores, valid=None):
    # Stable index tie break follows ascending Reuters NEWID.
    indices = np.argsort(-scores, kind="stable")
    return indices if valid is None else indices[valid[indices]]

def metrics(ranking, relevant, k=10):
    hits = np.asarray([int(i in relevant) for i in ranking], dtype=float)
    first = np.flatnonzero(hits)
    discount = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(np.dot(hits[:k], discount[:len(hits[:k])]))
    ideal = float(discount[:min(k, len(relevant))].sum())
    return {"P@10": float(hits[:k].sum() / k), "MRR": float(1 / (first[0] + 1)) if len(first) else 0., "nDCG@10": dcg / ideal if ideal else 0.}

class Engine:
    def __init__(self, directory=ROOT / "artifacts"):
        directory = Path(directory)
        self.meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        self.words = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
        self.vocab = {w: i for i, w in enumerate(self.words)}
        self.idf = np.load(directory / "idf.npy", allow_pickle=False)
        self.df = np.load(directory / "df.npy", allow_pickle=False)
        self.tfidf = sparse.load_npz(directory / "tfidf.npz")
        self.embeddings = {m: np.load(directory / f"{m}_words.npy", allow_pickle=False) for m in ("cbow", "sg")}
        self.word_units = {m: normalize_dense(v) for m, v in self.embeddings.items()}
        self.doc_vectors = {m: np.load(directory / f"{m}_docs.npy", allow_pickle=False) for m in ("cbow", "sg")}

    def search(self, query, method):
        tokens = tokenize(query)
        q = transform([tokens], self.vocab, self.idf)
        if method == "tfidf":
            scores = (self.tfidf @ q.T).toarray().ravel()
            valid = scores > 0  # Do not present zero-overlap ties as retrieved evidence.
        else:
            vector = normalize_dense(q @ self.embeddings[method])
            scores = self.doc_vectors[method] @ vector[0]
            valid = np.linalg.norm(self.doc_vectors[method], axis=1) > 0
            if not vector.any():
                valid[:] = False
        return rank(scores, valid), scores

    def unknown(self, query, method):
        return sorted({w for w in tokenize(query) if w not in self.vocab or (method != "tfidf" and not self.embeddings[method][self.vocab[w]].any())})

    def neighbors(self, word, method, k=5):
        i = self.vocab.get(word.lower())
        if i is None or not self.word_units[method][i].any():
            return []
        scores = self.word_units[method] @ self.word_units[method][i]
        scores[i] = -np.inf
        valid = np.linalg.norm(self.word_units[method], axis=1) > 0
        return [(self.words[j], float(scores[j])) for j in rank(scores, valid)[:k]]

    def relevant(self, topic):
        return {i for i, d in enumerate(self.docs) if topic in d["topics"]}
