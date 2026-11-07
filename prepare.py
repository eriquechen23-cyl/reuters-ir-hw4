"""Download Reuters-21578, train two Word2Vec models, and freeze real results."""
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from scipy import sparse
from gensim.models import Word2Vec
from retrieval import ROOT, METHODS, Engine, tokenize, fit_tfidf, transform, normalize_dense, metrics

SOURCE = "https://archive.ics.uci.edu/static/public/137/reuters%2B21578%2Btext%2Bcategorization%2Bcollection.zip"
PARAMS = dict(vector_size=100, window=5, min_count=2, workers=1, seed=42, epochs=15, negative=5, hs=0, sample=0.001)
PROBES = ["oil", "petroleum", "profit", "loss", "earnings", "acquisition", "merger", "wheat", "grain", "trade", "export", "import"]

def field(text, name):
    match = re.search(fr"<{name}\b[^>]*>(.*?)</{name}>", text, re.S)
    return match.group(1) if match else ""

def parse_sgml(text):
    docs = []
    for attrs, body in re.findall(r"<REUTERS\b([^>]*)>(.*?)</REUTERS>", text, re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        split = attrs.get("LEWISSPLIT")
        if attrs.get("TOPICS") != "YES" or split not in ("TRAIN", "TEST"):
            continue
        title = html.unescape(field(body, "TITLE")).strip()
        content = html.unescape(field(body, "BODY")).strip()
        if not title and not content:
            content = html.unescape(re.sub(r"<[^>]+>", " ", field(body, "TEXT"))).strip()
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
        topics = re.findall(r"<D>(.*?)</D>", field(body, "TOPICS"))
        docs.append(dict(id=int(attrs["NEWID"]), split=split, title=title, body=content, topics=topics))
    return docs

def download():
    path = ROOT / "data/raw/reuters.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print("Downloading official UCI Reuters archive", flush=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(SOURCE, timeout=90) as response:
                    data = response.read()
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("Downloaded ZIP failed CRC validation")
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"Download interrupted; retry {attempt+1}/2", flush=True)
        path.write_bytes(data)
    return path

def read_corpus(path):
    docs = []
    with zipfile.ZipFile(path) as z:
        tar_name = next(n for n in z.namelist() if n.endswith("reuters21578.tar.gz"))
        with tarfile.open(fileobj=io.BytesIO(z.read(tar_name)), mode="r:gz") as tar:
            for member in sorted(tar.getmembers(), key=lambda m: m.name):
                if re.fullmatch(r"(?:.*/)?reut2-\d{3}\.sgm", member.name):
                    docs.extend(parse_sgml(tar.extractfile(member).read().decode("latin-1")))
            readmes = [m for m in tar.getmembers() if Path(m.name).name.lower().startswith("readme") and m.isfile()]
            if readmes:
                (ROOT / "data/raw/README-original.txt").write_bytes(tar.extractfile(readmes[0]).read())
    return sorted(docs, key=lambda d: d["id"])

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    started = time.perf_counter()
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    archive = download()
    docs = read_corpus(archive)
    train_all = [d for d in docs if d["split"] == "TRAIN"]
    test_all = [d for d in docs if d["split"] == "TEST"]
    train = train_all  # Keep the complete official split, including zero-token records.
    test = test_all
    assert not {d["id"] for d in train} & {d["id"] for d in test}
    tokens = [tokenize(d["title"] + " " + d["body"]) for d in train]
    test_tokens = [tokenize(d["title"] + " " + d["body"]) for d in test]
    print(f"ModApte raw TRAIN/TEST: {len(train_all)}/{len(test_all)}; usable: {len(train)}/{len(test)}", flush=True)
    vocab, df, idf = fit_tfidf(tokens)
    words = list(vocab)
    matrix = transform(test_tokens, vocab, idf)
    save_json(out / "vocabulary.json", words)
    save_json(out / "documents.json", test)
    np.save(out / "idf.npy", idf); np.save(out / "df.npy", df)
    sparse.save_npz(out / "tfidf.npz", matrix)
    training_seconds = {}
    # Contexts stay within a document; sentence boundaries are not enforced.
    for method, sg in [("cbow", 0), ("sg", 1)]:
        start = time.perf_counter()
        print(f"Training {method}: {PARAMS}", flush=True)
        model = Word2Vec(sentences=tokens, sg=sg, **PARAMS)
        vectors = np.zeros((len(vocab), PARAMS["vector_size"]), dtype=np.float32)
        for word, i in vocab.items():
            if word in model.wv:
                vectors[i] = model.wv[word]
        np.save(out / f"{method}_words.npy", vectors)
        np.save(out / f"{method}_docs.npy", normalize_dense(matrix @ vectors))
        training_seconds[method] = round(time.perf_counter() - start, 3)
    metadata = dict(source=SOURCE, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    created_at=datetime.now(timezone.utc).isoformat(), params=PARAMS,
                    train_raw=len(train_all), test_raw=len(test_all), train_count=len(train), test_count=len(test),
                    excluded_empty_train=len(train_all)-len(train), excluded_empty_test=len(test_all)-len(test),
                    vocabulary_size=len(vocab), embedding_vocabulary=int((np.linalg.norm(vectors, axis=1)>0).sum()),
                    training_seconds=training_seconds, pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                    train_ids=[d["id"] for d in train],
                    topic_counts={t:sum(t in d["topics"] for d in test) for t in ("earn","acq","crude","grain","trade")})
    save_json(out / "metadata.json", metadata)
    engine = Engine(out)
    queries = json.loads((ROOT / "queries.json").read_text(encoding="utf-8"))
    rows, rankings = [], []
    for q in queries:
        for method in METHODS:
            order, scores = engine.search(q["query"], method)
            rows.append(dict(**q, method=method, **metrics(order, engine.relevant(q["topic"]))))
            rankings.append(dict(**q, method=method, top10=[dict(id=test[int(i)]["id"], score=float(scores[i]), relevant=q["topic"] in test[int(i)]["topics"]) for i in order[:10]]))
    summary = {m: {metric: float(np.mean([r[metric] for r in rows if r["method"] == m])) for metric in ("P@10","MRR","nDCG@10")} for m in METHODS}
    save_json(out / "evaluation.json", dict(summary=summary, rows=rows, rankings=rankings, query_count=len(queries)))
    neighbors = {m:{w:engine.neighbors(w,m) for w in PROBES} for m in ("cbow","sg")}
    save_json(out / "neighbors.json", neighbors)
    with (report / "evaluation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_json(out / "manifest.json", {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != "manifest.json"})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Finished in {time.perf_counter()-started:.1f}s", flush=True)

if __name__ == "__main__":
    main()
