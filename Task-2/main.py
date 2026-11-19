#!/usr/bin/env python3
"""
Task 2 - Retrieval-augmented QA built from scratch (no LangChain / LlamaIndex).

Pipeline: PDF -> clean text -> word chunks -> dense (FAISS) + BM25 (written here)
          -> Reciprocal Rank Fusion -> cross-encoder re-ranker -> LLM answer with citations.

    python main.py evaluate                       # recall@k for two chunk sizes + failure analysis
    python main.py ask "What is BERT's pre-training objective?"

Ground truth is a 10-25 word *evidence span* per question (questions.json), not a chunk id,
so the same questions can score both chunk sizes fairly. A retrieved chunk counts as correct
when it contains >= 80% of the evidence span's word trigrams.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ---- configuration ---------------------------------------------------------
DOCS = {
    "llama2": "https://arxiv.org/pdf/2307.09288",      # Llama 2: Open Foundation and Fine-Tuned Chat Models
    "rag_paper": "https://arxiv.org/pdf/2005.11401",   # Retrieval-Augmented Generation for Knowledge-Intensive NLP
    "bert": "https://arxiv.org/pdf/1810.04805",        # BERT
}
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # BGE: for queries only
RERANK_MODEL = "BAAI/bge-reranker-base"
CHUNK_CONFIGS = {"small_100w": (100, 25), "large_300w": (300, 60)}           # (words, overlap in words)
POOL = 50                 # candidates kept from each retriever
RERANK_POOL = 30          # candidates passed to the cross-encoder
CONTEXT_BUDGET_WORDS = 1500
METHODS = ["dense", "bm25", "hybrid_rrf", "hybrid+rerank"]

ANSWER_SYSTEM = ("Answer the question using ONLY the numbered context passages. Cite passages like [1] or [2][3]. "
                 "If the answer is not in the context, reply exactly: Not found in the provided documents. Be concise.")

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]),
    "nvidia": ("https://integrate.api.nvidia.com/v1", ["meta/llama-3.3-70b-instruct"]),
}


# ---- text cleaning ----------------------------------------------------------
LIG = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl",
       "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}


def clean(t):
    for k, v in LIG.items():
        t = t.replace(k, v)
    t = re.sub(r"-\n(?=[a-z])", "", t)               # re-join words hyphenated at a line end
    return re.sub(r"\s+", " ", t).strip()


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def trigrams(words):
    return set(zip(words, words[1:], words[2:]))


# ---- documents --------------------------------------------------------------
def fetch_pdfs(pdf_dir):
    import requests
    os.makedirs(pdf_dir, exist_ok=True)
    paths = {}
    for name, url in DOCS.items():
        path = os.path.join(pdf_dir, f"{name}.pdf")
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
            r.raise_for_status()
            assert r.content[:4] == b"%PDF", f"{url} did not return a PDF"
            open(path, "wb").write(r.content)
        paths[name] = path
    return paths


def load_pages(paths):
    import fitz                                       # PyMuPDF
    pages = {}
    for name, p in paths.items():
        with fitz.open(p) as pdf:
            pages[name] = [clean(pg.get_text()) for pg in pdf]
    total = sum(len(v) for v in pages.values())
    for n, v in pages.items():
        print(f"  {n}: {len(v)} pages, {sum(len(t.split()) for t in v)} words")
    assert total >= 30, "Need at least 30 pages in total"
    return pages


# ---- chunking ---------------------------------------------------------------
def make_chunks(pages_by_doc, size, overlap):
    """Sliding word window; every chunk remembers its document and page range."""
    step, chunks = size - overlap, []
    for name, pages in pages_by_doc.items():
        words, wpage = [], []
        for pno, txt in enumerate(pages, 1):
            w = txt.split()
            words += w
            wpage += [pno] * len(w)
        for start in range(0, max(1, len(words) - overlap), step):
            seg = words[start:start + size]
            if not seg:
                continue
            chunks.append({"id": len(chunks), "doc": name, "text": " ".join(seg),
                           "page_start": wpage[start], "page_end": wpage[start + len(seg) - 1]})
    return chunks


# ---- BM25 from scratch ------------------------------------------------------
STOP = set("a an the of and or to in on for with is are was were be by as at from that this it its "
           "which what who how why when do does did can than then into about".split())


def tokenize(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class BM25:
    """score(q, d) = sum_t idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*len(d)/avglen))"""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.N = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean())
        self.inv = defaultdict(list)                  # term -> [(doc_id, term_freq)]
        for i, d in enumerate(docs):
            for term, f in Counter(d).items():
                self.inv[term].append((i, f))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.inv.items()}

    def search(self, query, k=POOL):
        scores = np.zeros(self.N, dtype=np.float32)
        for t in set(tokenize(query)):
            for i, f in self.inv.get(t, []):
                scores[i] += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]


def rrf(rank_lists, k=60, topn=100):
    """Reciprocal Rank Fusion: score(d) = sum 1/(k + rank). Uses ranks only, no score calibration."""
    s = defaultdict(float)
    for lst in rank_lists:
        for r, i in enumerate(lst):
            s[i] += 1.0 / (k + r + 1)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])][:topn]


# ---- retriever --------------------------------------------------------------
class Retriever:
    def __init__(self, pages, configs):
        import faiss
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.faiss = faiss
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", device)
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self.embedder.max_seq_length = 512
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=device)
        self.index = {}
        for name in configs:
            size, ov = CHUNK_CONFIGS[name]
            chunks = make_chunks(pages, size, ov)
            print(f"Indexing {name}: {len(chunks)} chunks")
            emb = self.embedder.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=True).astype("float32")
            fx = faiss.IndexFlatIP(emb.shape[1])      # inner product of unit vectors = cosine
            fx.add(emb)
            self.index[name] = {"chunks": chunks, "faiss": fx, "bm25": BM25([c["text"] for c in chunks]),
                                "tri": [trigrams(norm(c["text"]).split()) for c in chunks]}

    def dense(self, cfg, query, k=POOL):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
        _, ids = self.index[cfg]["faiss"].search(q, k)
        return [int(i) for i in ids[0] if i >= 0]

    def bm25(self, cfg, query, k=POOL):
        return self.index[cfg]["bm25"].search(query, k)

    def rerank(self, cfg, query, cand_ids):
        if not cand_ids:
            return []
        chunks = self.index[cfg]["chunks"]
        sc = self.reranker.predict([(query, chunks[i]["text"]) for i in cand_ids],
                                   batch_size=16, show_progress_bar=False)
        return [cand_ids[j] for j in np.argsort(-np.asarray(sc))]

    def retrieve(self, cfg, query, k=5):
        """Production path: hybrid (dense + BM25 via RRF) -> cross-encoder -> top-k chunks."""
        h = rrf([self.dense(cfg, query), self.bm25(cfg, query)])
        chunks = self.index[cfg]["chunks"]
        return [chunks[i] for i in self.rerank(cfg, query, h[:RERANK_POOL])[:k]]


# ---- LLM client (any OpenAI-compatible endpoint) -----------------------------
def get_llm():
    from openai import OpenAI
    key = os.environ.get("LLM_API_KEY")
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    candidates = [model] if model else []
    if not key:
        for env, prov in (("GROQ_API_KEY", "groq"), ("NVIDIA_API_KEY", "nvidia")):
            if os.environ.get(env):
                key, base = os.environ[env], base or PROVIDERS[prov][0]
                candidates = candidates or PROVIDERS[prov][1]
                break
    if not key:
        sys.exit("No API key found. Set GROQ_API_KEY, NVIDIA_API_KEY, or LLM_API_KEY (+ LLM_BASE_URL, LLM_MODEL).")
    if not base or not candidates:
        sys.exit("LLM_API_KEY also needs LLM_BASE_URL and LLM_MODEL.")
    client = OpenAI(api_key=key, base_url=base)
    try:                                              # pick the first preferred model the key can use
        available = {m.id for m in client.models.list().data}
        usable = [m for m in candidates if m in available]
        if usable:
            return client, usable[0]
        print(f"Preferred model(s) {candidates} not available on this key.")
    except Exception as e:
        print("Could not list models:", str(e)[:100])
    return client, candidates[0]


def answer(client, model, retriever, cfg, question, k=5):
    chunks = retriever.retrieve(cfg, question, k)
    context = "\n\n".join(f"[{i + 1}] ({c['doc']}, p.{c['page_start']}) {c['text']}" for i, c in enumerate(chunks))
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": ANSWER_SYSTEM},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    return r.choices[0].message.content, chunks


# ---- evaluation -------------------------------------------------------------
def locate(doc_norm, evidence):
    """Return (doc, 1-based page) where the evidence span starts.
    Pass 1 looks inside single pages; pass 2 handles spans that cross a page break.
    (Checking page+next page in one pass would report the previous page for evidence
    that sits wholly on the next page.)"""
    ev = f" {norm(evidence)} "
    for d, pages in doc_norm.items():
        for p, t in enumerate(pages):
            if ev in f" {t} ":
                return d, p + 1
    for d, pages in doc_norm.items():
        for p in range(len(pages) - 1):
            if ev in f" {pages[p]} {pages[p + 1]} ":
                return d, p + 1
    return None


def verify_questions(gold, pages):
    doc_norm = {n: [norm(t) for t in pg] for n, pg in pages.items()}
    assert len(gold) >= 15, "Need at least 15 questions"
    for i, g in enumerate(gold, 1):
        n = len(g["evidence"].split())
        assert 10 <= n <= 25, f"Q{i}: evidence must be 10-25 words (it has {n})"
        loc = locate(doc_norm, g["evidence"])
        assert loc, f"Q{i}: evidence not found in the extracted text. Copy it from the extracted text, not the PDF viewer."
        g["doc"], g["page"] = loc
    return gold


def relevant_ids(tri_list, evidence, thr=0.8):
    ev = trigrams(norm(evidence).split())
    return {i for i, t in enumerate(tri_list) if len(ev & t) / len(ev) >= thr}


def first_rank(ranked, rel):
    for pos, i in enumerate(ranked, 1):
        if i in rel:
            return pos
    return None


def recall_at(ranks, k):
    return 100 * np.mean([r is not None and r <= k for r in ranks])


def mrr(ranks, k=10):
    return 100 * np.mean([1 / r if (r is not None and r <= k) else 0 for r in ranks])


def cmd_evaluate(args):
    import pandas as pd
    from tqdm.auto import tqdm

    paths = fetch_pdfs(args.pdf_dir)
    pages = load_pages(paths)
    gold = verify_questions(json.load(open(args.questions)), pages)
    ret = Retriever(pages, list(CHUNK_CONFIGS))

    rel = {cfg: [relevant_ids(ret.index[cfg]["tri"], g["evidence"]) for g in gold] for cfg in CHUNK_CONFIGS}
    for cfg, r in rel.items():
        assert all(len(x) > 0 for x in r), f"{cfg}: some evidence is not inside any chunk. Shorten that evidence span."

    ranked = {}
    for cfg in CHUNK_CONFIGS:
        out = {m: [] for m in METHODS}
        for g in tqdm(gold, desc=cfg):
            d, b = ret.dense(cfg, g["q"]), ret.bm25(cfg, g["q"])
            h = rrf([d, b])
            rr = ret.rerank(cfg, g["q"], h[:RERANK_POOL]) + h[RERANK_POOL:]
            out["dense"].append(d); out["bm25"].append(b); out["hybrid_rrf"].append(h); out["hybrid+rerank"].append(rr)
        ranked[cfg] = out
    ranks = {cfg: {m: [first_rank(ranked[cfg][m][q], rel[cfg][q]) for q in range(len(gold))] for m in METHODS}
             for cfg in CHUNK_CONFIGS}

    rows = []
    for cfg, (size, _) in CHUNK_CONFIGS.items():
        kb = max(1, round(CONTEXT_BUDGET_WORDS / size))   # same amount of text for both chunk sizes
        for m in METHODS:
            rk = ranks[cfg][m]
            rows.append({"chunk_size": cfg, "method": m, "R@1": recall_at(rk, 1), "R@3": recall_at(rk, 3),
                         "R@5": recall_at(rk, 5), "R@budget": recall_at(rk, kb), "k_budget": kb, "MRR@10": mrr(rk)})
    results = pd.DataFrame(rows).round(1)
    print("\n=== Retrieval results (", len(gold), "questions; one question = ", round(100 / len(gold), 1), "points) ===")
    print(results.to_string(index=False))
    print("\n=== Assignment table: recall@5, dense retrieval ===")
    print(results[results.method == "dense"][["chunk_size", "R@5", "R@budget", "k_budget", "MRR@10"]].to_string(index=False))

    best = max(CHUNK_CONFIGS, key=lambda c: recall_at(ranks[c]["dense"], 5))
    print(f"\n=== Dense vs BM25 disagreements at k=5 ({best}) ===")
    n_diff = 0
    for q, g in enumerate(gold):
        d_hit = ranks[best]["dense"][q] is not None and ranks[best]["dense"][q] <= 5
        b_hit = ranks[best]["bm25"][q] is not None and ranks[best]["bm25"][q] <= 5
        if d_hit != b_hit:
            n_diff += 1
            print(f"Q{q + 1}: {'DENSE only' if d_hit else 'BM25 only'} | {g['q']}")
    if not n_diff:
        print("Dense and BM25 agreed on every question at k=5.")

    print("\n=== Two retrieval failures (first two questions dense misses in the top 5) ===")
    seen, picks = set(), []
    for cfg in CHUNK_CONFIGS:
        for q in range(len(gold)):
            r = ranks[cfg]["dense"][q]
            if (r is None or r > 5) and q not in seen:
                seen.add(q); picks.append((cfg, q))
    for cfg, q in picks[:2]:
        g, chunks = gold[q], ret.index[cfg]["chunks"]
        print(f"\nQ{q + 1} | {cfg}\nQuestion : {g['q']}\nEvidence : {g['evidence']}  [{g['doc']} p.{g['page']}]")
        print("Rank of correct chunk (None = not in top 50):", {m: ranks[cfg][m][q] for m in METHODS})
        print("Top-3 dense results instead:")
        for i in ranked[cfg]["dense"][q][:3]:
            c = chunks[i]
            print(f"   - [{c['doc']} p.{c['page_start']}] {c['text'][:200]}...")

    os.makedirs(args.out_dir, exist_ok=True)
    results.to_csv(os.path.join(args.out_dir, "retrieval_results.csv"), index=False)
    json.dump(gold, open(os.path.join(args.out_dir, "test_questions_with_sources.json"), "w"), indent=2)
    print(f"\nSaved results to {args.out_dir}/")


def cmd_ask(args):
    paths = fetch_pdfs(args.pdf_dir)
    pages = load_pages(paths)
    cfg = args.chunk_size
    ret = Retriever(pages, [cfg])
    client, model = get_llm()
    print("Answer model:", model)
    ans, chunks = answer(client, model, ret, cfg, args.question)
    print("\nQ:", args.question, "\nA:", ans)
    print("Sources:", [f"[{i + 1}] {c['doc']} p.{c['page_start']}-{c['page_end']}" for i, c in enumerate(chunks)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", default="pdfs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="recall@k for two chunk sizes, plus failure analysis")
    ev.add_argument("--questions", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json"))
    ev.add_argument("--out-dir", default="results")
    ev.set_defaults(fn=cmd_evaluate)
    ak = sub.add_parser("ask", help="answer one question with the full pipeline")
    ak.add_argument("question")
    ak.add_argument("--chunk-size", choices=list(CHUNK_CONFIGS), default="large_300w")
    ak.set_defaults(fn=cmd_ask)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
