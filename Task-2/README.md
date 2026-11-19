# Task 2: Retrieval-augmented QA from scratch

A question-answering pipeline over 112 pages of PDF text, built without LangChain or LlamaIndex. Chunking, BM25, rank fusion, the retrieval metrics and the answer prompt are written in plain Python and NumPy. Third-party libraries are used only for the embedding model, the FAISS index, the cross-encoder and PDF text extraction.

```
PDF -> clean text -> word chunks -> dense search (FAISS)  --\
                                 -> BM25 (written here)  ---+-> RRF -> cross-encoder re-rank -> LLM answer with citations
```

| Piece | What is used |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers), query prefix on queries only |
| Vector search | FAISS `IndexFlatIP` on normalised vectors (cosine, exact search) |
| Keyword search | BM25 with `k1=1.5`, `b=0.75`, dict-based inverted index (`BM25` class in `main.py`) |
| Fusion | Reciprocal Rank Fusion, `score = sum 1/(60 + rank)` |
| Re-ranker | `BAAI/bge-reranker-base` cross-encoder over the top 30 fused candidates |
| Answer model | any OpenAI-compatible chat model; the reported run used `openai/gpt-oss-120b` on Groq |

## Documents

Three public arXiv papers, 112 pages in total (requirement: at least 30).

| Document | Pages | Words |
|---|---:|---:|
| Llama 2: Open Foundation and Fine-Tuned Chat Models (2307.09288) | 77 | 42,284 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP (2005.11401) | 19 | 9,839 |
| BERT (1810.04805) | 16 | 9,824 |

Chunk sizes compared: **100 words** (25 overlap, 826 chunks) and **300 words** (60 overlap, 258 chunks).

## Test set

[`questions.json`](questions.json) holds 15 questions, each with an *evidence span* of 10–25 words copied from the extracted text. Using the span as ground truth, rather than a chunk id, lets the same questions score both chunk sizes. A retrieved chunk counts as correct if it contains at least 80% of the span's word trigrams. The questions are spread over all three documents and mix keyword-style, paraphrased and number-heavy items.

## Results

Hit rate at k (recall@k) and MRR@10, in %. With 15 questions, **one question is worth 6.7 points**. `R@budget` gives each chunk size the same amount of text (about 1,500 words: top 15 small chunks or top 5 large chunks), which removes the advantage large chunks get from showing more text per hit.

| Chunk size | Method | R@1 | R@3 | R@5 | R@budget | MRR@10 |
|---|---|---:|---:|---:|---:|---:|
| 100 words | dense | 53.3 | 80.0 | 80.0 | 93.3 | 66.7 |
| 100 words | BM25 | 53.3 | 80.0 | 86.7 | 93.3 | 67.8 |
| 100 words | hybrid (RRF) | 60.0 | 80.0 | 86.7 | 93.3 | 72.2 |
| 100 words | hybrid + re-rank | 53.3 | 66.7 | 73.3 | 93.3 | 62.6 |
| 300 words | dense | 66.7 | 80.0 | 86.7 | 86.7 | 74.6 |
| 300 words | BM25 | 73.3 | 86.7 | 86.7 | 86.7 | 80.0 |
| 300 words | hybrid (RRF) | 66.7 | 86.7 | **93.3** | 93.3 | 78.0 |
| 300 words | hybrid + re-rank | 46.7 | 73.3 | **93.3** | 93.3 | 64.3 |

Assignment table (recall@5, dense retrieval): 100 words **80.0%**, 300 words **86.7%**. Full CSV: [`results/retrieval_results.csv`](results/retrieval_results.csv).

### What the numbers say

**Chunk size.** For dense retrieval the 300-word chunks are better on recall@5 (86.7 vs 80.0), R@1 (66.7 vs 53.3) and MRR (74.6 vs 66.7). Part of that is simply more text per hit: at equal text budget the 100-word chunks come out ahead (93.3 vs 86.7). Each gap is one or two questions, so the honest conclusion is that neither size clearly wins on this test set. The ranking-quality advantage of large chunks is the more consistent signal.

**Hybrid search.** Fusing dense and BM25 gave the best recall@5 at both sizes: 86.7 at 100 words (tied with BM25, one question above dense) and 93.3 at 300 words (one question above either method alone). At 300 words BM25 alone had the best MRR (80.0, against 78.0 for the fusion).

**Re-ranker.** In this setup the cross-encoder did not help. R@5 stayed at 93.3 for 300-word chunks and fell from 86.7 to 73.3 for 100-word chunks; R@1 dropped from 66.7 to 46.7 at 300 words. It fixed one of the two failures below, not the other. Possible causes I did not test: much of the evidence is table or number text that PDF extraction flattens, which a prose-trained re-ranker handles badly, and 300-word chunks of number-heavy text can exceed the re-ranker's 512-token limit and get truncated.

### Where dense and BM25 disagree (300-word chunks, top 5)

- **BM25 only:** *"What zero-shot exact-match result does MPT 7B achieve on the SQuAD benchmark?"* The answer sits in a table row full of exact strings (`MPT 7B`, `SQuAD`, `59.5`), which embeddings tend to blur.
- **Dense only:** *"What is the main advantage of BERT's bidirectional pre-training compared to left-to-right models?"* The question is a paraphrase; the passage says the model conditions "on both left and right context", so beyond "BERT" and "bidirectional" (words that appear throughout the paper) there is little for BM25 to match.

### Two retrieval failures (100-word chunks)

| | Question | Rank of the correct chunk (dense / BM25 / hybrid / +re-rank) |
|---|---|---|
| Q4 | How many single-turn and multi-turn prompts were allocated to ChatGPT in the human evaluation? | 6 / 7 / 5 / **3** |
| Q13 | What pre-training objective does BERT use to learn deep bidirectional representations? | 14 / 5 / 8 / 8 |

*Q4.* The evidence is a table row (`ChatGPT 1917 2256`). The top dense results were the prose on page 18 that introduces the human evaluation in general terms ("over 4,000 single and multi-turn prompts"), plus a reference-list chunk. The table row with the per-model counts ranked 6th. Hybrid search moved it to 5th and the re-ranker to 3rd, so the later stages of the pipeline fix this one.

*Q13.* All three top dense hits came from the BERT paper, so the topic was right but the specific passage was not: the sentence that names the "masked language model" objective ranked 14th. BM25 ranked it 5th, fusion lifted it only to 8th and the re-ranker did not move it. Untested fixes: larger chunks or a larger k, query expansion, or sentence-aware chunking so the sentence is not diluted by neighbouring text.

## Short explanations

**BM25.** It scores each chunk by how many of the query's words it contains, weighting rare words more (inverse document frequency), letting repeated words count with diminishing returns (`k1`), and correcting for chunk length (`b`). It is strong when the answer depends on an exact string such as a model name, a number or an acronym, as in the MPT 7B example. It cannot match a paraphrase, which is where dense embeddings do better, so the two fail in different places and combining them with RRF is more reliable than either alone.

**Re-ranker.** A bi-encoder embeds the query and each chunk separately and compares the vectors, which is fast enough to search the whole corpus but coarse. A cross-encoder reads the query and one chunk together in a single pass and scores the pair directly, so it can judge fine detail, but it is far too slow to run on every chunk. The usual pattern is to retrieve 30–50 candidates cheaply, re-order them with the cross-encoder, and pass the top 3–5 to the LLM. Here it helped on Q4 but lowered overall R@1 and MRR, so on this corpus it is a net negative.

## Answer generation

`python main.py ask "..."` runs hybrid search, re-ranking, and asks the LLM to answer only from the numbered passages and cite them. Examples from the reported run:

- *Per-GPU power limits for the RSC and production clusters?* → 400 W and 350 W, cited to the correct Llama 2 passage.
- *Which pizza style does the speaker praise as the best?* → Chicago-style, cited.
- *What is the capital of France?* → `Not found in the provided documents.` (the question is deliberately outside the documents, and the model refused instead of guessing).

The `gpt-oss` model writes its own citation markers (`【2†L9-L12】`) in addition to the `[1]` style the prompt asks for. This is the model's behaviour, not part of the pipeline.

## Run it

```bash
cd task2-rag-from-scratch
pip install -r requirements.txt

export GROQ_API_KEY=...          # or NVIDIA_API_KEY, or LLM_API_KEY + LLM_BASE_URL + LLM_MODEL
python main.py evaluate          # recall@k table, disagreements, two failure cases -> results/
python main.py ask "What pre-training objective does BERT use?"
python main.py ask "..." --chunk-size small_100w
```

`evaluate` does not need an API key; only `ask` does. The first run downloads the three PDFs into `pdfs/` and the two Hugging Face models (the re-ranker is the larger, around 1 GB). A GPU speeds up embedding but is not required.

To use your own documents, put the PDFs into `pdfs/` (at least 30 pages in total), edit the `DOCS` dictionary in `main.py` and replace `questions.json`. `evaluate` checks that every evidence span is 10–25 words and exists in the extracted text.

## Layout

```
main.py             pipeline, evaluation and answer generation
questions.json      15 questions with evidence spans
requirements.txt
results/            retrieval_results.csv from the reported run
notebook/           the Colab notebook the script was derived from
```

## Notes

- The reported numbers come from the Colab notebook. `main.py` uses the same chunking, retrieval and metrics code, but the full script has not been re-run end to end after the refactor.
- The notebook printed the source page of each evidence span one page too low whenever the span sat wholly on the following page (for example Q1 was listed as p.23; the passage is on p.24). `locate()` in `main.py` fixes this. Retrieval metrics are not affected, because relevance is decided by trigram overlap, not by page number.
- The answer model is chosen from the models your key can use. The reported run used `openai/gpt-oss-120b` because the default Llama model was not available on that key.
- Evaluation set size is small (15 questions), so small differences between methods should not be over-read.
