# Task 3: Running a model locally in two quantisation settings

Benchmarks `Qwen2.5-3B-Instruct` served by [Ollama](https://ollama.com) at two precisions, **fp16** (reference) and **q4_K_M** (4-bit), on speed, GPU memory and answer quality, and then works out the KV-cache cost and a plan for serving 20 concurrent users on one 24 GB GPU.

The model under test always runs locally. An external LLM (Groq, NVIDIA or any OpenAI-compatible API) is used only as a judge that grades the answers.

## What is measured and what is estimated

| Part | Source |
|---|---|
| Speed, memory, load time | **Measured** on a Colab T4 (15 GB) |
| Answer quality | **Measured** (automatic checks plus a blind LLM judge) |
| KV-cache sizes, 20-user plan | **Calculated** from model configs; the vLLM setup was not run (no 24 GB GPU available) |

Same settings for both runs: temperature 0, seed 42, context 4096, at most 256 new tokens, the same 10 prompts.

## Results: speed and memory

| | fp16 | q4_K_M | Ratio |
|---|---:|---:|---:|
| File size | 6.18 GB | 1.93 GB | 3.2× smaller |
| GPU memory with model loaded (`nvidia-smi`) | 6,255 MB | 2,191 MB | 2.9× smaller |
| Decode speed (mean ± std over 10 prompts) | 38.0 ± 3.3 tok/s | 75.7 ± 8.8 tok/s | 2.0× faster |
| Prompt processing | 1,122 tok/s | 1,662 tok/s | 1.5× faster |
| Load time | 16.0 s | 3.3 s | |

Decode speed is `eval_count / eval_duration` from Ollama's response, so it excludes prompt processing. Memory is the `nvidia-smi` difference before and after loading; Ollama's own `size_vram` agrees (6.41 GB and 2.16 GB, which also counts runtime buffers such as the KV cache for the 4096 context). Full CSV: [`results/benchmark_speed_memory.csv`](results/benchmark_speed_memory.csv).

Sanity check against the hardware: the T4 has about 320 GB/s of memory bandwidth, and decoding has to read every weight once per token. That caps fp16 at roughly 320 / 6.2 ≈ 52 tok/s and q4_K_M at about 320 / 1.9 ≈ 166 tok/s. The measured 38 and 75.7 tok/s sit below those ceilings (73% and 46% of them), as they should. The 4-bit model gets a smaller share, likely because dequantisation and other fixed per-token overheads matter more when the weights are small.

The q4_K_M file averages 5.0 bits per weight, not 4, because the format keeps some tensors at higher precision and stores per-block scales.

## Results: quality on the same 10 prompts

Two kinds of evidence, since neither is enough alone.

**Automatic checks** on the three prompts with a verifiable answer: the train arrival time (5:15 PM), valid JSON with the right fields, and 17 × 24 = 408. Both settings pass all three.

**Blind LLM judge** (`openai/gpt-oss-120b` via Groq). Each pair is shown in random A/B order and scored 1–5 for correctness, instruction-following and clarity.

| Prompt | fp16 | q4_K_M | Judge's note |
|---|---:|---:|---|
| 1. Explain why the sky is blue in 3 sentences | 3 | 3 | Both exceed the three-sentence limit |
| 2. Train arrival time | 5 | 5 | |
| 3. Fibonacci function with docstring | 5 | 4 | fp16 has clearer indexing and a fuller docstring |
| 4. Summarise in exactly two sentences | 2 | 2 | Both miss the vote count and the length limit |
| 5. Translate to French | 5 | 5 | |
| 6. Markdown table, process vs thread | 5 | 2 | q4_K_M table reported as malformed |
| 7. Return only JSON | 5 | 5 | |
| 8. 17 × 24 | 5 | 5 | |
| 9. Four-line poem | 5 | 5 | |
| 10. Microservices pros and cons | 5 | 5 | |
| **Mean** | **4.5** | **4.1** | |

The two settings gave identical text on 3 of the 10 prompts, so they are not interchangeable output-for-output (with greedy decoding any difference comes from the weights alone). Most of the 0.4 gap is one prompt, the markdown table, where the 4-bit model produced a malformed table; a smaller part is the code prompt. On straightforward tasks (translation, arithmetic, JSON, short writing) the two are indistinguishable. Prompts 1 and 4 are failures of both models at following a length constraint, which quantisation did not cause. With 10 prompts and one judge, treat the gap as indicative: a single prompt moves the mean by 0.1–0.3.

Raw answers are in `results/outputs.json` after a run, and the judge scores in [`results/judge_scores.csv`](results/judge_scores.csv).

## Explanations

### KV cache

When a transformer generates text, each new token attends to the keys and values of every earlier token. Recomputing them at every step would repeat the same work again and again, so they are stored (the KV cache) and only the new token's keys and values are computed. The price is memory that grows linearly with context length and with the number of simultaneous users.

```
bytes per token = 2 (K and V) x layers x KV heads x head dim x bytes per value
```

| Model | Per token (fp16) | One 4096-token sequence |
|---|---:|---:|
| Qwen2.5-3B (36 layers, 2 KV heads, head dim 128) | 36 KB | 0.15 GB |
| Llama-3.1-8B (32 layers, 8 KV heads, head dim 128) | 128 KB | 0.54 GB |

For 20 users at 4096 tokens on the 8B model that is 10.7 GB, nearly twice the ~5.7 GB the 4-bit weights need. At high concurrency the cache, not the weights, is what runs out first.

### Why quantisation saves memory

Quantisation stores each weight in fewer bits: 16-bit uses 2 bytes per parameter, 8-bit uses 1, 4-bit about 0.5, plus small per-block scales to map the integers back to approximate values. Measured here: 6.18 GB → 1.93 GB on disk and 6.3 GB → 2.2 GB on the GPU. Token generation is limited by memory bandwidth rather than arithmetic, so reading fewer bytes per token also made decoding 2× faster. The cost is a small loss of precision, visible here on one formatting-heavy prompt. Quantising the weights does not shrink the KV cache; that needs a separate setting (for example an fp8 KV cache).

## Plan: 20 concurrent users on one 24 GB GPU

**Assumptions:** Llama-3.1-8B-class model in 4-bit AWQ (about 5.7 GB, because embeddings stay in 16-bit); RTX 4090 / 3090-class card with about 1,000 GB/s bandwidth; each user up to 4096 tokens of context; vLLM as the server (continuous batching and PagedAttention, so cache memory is allocated in blocks as sequences grow instead of being reserved at the maximum length); about 2 GB for CUDA context, activations and graphs; 90% of GPU memory usable.

| Weights | KV cache | Total needed | Fits in 21.6 GB budget? |
|---|---|---:|---|
| 4-bit AWQ | fp16 (10.7 GB) | 18.4 GB | **Yes** |
| 4-bit AWQ | fp8 (5.4 GB) | 13.1 GB | Yes |
| fp16 (16.1 GB) | fp16 | 28.8 GB | No |
| fp16 | fp8 | 23.4 GB | No |

- 4-bit weights with an fp16 cache is enough for 20 users, with room for about 26 full-length users in total.
- An fp8 KV cache (`--kv-cache-dtype fp8`) halves the cache if more users or longer contexts are needed.
- Estimated speed: reading weights plus the whole cache once per step takes about 27 ms at 60% of peak bandwidth, so roughly 37 tok/s per user and about 730 tok/s in total. This assumes every user is at the full 4096 tokens, so it is conservative.
- Prefill is compute-bound, so many long prompts arriving together will queue. Chunked prefill and prefix caching (the shared system prompt is computed once) reduce that.

Launch command for the GPU machine (not run for this report; check that the model repository name is still current):

```bash
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq --max-model-len 4096 --max-num-seqs 20 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching
```

`python main.py --calc-only` reproduces every number in this section without any GPU.

## Run it

```bash
cd task3-local-quantisation
pip install -r requirements.txt

# 1. Install Ollama (https://ollama.com) and start it in another terminal
ollama serve

# 2. Benchmark both settings (downloads about 8 GB of models on the first run)
export GROQ_API_KEY=...          # judge only; or NVIDIA_API_KEY, or LLM_API_KEY + LLM_BASE_URL + LLM_MODEL
python main.py --pull

python main.py --skip-judge      # no API key needed
python main.py --calc-only       # calculations only: no Ollama, no GPU
```

An NVIDIA GPU is needed for the GPU-memory column (read through `nvidia-smi`); without it the script still runs and reports Ollama's own `size_vram`. Results are written to `results/`. Speeds and memory depend on the hardware: the numbers above are for a T4.

On Colab, install Ollama with `curl -fsSL https://ollama.com/install.sh | sh`, start it with `ollama serve &`, then run the commands above.

## Layout

```
main.py             benchmark, quality checks, judge, KV-cache and serving calculations
requirements.txt
results/            reference results from the Colab run
notebook/           the Colab notebook the script was derived from
```

## Notes

- The reported numbers come from the Colab notebook. `main.py` contains the same benchmark and calculation code; the benchmark path needs a running Ollama and has not been re-run end to end after the refactor, while `--calc-only` has been run and matches the notebook.
- The notebook's introduction names Llama-3.3-70B as the judge, but the code and the scores above use `openai/gpt-oss-120b`.
