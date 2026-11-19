# Task 1: LoRA fine-tuning for text-to-JSON

Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA so that it turns a plain-English meeting request into a fixed JSON schema, then compares it against the untouched base model on 100 held-out examples.

```
"Cancel the code review with Arjun scheduled for August 20, 2026 at 8:15 am, low priority."
  ->  {"action": "cancel", "title": "code review", "date": "2026-08-20", "time": "08:15",
       "duration_min": 60, "attendees": ["Arjun"], "priority": "low"}
```

## How it works

1. **Data.** 600 JSON specs are generated programmatically with a fixed seed, so the labels are exact. An LLM (any OpenAI-compatible API) only rewrites each spec as a natural request. A rewrite is kept only if it still contains every attendee name, the month and the day; otherwise a template sentence is used. Split: 450 train, 50 validation, 100 test (never used for training or tuning).
2. **Training.** LoRA on all seven linear layers of the model, loss computed on the JSON answer only.
3. **Evaluation.** Greedy decoding on the 100 test examples, three runs: base with the same short prompt the tuned model sees, base with a long prompt that spells out the schema (the best-effort prompting baseline), and the fine-tuned model.
4. **Memory estimate** for training a 7B model three ways (full fine-tuning, LoRA, QLoRA).

## Results

Run on a free Colab T4, 100 held-out examples (one example = 1 percentage point).

| Metric | Base, short prompt | Base, schema prompt | Fine-tuned |
|---|---:|---:|---:|
| Strict valid JSON | 2.0 | 67.0 | **100.0** |
| Exact match (all 7 fields) | 0.0 | 22.0 | **94.0** |
| Field accuracy (mean of 7 fields) | 1.9 | 81.6 | **99.1** |
| `action` | 0.0 | 95.0 | 100.0 |
| `title` | 0.0 | 78.0 | 100.0 |
| `date` | 5.0 | 100.0 | 100.0 |
| `time` | 0.0 | 60.0 | 100.0 |
| `duration_min` | 0.0 | 75.0 | 95.0 |
| `attendees` | 2.0 | 100.0 | 100.0 |
| `priority` | 6.0 | 63.0 | 99.0 |

"Strict" means the whole reply parses as JSON. The base model with the short prompt does produce extractable JSON 98% of the time, but typically wrapped in a markdown code fence or extra text (see the sample outputs in the notebook), which is why strict validity is only 2%. Full table: [`results/results_base_vs_finetuned.csv`](results/results_base_vs_finetuned.csv).

Training took 1 min 14 s (87 optimizer steps) and peaked at 7.51 GB of GPU memory. Loss per epoch: [`results/training_log.csv`](results/training_log.csv) (validation loss 0.0028 → 0.0015 → 0.0008, still falling at epoch 3, no sign of overfitting).

## LoRA settings and why

| Setting | Value | Reason |
|---|---|---|
| Rank `r` | 16 | Narrow task with a fixed schema; 8–16 is enough. 16 leaves room for the date and time handling without overfitting 450 examples. |
| `lora_alpha` | 32 | Scaling alpha/r = 2, a common default that keeps update size stable. |
| Target modules | q, k, v, o, gate, up, down | Adapting attention and MLP layers tends to beat attention-only (QLoRA paper). On a 0.5B model the extra parameters are small: 8,798,208 trainable of 502,830,976 (1.75%). |
| Dropout | 0.05 | Light regularisation for a small dataset. |
| Learning rate | 2e-4, cosine, 10 warm-up steps | Typical for LoRA; adapters are small, so a higher rate than full fine-tuning is fine. |
| Epochs | 3 | Training loss was already very low by epoch 2 and validation loss kept improving through epoch 3. |
| Batch size | 8 × 2 accumulation = 16 | Fits a T4. |
| Precision | fp32 base weights with fp16 autocast | The T4 has no bf16; this avoids the "unscale FP16 gradients" error. |

## Failure analysis

The fine-tuned model got 6 of 100 examples wrong: 5 on `duration_min`, 1 on `priority` ([`results/failures.json`](results/failures.json)). Checking them against the data showed that none is a real reasoning error.

**Duration is not in the request (5 failures).** All five are `cancel` requests that never mention a length, for example *"Please cancel the quarterly planning with Kavya on October 21, 2026 at 9:30 am, regular priority."* The label says 60 minutes, the model guessed 45. There is nothing in the text to infer it from. In the test set exactly 5 of the 33 cancel requests omit the duration, and these are the five the model missed. The other four are the same situation: the model picks a plausible length (45, 90, 45, 90) that differs from the label (60, 45, 15, 15).

**Contradictory label (1 failure).** *"Cancel the quarterly planning meeting with Divya on October 26, 2026 at 5:00 pm for one hour, no rush."* is labelled `medium`; the model said `low`. In the rest of the dataset "no rush" maps to `low`, and this is the only medium example containing low-priority wording. The rewrite check does not verify priority wording, so the LLM paraphrase introduced the mismatch. The model's answer follows the text.

Put together: 94% exact match is what is reachable on this test set (100 − 5 unanswerable − 1 mislabelled), and every example whose answer can be recovered from the text and is correctly labelled was answered correctly.

## Memory needed to train a 7B model

Estimates, with the working in `memory_estimate()` in `main.py`.

| Method | Calculation | Total |
|---|---|---:|
| Full fine-tuning | 7B × 16 bytes (fp16 weights 2 + fp16 grads 2 + fp32 master copy 4 + Adam m 4 + Adam v 4) = 112 GB, plus ~10 GB activations | **~122 GB** |
| LoRA (r=16, all linear layers) | frozen fp16 weights 14 GB + 40.0M trainable params × 16 bytes = 0.64 GB + ~4 GB activations | **~18.6 GB** |
| QLoRA | 4-bit NF4 weights 3.5 GB (+0.2 GB quantisation constants) + 0.64 GB LoRA state + ~2.5 GB activations with gradient checkpointing | **~6.8 GB** |

The 40.0M figure comes from a Llama-7B shape (32 layers, hidden 4096, MLP 11008): 78,080 × r parameters per layer. The activation numbers are assumptions (batch 1–4, sequence 512), not measurements. The one measured number is the 0.5B LoRA run above (7.51 GB peak); it is a different model and precision setup (fp32 base weights, batch size 8), so it is not directly comparable to the 7B estimates.

## Run it

```bash
cd task1-lora-finetuning
pip install -r requirements.txt

# API key for the data step (pick one)
export GROQ_API_KEY=...          # default model: openai/gpt-oss-120b
export NVIDIA_API_KEY=...        # default model: meta/llama-3.3-70b-instruct (check the name on build.nvidia.com)
# any other OpenAI-compatible service:
# export LLM_API_KEY=... LLM_BASE_URL=https://.../v1 LLM_MODEL=...

python main.py                   # data -> train -> evaluate -> memory
python main.py --stage evaluate  # one stage at a time: data | train | evaluate | memory
python main.py --no-llm          # template-only dataset, no API key (results will not be comparable)
```

Needs a CUDA GPU (a free Colab T4 is enough). On Colab: `!pip install -r requirements.txt` then `!python main.py`. If PEFT complains about `torchao`, run `pip install -U "torchao>=0.16.0"`.

The dataset in `data/all.json` is the one used for the reported numbers; if the file exists, `main.py` reuses it, so you get the same train/validation/test split. Delete it to regenerate (the specs are identical because of the seed, but the LLM wording will differ).

## Layout

```
main.py                     pipeline (data, train, evaluate, memory)
requirements.txt
data/all.json               600 labelled examples used in the reported run
results/                    reference results from the Colab run (CSV, JSON, adapter_config.json)
notebook/                   the Colab notebook the script was derived from
```

The trained adapter (about 35 MB) is not committed. Re-create it with `--stage train`, or upload it to the Hugging Face Hub.

## Limitations

- The data is synthetic and the templates are narrow (16 names, 15 titles, 2026 dates). The 99% field accuracy says little about messier real-world requests.
- 100 test examples and one training run, so differences of a point or two are noise.
- The data check does not confirm that duration and priority are actually stated in the request. That caused all six errors above. Requiring a duration phrase for every example, and checking priority wording, would remove most of them.
- Tested versions in the Colab run: torch 2.14.1, transformers 5.23.0, peft 0.21.3, Python 3.13.
