# Community benchmark: GTX 1660 SUPER with 6 GB

Gemma 4 26B-A4B on a card with less VRAM than PolyStrata supports. What setup's
estimate said and what the engine actually did, and the speeds between them. One
card, one model, one size, two runs; the limit of it is stated at the end.

## Hardware and software

- GPU: NVIDIA GeForce GTX 1660 SUPER, 6,144 MiB, compute capability 7.5 (Turing);
  display attached (`display_active: Enabled`), compute mode Default
- CPU: AMD Ryzen 5 3600, 6 cores / 12 threads, AVX2, no AVX-512
- RAM: 31.9 GiB; page file 8 GB
- PCIe link: 3.0 x16. Disk: NVMe, model and data on `I:`
- OS: Windows 10 build 19044; driver 580.97; CUDA 12.4 and 13.0 toolkits present
- PolyStrata: this fork, setup.py 0.1.42 line; `strata-poly` **release build**
  (`strata-poly-windows-x64-965bc5f225bb33ed.zip`, SHA-256 verified against
  GitHub's), built for sm_75, sm_80, sm_86, sm_89 and sm_120. Nothing compiled.
  NVIDIA's CUDA 13.0 runtime and cuBLAS from pip.
- Background: the desktop on the same card. No other workload.

## Model and configuration

- `unsloth/gemma-4-26B-A4B-it-GGUF`, `gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf`
  (10.55 GB, SHA-256 verified), plus Google's draft model
  `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (0.46 GB)
- Context **16,384** (was 8,192 first; see the comparison below); no images; no conversation cache;
  **no `--tail-skip`** (Gemma 4 has no measured default for it in this repository)
- 12 CPU threads, the engine's own count

```text
I:\workspace\PolyStrata\engine-poly\strata-poly.exe --serve \
  --model I:\workspace\PolyStrata-data\models\gemma4-UD-Q2_K_XL\gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf \
  --max-context 8192 \
  --expert-profile I:\workspace\PolyStrata-data\packs\gemma4-ud-q2_k_xl\expert-profile.txt \
  --draft-model I:\workspace\PolyStrata-data\models\gemma4-UD-Q2_K_XL\mtp-gemma-4-26B-A4B-it-Q8_0.gguf
```

## Method

`polystrata.py benchmark --model gemma-4-26b-a4b-ud-q2_k_xl`, greedy, 256 tokens
of answer, through the server. Each run sends a short prompt and a ~2,000-token
prompt; the 8,000-token prompt was skipped, the context is 8,192. Two runs back
to back with the engine already loaded and its expert profile already filled, so
run 2 starts warm. Raw per-run output is `run1.json`/`run2.json` (8,192 context) and
`ctx16384-run1.json`/`ctx16384-run2.json` (16,384). Times are
the engine's own clock. VRAM and RAM are the machine's peak use while the request
ran, not the engine's alone.

## Results, context 16,384

| Prompt | Tokens in | TTFT s | Reads tok/s | Writes tok/s | Experts from VRAM | MTP accepted | VRAM GiB | RAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short (run 1) | 26 | 0.925 | 32.9 | 39.2 | 58.3% | 69.1% | 5.27 | 20.11 |
| short (run 2) | 26 | 0.957 | 30.0 | 37.9 | 44.3% | 71.9% | 5.27 | 20.32 |
| 2,000 (run 1) | 2,838 | 17.435 | 163.7 | 35.0 | 44.4% | 73.8% | 5.29 | 20.47 |
| 2,000 (run 2) | 2,838 | 17.551 | 162.5 | 37.2 | 42.0% | 76.4% | 5.27 | 20.55 |
| 10,000 (run 1) | 10,982 | 61.738 | 145.0 | 36.7 | 50.1% | 80.8% | 5.29 | 21.09 |
| 10,000 (run 2) | 10,982 | 62.110 | 144.1 | 35.7 | 49.6% | 77.7% | 5.27 | 21.10 |

At load the engine reported **533 of 3,840 routed experts in VRAM (1.1 GiB)**, in
2.1 s, with 12 CPU threads. `nvidia-smi` during the runs held 5.27 to 5.29 GiB of
the card's 6,144 MiB, leaving about 0.9 GiB.

### What these writes are not

**All six runs spent the whole 256-token budget thinking and answered nothing.**
Gemma 4 at effort `high` writes 1,400 to 1,800 characters of reasoning before it
starts, so the benchmark's answer cap is below its thinking; the server logged
"the reply reached max tokens while still thinking, so it has no answer" for each
of the six. The writes column is the speed of *generation*, not of a completed
reply.

It matches replies that did finish, on the same engine with room to answer:
38.7 to 42.0 tok/s over 454- and 1,120-token replies (`finish: stop`). The number is
sound as a generation rate, but it is not a figure for a reply that ended.
`reasoning_budget_tokens` is what leaves room for the answer.

### Context 8,192 against 16,384

| | 8,192 | 16,384 |
| --- | ---: | ---: |
| Experts in VRAM at load | 532 | 533 |
| VRAM held while answering | 5.23 to 5.31 GiB | 5.27 to 5.29 GiB |
| Writes, short prompt | 37.7 to 41.2 | 37.9 to 39.2 |
| Writes, 2,838 tokens | 37.5 to 39.7 | 35.0 to 37.2 |
| RAM peak | 20.05 to 20.40 GiB | 20.11 to 21.10 GiB |
| 10,982-token prompt | **skipped**, does not fit | 144 tok/s read, 35.7 to 36.7 written |

Doubling the context cost nothing measurable: the same experts, the same VRAM, and
writes inside the spread of two runs at 8,192. RAM rose about 0.7 GiB at the
longest prompt. 16,384 is kept because it is free and buys the long-prompt row,
not because it is faster. Setup's own estimate says this size needs 6.16 GB of the
card's 6.0 GB at 16,384 and asks to go on anyway; it loaded without trouble because
the engine sizes the expert cache for the context as it grows, not for the maximum.

## What this says about setup's estimate

Setup's estimate for this card at 8,192 tokens was **about 25% of the experts and
2.2 GB**. The engine put **14% and 1.1 GiB** - a little over half the estimate.
The estimate is `vram_beside_gb` (measured on an RTX 5090 at 32K) less the KV
cache at the context asked. It has no term for what this card spends beyond the
weights and the caches: the desktop it drives, the driver's own use, and the
engine's own reserve (512 MiB by default). Those cost about 1.1 GiB here.

So on a card under 8 GB, the share of experts setup prints is an upper bound, and
the estimate should not be used to predict speed from VRAM alone. It was also
pessimistic about the context: it says the size does not fit at 16,384, and the
engine loaded there with the same expert cache it had at 8,192.

## Correctness and limitations

One smoke test only: "In one short sentence: what is the capital of France?"
answered "The capital of France is Paris." in 1.4 s. **No needle test, no
HumanEval, no coding tasks, no tool calls and no image questions were run.** This
report is about speed and about the VRAM estimate, not about answer quality.

Not measured here: Qwen3.6-35B-A3B and Ornith 1.5, the other two families setup
says fit this card at 8,192; any context above 16,384; images (`--vision cpu`);
prompts longer than 10,982 tokens; a run after a reboot (cold file cache);
`--tail-skip` at any value, which is the obvious next thing to try on a card where
86% of the experts are on the CPU; and a 6 GB card that does not drive a desktop.

Reproduce: `python setup.py --family gemma4 --model UD-Q2_K_XL --context 16384
--vision no`, then `polystrata.py benchmark --model gemma-4-26b-a4b-ud-q2_k_xl`.