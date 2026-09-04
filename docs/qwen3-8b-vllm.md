# Qwen3-8B through tunneled vLLM

Qwen3-8B is served by vLLM's native Anthropic API, exactly like QED-Nano;
LiteLLM is not involved. The harness model id is `vllm/qwen3-8b`: the prefix
selects the endpoint and is stripped before `qwen3-8b` is sent in the Anthropic
request. Everything in `docs/qed-nano-vllm.md` about the tunnel applies here —
only the job script, the API key, and the context settings differ.

## 1. Start the ORCD job

```bash
mkdir -p /orcd/pool/008/notadib/vllm/logs/slurm
sbatch /orcd/pool/008/notadib/vllm/scripts/serve_qwen.sbatch
squeue -u notadib
```

Read the assigned compute node from the job log after the job starts:

```bash
tail -f /orcd/pool/008/notadib/vllm/logs/slurm/qwen_<job-id>.log
```

Wait until vLLM reports that the server is listening on port 8000. The
submitted script and local harness both use the fixed key `qwen-local-key`.

## 2. Open the tunnel from the Mac

Replace `<node>` with the `NODE:` value printed in the log:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 0.0.0.0:8000:<node>:8000 \
  mitengage
```

Verify it:

```bash
curl -fsS -H "Authorization: Bearer qwen-local-key" http://127.0.0.1:8000/health
curl -fsS -H "Authorization: Bearer qwen-local-key" http://127.0.0.1:8000/v1/models
```

## 3. Configure and run the harness

Add these git-ignored values to `.env`:

```bash
VLLM_API_KEY=qwen-local-key
VLLM_BASE_URL=http://host.docker.internal:8000
```

Run one problem first, then the real arm:

```bash
python scripts/prepare_aime26.py   # once: writes local_data/, git-ignored

./run.sh run --dataset aime26 --dataset-dir local_data \
  --arm baseline --model vllm/qwen3-8b --problems aime-2026-01 --seeds 1

./run.sh run --dataset aime26 --dataset-dir local_data \
  --arm baseline --model vllm/qwen3-8b --seeds 1,2,3
./run.sh audit --dataset aime26 --dataset-dir local_data \
  --arm baseline --model vllm/qwen3-8b --seeds 1,2,3
```

Drop `--dataset-dir` once the two files are uploaded to the dataset repository.

Results land under `results-aime26/vllm-qwen3-8b/` for `--dataset aime26` and
under `results/vllm-qwen3-8b/` for the default dataset.

## Context and compaction

Qwen3-8B trains to 40,960 positions with `rope_scaling: null`, which is far
below every other route in this harness — a 200k-output-token attempt would
otherwise spend most of its life compacting. `scripts/serve_qwen.sbatch`
therefore serves it with YaRN at factor 4, giving a 131,072-token window, and
`QWEN3_8B_AUTO_COMPACT_WINDOW` compacts the transcript at 100k so the summary,
prompt, tools, and next response still fit. Other `vllm/*` routes keep the
262k-context default of 200k.

Two caveats to disclose when reporting Qwen3-8B numbers:

- YaRN in vLLM is a static scaling applied to **every** request, short ones
  included. Qwen documents a small short-context accuracy cost for this. To
  measure the unextended model instead, drop `--rope-scaling`, set
  `--max-model-len 40960`, and lower the compaction window to about 30k.
- Qwen3-8B compacts a long sequential trajectory more often than a
  larger-context model. As with QED-Nano, this does not uniquely invalidate its
  8x arm — the 1.6M-output-token protocol already exceeds every model's context
  — but compare compute scaling primarily within each fixed model-agent system.

Sampling: Qwen recommends `temperature=0.6, top_p=0.95, top_k=20` in thinking
mode and warns against greedy decoding, which can produce endless repetition.

## Answer-graded datasets

Qwen3-8B is the intended solver for `--dataset aime26`, which is graded by
final-answer equivalence rather than proof review. That dataset publishes no
oracle hints, outlines, or reference proofs, so only the no-hint arms
(`baseline`, `baseline-sequential`, `baseline-parallel`,
`baseline-uniform-strategy`) run, and the state-annotation stage is skipped.
Every other arm is refused up front, naming the missing artifact.
