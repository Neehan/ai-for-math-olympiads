# QED-Nano through tunneled vLLM

QED-Nano is served by vLLM's native Anthropic API. LiteLLM is not involved.
The harness model id is `vllm/qed-nano`; the prefix selects the endpoint and is
stripped before `qed-nano` is sent in the Anthropic request.

## 1. Start the ORCD job

The SLURM log directory must already exist on ORCD. From the login node:

```bash
mkdir -p /orcd/pool/008/notadib/vllm/logs/slurm
sbatch /orcd/pool/008/notadib/vllm/scripts/slurm_vllm.sh
squeue -u notadib
```

Read the assigned compute node from the job log after the job starts:

```bash
tail -f /orcd/pool/008/notadib/vllm/logs/slurm/vllm_<job-id>.log
```

Wait until vLLM reports that the server is listening on port 8000.
The submitted script and local harness both use the fixed key
`qed-local-key`; there is no generated key file to copy between machines.

## 2. Open the tunnel from the Mac

Replace `<node>` with the `NODE:` value printed in the log:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 0.0.0.0:8000:<node>:8000 \
  mitengage
```

The wildcard bind makes the Mac-side forward reachable from Docker Desktop;
a loopback-only forward is not. vLLM still requires the API key, but this
listener is reachable on the Mac's interfaces; use it only on a trusted network
with the host firewall enabled. Keep this terminal open.

Verify the tunnel on the Mac:

```bash
curl -fsS -H "Authorization: Bearer qed-local-key" \
  http://127.0.0.1:8000/health
curl -fsS -H "Authorization: Bearer qed-local-key" \
  http://127.0.0.1:8000/v1/models
```

## 3. Configure and run the harness

Add these git-ignored values to `.env`:

```bash
VLLM_API_KEY=qed-local-key
VLLM_BASE_URL=http://host.docker.internal:8000
```

Run one problem first. `run.sh` builds the harness container, permits only the
resolved tunnel IP and port through its firewall, probes `/health`, and sends
the model name `qed-nano` directly to the native Anthropic endpoint:

```bash
./run.sh run \
  --arm baseline \
  --model vllm/qed-nano \
  --problems serbia-mo-2026-03 \
  --seeds 1
```

Then run the combinatorics pilot normally:

```bash
./run.sh run \
  --arm baseline \
  --model vllm/qed-nano \
  --domain combinatorics \
  --seeds 1,2,3

./run.sh run \
  --arm hint \
  --model vllm/qed-nano \
  --domain combinatorics \
  --seeds 1,2,3
```

Audit with the configured independent judge; the vLLM tunnel is not used by
the audit stage:

```bash
./run.sh audit \
  --arm baseline \
  --model vllm/qed-nano \
  --domain combinatorics \
  --seeds 1,2,3
```

Results are written under `results/vllm-qed-nano/`.

## Context and compaction

The server declares a 262,144-token context, so QED-Nano will compact a long
sequential trajectory more frequently than a larger-context model. This does
not uniquely invalidate its 8x arm: the 1.6M-output-token sequential protocol
already exceeds even Opus's context and therefore already relies on the same
Claude CLI compaction mechanism. Hold the CLI version and compaction settings
fixed, disclose each model's context limit, and compare compute scaling
primarily within each fixed model-agent system; cross-model boundary shifts
remain supporting observational evidence. Parallel 8x uses independent 1x
contexts and is unaffected by cross-trajectory compaction.
