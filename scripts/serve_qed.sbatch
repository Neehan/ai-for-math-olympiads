#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o /orcd/pool/008/notadib/vllm/logs/slurm/vllm_%j.log
#SBATCH -e /orcd/pool/008/notadib/vllm/logs/slurm/vllm_%j.log
#SBATCH --job-name=vllm

# Serves lm-provers/QED-Nano through vLLM's native Anthropic /v1/messages endpoint.
# Submit:  sbatch /orcd/pool/008/notadib/vllm/serve_qed.sbatch
# Then read the NODE line out of the log and open the tunnel from your Mac.

set -euo pipefail

# ---- toolchain -------------------------------------------------------------
# vllm 0.26.0 is installed in ~/.local against the module's python3.12.
# Do NOT `conda activate vllm` -- that env no longer exists.
module load miniforge

# miniforge's libstdc++ provides GLIBCXX_3.4.29; without it numpy fails to import.
# CUDA 13.0 matches torch's build and supplies nvcc for FlashInfer's JIT (needed by fp8 KV).
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/orcd/software/core/001/pkg/miniforge/25.11.0-0/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME=/orcd/pool/008/notadib/hf_cache
# Fixed key shared by the short-lived ORCD server and the local harness. This
# is intentionally not treated as a secret: access still requires the user's
# SSH tunnel, and each Slurm job expires after at most six hours.
export VLLM_API_KEY=qed-local-key

# ---- tunnel instructions ---------------------------------------------------
echo "=============================================================="
echo "NODE:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID:-interactive}"
echo ""
echo "From your Mac:"
echo "  ssh -N -o ExitOnForwardFailure=yes -L 0.0.0.0:8000:$(hostname):8000 mitengage"
echo ""
echo "Add to the harness .env:"
echo "  VLLM_BASE_URL=http://host.docker.internal:8000"
echo "  VLLM_API_KEY=$VLLM_API_KEY"
echo "Run with: --model vllm/qed-nano"
echo "=============================================================="

# ---- serve -----------------------------------------------------------------
# Native max_position_embeddings is 262144; rope_scaling is null, so 256K is the
# real ceiling.
#
# GPU budget on L40S (48 GB, vs 80 GB on the H100 this used to request):
#   0.92 util  -> ~41 GiB usable
#   weights    -> ~7.5 GiB (4B params, bf16)
#   overhead   -> ~3 GiB (activations + CUDA graphs)
#   KV cache   -> ~30 GiB
# fp8 KV costs 2*36 layers*8 kv-heads*128 head-dim = 72 KiB/token, so that is
# ~425K tokens of cache PER GPU. A single full-length 256K request needs 18 GiB,
# so max-model-len 262144 still fits with room for concurrency alongside it.
#
# 2 GPUs are used as 2 independent replicas (data parallel), NOT tensor parallel:
# the 4B model fits on one L40S with room to spare, so splitting it would only add
# all-reduce traffic -- and L40S has no NVLink, so that traffic would cross PCIe.
# DP gives ~2x throughput behind a single :8000 endpoint, which is what you want
# for running many experiments. Use --tensor-parallel-size 2 instead only if you
# ever need lower latency on a single request.
#
# WHY ONLY 2 GPUs / 32 CPUs: the mit_normal_gpu QOS caps a single user at
# gres/gpu=2, cpu=32, mem=515G (`sacctmgr show qos mit_normal_gpu format=MaxTRESPU`).
# This job sits exactly on the GPU and CPU caps; asking for more will never start.
# 4 GPUs is only reachable in mit_preemptable, which allows gres/gpu=4 but can
# preempt the server mid-run.
exec /home/notadib/.local/bin/vllm serve lm-provers/QED-Nano \
  --served-model-name qed-nano claude-sonnet-4-5 claude-3-5-haiku-20241022 claude-opus-4-6 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --data-parallel-size 2 \
  --max-num-seqs 32

# Token accounting for the paper:
#   Per-request:  usage.input_tokens / usage.output_tokens on every response
#                 (streaming: final `message_delta` event carries the same numbers).
#   Per-job total: curl -s -H "Authorization: Bearer $VLLM_API_KEY" \
#                    http://127.0.0.1:8000/metrics | grep -E 'vllm:(prompt|generation)_tokens_total'
#                  Counters are per server process, so they reset each job = clean per-run totals.
#
# CONCURRENCY: --max-num-seqs is PER REPLICA, so 64 x DP2 = 128 requests in flight
# before anything queues. That divides the ~425K-token KV pool into ~6.6K tokens per
# slot; requests longer than that make vLLM preempt and recompute rather than fail.
# If you run long-context experiments and see throughput drop, check
#   curl -s http://127.0.0.1:8000/metrics | grep -E 'vllm:num_preemptions_total'
# and walk --max-num-seqs back toward 32.
#
# NOTE: prefix caching is ON by default, so input_tokens is the LOGICAL prompt length,
# not tokens actually prefilled. For honest compute numbers either report
#   vllm:prompt_tokens_by_source_total{source="local_compute"}
# or add --no-enable-prefix-caching below for measurement runs.

