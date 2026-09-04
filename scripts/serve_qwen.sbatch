#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -o /orcd/pool/008/notadib/vllm/logs/slurm/qwen_%j.log
#SBATCH -e /orcd/pool/008/notadib/vllm/logs/slurm/qwen_%j.log
#SBATCH -t 6:00:00
#SBATCH --job-name=qwen

# Serves Qwen/Qwen3-8B through vLLM's native Anthropic /v1/messages endpoint.
# Submit:  sbatch /orcd/pool/008/notadib/vllm/serve_qwen.sbatch
# Then read the NODE line out of the log and open the tunnel from your Mac.
#
# Qwen3-8B is a dense 8.2B model: 36 layers, 32 query heads over 8 KV heads,
# head_dim 128, bf16. Its KV geometry is identical to QED-Nano's, so the L40S
# budget below is the same shape as scripts/serve_qed.sbatch.

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
export VLLM_API_KEY=qwen-local-key

# ---- tunnel instructions ---------------------------------------------------
echo "=============================================================="
echo "NODE:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID:-interactive}"
echo ""
echo "From the harness machine (use whichever name your ssh config gives the"
echo "ORCD login node -- orcd-login here, mitengage on the Mac):"
echo "  ssh -N -o ExitOnForwardFailure=yes -L 0.0.0.0:8000:$(hostname):8000 orcd-login"
echo ""
echo "Add to the harness .env:"
echo "  VLLM_BASE_URL=http://host.docker.internal:8000"
echo "  VLLM_API_KEY=$VLLM_API_KEY"
echo "Run with: --model vllm/qwen3-8b"
echo ""
echo "SAMPLING (Qwen's thinking-mode recommendation):"
echo "  temperature=0.6  top_p=0.95  top_k=20  (never greedy in thinking mode)"
echo "=============================================================="

# ---- serve -----------------------------------------------------------------
# CONTEXT: Qwen3-8B trains to max_position_embeddings=40960 and ships with
# rope_scaling null. 40K is far too small for this harness -- one 200k-token
# attempt compacts many times over -- so YaRN extends it 4x to 131072, which is
# the extension Qwen documents for the dense Qwen3 models. YaRN is a STATIC
# scaling in vLLM: it is applied to every request, including short ones, and
# Qwen notes it can cost a little short-context accuracy. That trade is
# accepted here because a 40K ceiling would truncate almost every trajectory.
# To measure short-context behaviour instead, drop --rope-scaling, set
# --max-model-len 40960, and lower QWEN3_8B_AUTO_COMPACT_WINDOW to ~30k.
#
# GPU budget on L40S (48 GB):
#   0.92 util  -> ~41 GiB usable
#   weights    -> ~15.3 GiB (8.2B params, bf16)
#   overhead   -> ~3 GiB (activations + CUDA graphs)
#   KV cache   -> ~22 GiB
# fp8 KV costs 2*36 layers*8 kv-heads*128 head-dim = 72 KiB/token, so that is
# ~310K tokens of cache PER GPU; a single full 131K-token request needs ~9 GiB,
# leaving room for concurrency alongside it.
#
# 2 GPUs are used as 2 independent replicas (data parallel), NOT tensor parallel:
# the 8B model fits on one L40S with room to spare, so splitting it would only add
# all-reduce traffic -- and L40S has no NVLink, so that traffic would cross PCIe.
# DP gives ~2x throughput behind a single :8000 endpoint, which is what you want
# for running many experiments. Use --tensor-parallel-size 2 instead only if you
# ever need lower latency on a single request.
#
# WHY ONLY 2 GPUs / 32 CPUs: the mit_normal_gpu QOS caps a single user at
# gres/gpu=2, cpu=32, mem=515G (`sacctmgr show qos mit_normal_gpu format=MaxTRESPU`).
#
# The Claude Agent SDK sends whatever model id the harness passes plus, for some
# internal calls, a small fixed Claude id; --served-model-name aliases all of
# them onto this one server exactly as serve_qed.sbatch does.
exec /home/notadib/.local/bin/vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b claude-sonnet-4-5 claude-3-5-haiku-20241022 claude-opus-4-6 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}' \
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
# CONCURRENCY: --max-num-seqs is PER REPLICA, so 32 x DP2 = 64 requests in flight
# before anything queues. If you run long-context experiments and see throughput
# drop, check
#   curl -s http://127.0.0.1:8000/metrics | grep -E 'vllm:num_preemptions_total'
# and walk --max-num-seqs down.
#
# NOTE: prefix caching is ON by default, so input_tokens is the LOGICAL prompt length,
# not tokens actually prefilled. For honest compute numbers either report
#   vllm:prompt_tokens_by_source_total{source="local_compute"}
# or add --no-enable-prefix-caching below for measurement runs.
