#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o /orcd/pool/008/notadib/vllm/logs/slurm/glm_%j.log
#SBATCH -e /orcd/pool/008/notadib/vllm/logs/slurm/glm_%j.log
#SBATCH --job-name=glm

# Serves zai-org/GLM-4.7-Flash through vLLM's native Anthropic /v1/messages endpoint.
# Submit:  sbatch /orcd/pool/008/notadib/vllm/serve_glm.sbatch
# Then read the NODE line out of the log and open the tunnel from your Mac.
#
# GLM-4.7-Flash is a 30B-A3B MoE (31B params, bf16 on disk = 62.5 GB), 47 layers,
# 64 routed experts, top-4 routing, MLA attention, 202752 native context.
#
# REPRODUCIBILITY: this runs STOCK vllm 0.26.0 with ZERO patches. Verified on
# 2026-08-09 that every .py under site-packages/vllm matches the wheel's own
# sha256 in vllm-0.26.0.dist-info/RECORD. Do not edit site-packages; if a change
# is ever truly required, ship it as a file under patches/ applied here.
#
# Every flag below was arrived at by hitting the failure it prevents -- see the
# L40S NOTES block after the exec.

set -euo pipefail

# ---- toolchain -------------------------------------------------------------
# vllm 0.26.0 is installed in ~/.local against the module's python3.12.
# Do NOT `conda activate vllm`. The directory /home/notadib/miniconda3/envs/vllm
# exists but is hollow (only conda-meta/ and etc/, no bin/python), so the
# activate SUCCEEDS and then every python command fails confusingly.
module load miniforge

# miniforge's libstdc++ provides GLIBCXX_3.4.29. Without it numpy fails with a
# misleading "do not import numpy from its source directory" error -- if you see
# that, LD_LIBRARY_PATH is wrong, the problem has nothing to do with numpy.
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/orcd/software/core/001/pkg/miniforge/25.11.0-0/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME=/orcd/pool/008/notadib/hf_cache
# Weights are already in the cache; do not let a hub lookup stall the job.
export HF_HUB_OFFLINE=1
# Fixed key shared by the short-lived ORCD server and the local harness. This
# is intentionally not treated as a secret: access still requires the user's
# SSH tunnel, and each Slurm job expires after at most six hours.
export VLLM_API_KEY=glm-local-key

# ---- tunnel instructions ---------------------------------------------------
echo "=============================================================="
echo "NODE:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID:-interactive}"
echo ""
echo "From your Mac:"
echo "  ssh -N -o ExitOnForwardFailure=yes -L 0.0.0.0:8000:$(hostname):8000 mitengage"
echo ""
echo "Harness env:"
echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:8000      (or host.docker.internal)"
echo "  ANTHROPIC_AUTH_TOKEN=$VLLM_API_KEY            (NOT ANTHROPIC_API_KEY)"
echo "  CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536           (SDK has no max_tokens option)"
echo ""
echo "SAMPLING (z.ai 'Default / Most Tasks', what their AIME numbers use):"
echo "  temperature=1.0  top_p=0.95"
echo "  generation_config.json pins ONLY temperature -- set top_p yourself."
echo "=============================================================="

# ---- serve -----------------------------------------------------------------
exec /home/notadib/.local/bin/vllm serve zai-org/GLM-4.7-Flash \
  --served-model-name glm-4.7-flash claude-sonnet-4-5 claude-3-5-haiku-20241022 claude-opus-4-6 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 202752 \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --kernel-config '{"moe_backend":"marlin"}' \
  --max-num-seqs 8

# ---------------------------------------------------------------------------
# L40S NOTES -- three failures, all the same 101,376-byte shared-memory limit
# on SM 8.9. Each flag below exists because of one of them. Measured 2026-08-09.
#
# 1. moe_backend=marlin  (THE important one)
#    vLLM's auto-selection picks the TRITON fp8 MoE backend on this GPU. That
#    kernel wants 102,400 bytes of shared memory vs the hardware's 101,376 and
#    crashes during CUDA-graph profiling at engine init:
#      triton.runtime.errors.OutOfResources: Required: 102400, Hardware limit: 101376
#    Working around it with --enforce-eager "worked" but cost 19x throughput:
#      triton + enforce-eager :   6 tok/s
#      marlin + cuda graphs   : 117 tok/s
#    Marlin is the Ada-optimised path and needs no eager fallback. This is the
#    ONLY non-default choice in the whole command; it is a hardware-driven
#    backend selection, not a modification to vLLM.
#
# 2. NO --kv-cache-dtype fp8
#    GLM-4.7 uses MLA, and TRITON_MLA is the only attention backend vLLM 0.26.0
#    offers for it here ("out of potential backends: ['TRITON_MLA']"). Its
#    fp8-KV variant overruns the same shared-memory limit. THIS FAILS LATE:
#    engine init completes, "Application startup complete" is logged, the server
#    looks healthy -- and then the first real request 500s and kills the engine.
#    bf16 KV halves the pool but that is still ample (see below), and it also
#    removes the fp8-KV accuracy caveat, which matters for proof-quality work.
#
# 3. --enforce-eager is NOT needed once moe_backend=marlin is set.
#    Do not re-add it "for safety" -- it is what caused the 6 tok/s.
#
# MEASURED with the exact config above (2x L40S, TP=2):
#   weights per GPU     14.6 GiB (fp8, loaded in 40 s from POOL)
#   GPU KV cache        488,496 tokens  (bf16 KV)
#   concurrency         ~2.4x at full 202,752-token context
#   throughput          88-120 tok/s single stream
#
# WHY fp8 WEIGHTS: GLM-4.7-Flash has NO grouped-query attention --
# num_key_value_heads == num_attention_heads == 20 -- so KV costs ~3x per token
# what a GQA model of the same depth would. bf16 weights (~31 GiB/GPU) would
# leave too little room for the full context; fp8 leaves ~27 GiB/GPU for cache.
#
# WHY ONLY 2 GPUs / 32 CPUs: the mit_normal_gpu QOS caps a single user at
# gres/gpu=2, cpu=32, mem=515G (`sacctmgr show qos mit_normal_gpu format=MaxTRESPU`).
#
# TOOL CALLING works end to end through the Claude Agent SDK: verified a full
# round-trip (thinking block -> ToolUseBlock -> tool result -> final text,
# is_error=False). Unlike nomos-1 and QED-Nano, this model actually emits tool
# calls. Both parsers are named `glm47`; they are different registries
# (--reasoning-parser vs --tool-call-parser), so both flags are required.
#
# PRESERVED THINKING is UNVERIFIED. z.ai recommends
#   "thinking": {"type": "enabled", "clear_thinking": false}
# for multi-turn agentic work, but vLLM's /v1/messages silently DROPS `thinking`
# fields it does not model, so this probably never reaches GLM through the
# Anthropic endpoint. Check before relying on cross-turn reasoning continuity.
#
# Token accounting for the paper:
#   Per-request:  usage.input_tokens / usage.output_tokens on every response.
#   Per-job total: curl -s -H "Authorization: Bearer $VLLM_API_KEY" \
#                    http://127.0.0.1:8000/metrics | grep -E 'vllm:(prompt|generation)_tokens_total'
#                  Counters are per server process. NOTE they are cumulative
#                  since server start, so snapshot before AND after a run --
#                  they are not per-request.
#
# NOTE: prefix caching is ON by default, so input_tokens is the LOGICAL prompt
# length, not tokens actually prefilled. For honest compute numbers report
# vllm:prompt_tokens_by_source_total{source="local_compute"} or add
# --no-enable-prefix-caching for measurement runs.