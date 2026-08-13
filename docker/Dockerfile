# Throwaway run environment for one experiment arm.
# Isolation model: the container holds only the harness, the problems, and the
# prompts (no reference solutions, no other repos); the entrypoint firewall
# limits egress to the configured LLM endpoints alone.
FROM python:3.12-slim

# The SDK-bundled CLI (2.1.92) ignores CLAUDE_CODE_MAX_OUTPUT_TOKENS on Opus
# (32k clamp bug), so pin a fixed CLI via npm and point the SDK at it.
# iptables for the firewall; gosu to drop root — the CLI refuses
# bypassPermissions as root, and a non-root agent cannot touch iptables.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm iptables gosu ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code@2.1.222 \
    && useradd -m appuser
ENV HARNESS_CLI_PATH=/usr/local/bin/claude

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# prompts/, config.json, both agent settings profiles, and results/ are bind-mounted
# at run time (see run.sh); problems/hints are fetched by the entrypoint
# before the firewall closes and never stored in the image.
ENTRYPOINT ["/entrypoint.sh"]
