# Throwaway run environment for one experiment arm.
# Isolation model: the container holds only the harness, the problems, and the
# prompts (no reference solutions, no other repos); the entrypoint firewall
# limits egress to the Anthropic API alone.
FROM python:3.12-slim

# nodejs/npm for the Claude Code CLI the SDK drives; iptables for the firewall.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm iptables ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code@2.1.209

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# problems.jsonl, prompts/, config.json, agent_settings.json, and results/ are
# bind-mounted at run time (see run.sh), so editing them needs no rebuild.
ENTRYPOINT ["/entrypoint.sh"]
