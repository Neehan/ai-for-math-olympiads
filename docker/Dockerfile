# Throwaway run environment for one experiment arm.
# Isolation model: the container holds only the harness, the problems, and the
# prompts (no reference solutions, no other repos); the entrypoint firewall
# limits egress to the Anthropic API alone.
FROM python:3.12-slim

# The pinned claude-agent-sdk wheel ships its own standalone CLI binary.
# iptables for the firewall; gosu to drop root — the CLI refuses
# bypassPermissions as root, and a non-root agent cannot touch iptables.
RUN apt-get update \
    && apt-get install -y --no-install-recommends iptables gosu ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# prompts/, config.json, agent_settings.json, and results/ are bind-mounted
# at run time (see run.sh); problems/hints are fetched by the entrypoint
# before the firewall closes and never stored in the image.
ENTRYPOINT ["/entrypoint.sh"]
