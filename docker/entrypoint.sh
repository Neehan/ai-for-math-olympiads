#!/bin/sh
# Egress firewall + harness launch. Requires --cap-add=NET_ADMIN.
#
# Internal usage: entrypoint.sh <run|audit|state-audit> [args...] — every solver and
# audit stage runs behind the SAME firewall: a judge with tools and internet
# could fetch official solutions from public archives, and its archived
# scratch would contaminate future runs.
#
# Policy: the agent may talk to the LLM API and NOTHING else. We allow
# loopback, DNS to the container's configured resolvers (on the default
# bridge these are NOT loopback — blocking them breaks all resolution), and
# TLS (443) only to the external provider endpoints plus explicitly configured
# local LiteLLM/vLLM endpoints; every other outbound packet is dropped. The
# firewall is self-tested before any token is spent.
set -eu

STAGE="${1:?usage: entrypoint.sh <run|audit|state-audit> [args...]}"
shift
case "$STAGE" in
    run|audit) MODULE="$STAGE" ;;
    state-audit) MODULE="state_audit" ;;
    *) echo "unknown stage '$STAGE' (expected run, audit, or state-audit)" >&2; exit 2 ;;
esac

# Prefetch the problem/hint datasets BEFORE the firewall closes: HuggingFace
# must stay unreachable while agents run (an agent that could fetch the hints
# file would be contaminated). The harness loader reads these files once and
# deletes them before any agent spawns, so no trace remains.
mkdir -p /run/contest
python - <<'PY'
import urllib.request
from src.constants import HINTS_URL, OUTLINES_URL, PROBLEMS_URL

for url, name in [
    (PROBLEMS_URL, "problems.jsonl"),
    (HINTS_URL, "hints.jsonl"),
    (OUTLINES_URL, "outlines.jsonl"),
]:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with open(f"/run/contest/{name}", "wb") as handle:
        handle.write(data)
print("datasets prefetched")
PY
export PROBLEMS_FILE=/run/contest/problems.jsonl
export HINTS_FILE=/run/contest/hints.jsonl
export OUTLINES_FILE=/run/contest/outlines.jsonl
if [ "$STAGE" = "state-audit" ]; then
    python - <<'PY'
import urllib.request
from src.constants import SOLUTIONS_URL

with urllib.request.urlopen(SOLUTIONS_URL, timeout=60) as response:
    data = response.read()
with open("/run/contest/solutions.jsonl", "wb") as handle:
    handle.write(data)
print("reference solutions prefetched")
PY
    export SOLUTIONS_FILE=/run/contest/solutions.jsonl
fi

PROVIDER_KIND="${HARNESS_PROVIDER_KIND:?HARNESS_PROVIDER_KIND is required}"
case "$PROVIDER_KIND" in
    anthropic) API_HOSTS="api.anthropic.com claude.ai console.anthropic.com" ;;
    openrouter) API_HOSTS="openrouter.ai" ;;
    litellm) API_HOSTS="" ;;
    vllm) API_HOSTS="" ;;
    *) echo "unknown HARNESS_PROVIDER_KIND '$PROVIDER_KIND'" >&2; exit 2 ;;
esac

iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT
# DNS: the container's resolvers (from resolv.conf) must stay reachable —
# they are external IPs on the default bridge, not 127.0.0.11.
for ns in $(awk '/^nameserver/ {print $2}' /etc/resolv.conf); do
    iptables -A OUTPUT -d "$ns" -p udp --dport 53 -j ACCEPT
    iptables -A OUTPUT -d "$ns" -p tcp --dport 53 -j ACCEPT
done
for host in $API_HOSTS; do
    for ip in $(getent ahostsv4 "$host" | awk '{print $1}' | sort -u); do
        iptables -A OUTPUT -d "$ip/32" -p tcp --dport 443 -j ACCEPT
    done
done

# Local LiteLLM sidecars and tunneled vLLM endpoints are permitted only when
# explicitly supplied by the controller. Resolve them before locking
# DNS/egress, then allow only their exact IP and port.
python - <<'PY' >/tmp/local-provider-targets
import os
import re
import socket
from urllib.parse import urlparse

pattern = re.compile(r"^(?:LITELLM|VLLM)_BASE_URL(?:_\d+)?$")
targets = set()
for name, value in os.environ.items():
    if not pattern.fullmatch(name) or not value.strip():
        continue
    parsed = urlparse(value)
    if parsed.scheme != "http" or not parsed.hostname:
        raise SystemExit(f"invalid {name}: expected an http:// sidecar URL")
    port = parsed.port or 80
    for _, _, _, _, address in socket.getaddrinfo(
        parsed.hostname, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        targets.add((address[0], port))
for ip, port in sorted(targets):
    print(ip, port)
PY
while read -r ip port; do
    [ -n "$ip" ] || continue
    iptables -A OUTPUT -d "$ip/32" -p tcp --dport "$port" -j ACCEPT
done </tmp/local-provider-targets
iptables -P OUTPUT DROP

python - <<'PY'
import socket
import sys
import os
import re
import urllib.error
import urllib.request

socket.setdefaulttimeout(8)
try:
    urllib.request.urlopen("https://example.com")
    sys.exit("FIREWALL SELF-TEST FAILED: example.com is reachable")
except urllib.error.HTTPError:
    sys.exit("FIREWALL SELF-TEST FAILED: example.com is reachable")
except OSError:
    pass
provider_kind = os.environ["HARNESS_PROVIDER_KIND"]
external_probe = {
    "anthropic": ("Anthropic", "https://api.anthropic.com/"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/"),
}.get(provider_kind)
if external_probe is not None:
    provider_name, provider_url = external_probe
    try:
        urllib.request.urlopen(provider_url)
    except urllib.error.HTTPError:
        pass  # any HTTP response means the selected provider is reachable
    except OSError as error:
        sys.exit(f"FIREWALL SELF-TEST FAILED: {provider_name} unreachable: {error}")
for name, value in sorted(os.environ.items()):
    match = re.fullmatch(r"(LITELLM|VLLM)_BASE_URL(?:_\d+)?", name)
    if match is None or not value.strip():
        continue
    provider_prefix = match.group(1)
    health_path = "/health/liveliness" if provider_prefix == "LITELLM" else "/health"
    api_key = os.environ.get(f"{provider_prefix}_API_KEY", "")
    request = urllib.request.Request(
        value.rstrip("/") + health_path,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        urllib.request.urlopen(request)
    except OSError as error:
        sys.exit(f"FIREWALL SELF-TEST FAILED: {name} unreachable: {error}")
print("firewall ok: egress restricted to the allowed LLM APIs")
PY

# Drop root for the harness and every agent it spawns: the CLI refuses
# bypassPermissions as root, and a non-root agent cannot alter the firewall.
chown appuser /app
chown -R appuser /run/contest
if [ -d /c ]; then
    chown -R appuser /c
fi
if [ -d /app/state-results ]; then
    chown -R appuser /app/state-results
fi
export HOME=/home/appuser
if [ "$PROVIDER_KIND" = "openrouter" ]; then
    export HARNESS_OPENROUTER_PROXY_URL=http://127.0.0.1:8787/api
    gosu appuser python -m src.openrouter_proxy &
    proxy_pid=$!
    python - <<'PY'
import time
import urllib.request

for _ in range(100):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8787/health", timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(0.05)
else:
    raise SystemExit("OpenRouter routing shim failed to start")
PY
    kill -0 "$proxy_pid"
fi
exec gosu appuser python -m "src.$MODULE" "$@"
