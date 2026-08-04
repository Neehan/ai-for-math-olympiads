#!/bin/sh
# Egress firewall + harness launch. Requires --cap-add=NET_ADMIN.
#
# Policy: the agent may talk to the LLM API and NOTHING else. We allow
# loopback (which covers Docker's embedded DNS at 127.0.0.11), replies on
# established connections, and TLS (443) only to the Anthropic endpoints
# resolved right now; every other outbound packet is dropped. The firewall is
# self-tested before any token is spent: the run aborts unless a non-Anthropic
# host is unreachable AND the Anthropic API is reachable.
set -eu

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

ANTHROPIC_HOSTS="api.anthropic.com claude.ai console.anthropic.com"

iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT
for host in $ANTHROPIC_HOSTS; do
    for ip in $(getent ahostsv4 "$host" | awk '{print $1}' | sort -u); do
        iptables -A OUTPUT -d "$ip/32" -p tcp --dport 443 -j ACCEPT
    done
done
iptables -P OUTPUT DROP

python - <<'PY'
import socket
import sys
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
try:
    urllib.request.urlopen("https://api.anthropic.com/")
except urllib.error.HTTPError:
    pass  # any HTTP response means the API is reachable
except OSError as error:
    sys.exit(f"FIREWALL SELF-TEST FAILED: Anthropic API unreachable: {error}")
print("firewall ok: egress restricted to Anthropic API")
PY

exec python -m src.run "$@"
