# Codex subscription pool through LiteLLM

Run only `scripts/codex_pool.sh`; the Python files in
`scripts/codex_pool_internal/` are private helpers called by that script.

This setup runs one isolated LiteLLM sidecar per authorized Codex
subscription on a single machine. The experiment harness addresses GPT-5.4,
GPT-5.4 mini, GPT-5.5, GPT-5.6 Luna, GPT-5.6 Terra, and GPT-5.6 Sol as
`litellm/gpt-5.4`, `litellm/gpt-5.4-mini`, `litellm/gpt-5.5`,
`litellm/gpt-5.6-luna`, `litellm/gpt-5.6-terra`, and `litellm/gpt-5.6-sol`, assigns
concurrent sessions across healthy sidecars, and
uses its existing cooldown/recovery logic when a sidecar reports a limit.

Each sidecar has its own OAuth volume. Raw credentials are never baked into an
image, written to the repository, or printed by the internal converter.

## 1. Stage the auth files privately

Use one numbered directory per subscription outside the repository:

```text
~/.codex-subscriptions/
├── slot-1/auth.json
├── slot-2/auth.json
└── slot-3/auth.json
```

The default root is `~/.codex-subscriptions`; override it with
`CODEX_LITELLM_AUTH_ROOT` when necessary. Directories should be mode `0700`
and each `auth.json` mode `0600`.

For the account currently logged into Codex, its source is normally
`~/.codex/auth.json`. After signing into each authorized account, save that
account's complete file under the corresponding numbered path. Do not combine
fields from different accounts.

The source files are needed only when adding an account. After successful verification,
retain or delete them according to your credential-backup policy; the runtime
copies live in Docker volumes.

## 2. Add each subscription

For three subscriptions stored in the default layout:

```bash
./scripts/codex_pool.sh add-all 3
```

To add a single slot from an explicit path:

```bash
./scripts/codex_pool.sh add 1 ~/.codex/auth.json
./scripts/codex_pool.sh add 2 /private/path/account-2-auth.json
```

The helper checks that every source path in an `add-all` batch exists
before changing any slot. It validates each Codex credential while copying,
converts the nested schema into LiteLLM's flat schema, writes it atomically
with mode `0600`, and stores it in these named volumes:

```text
olympiad-codex-litellm-auth-1
olympiad-codex-litellm-auth-2
olympiad-codex-litellm-auth-3
```

Re-adding a slot stops only that slot's sidecar, then replaces its OAuth
copy. It never touches another slot. The helper also stores a one-way
identity digest; startup fails before launching the pool if two slots contain
the same subscription.

## 3. Start and verify the pool

Launch three sidecars, then verify both a direct Responses request and a Claude
Agent SDK `Write` call on every subscription:

```bash
./scripts/codex_pool.sh start 3
./scripts/codex_pool.sh verify 3
```

The host-only diagnostic endpoints are `127.0.0.1:4101`, `:4102`, and
`:4103`. Sidecars also join the dedicated Docker network
`olympiad-codex-litellm`; experiment containers can reach only their port 4000
addresses through the egress firewall.

Check pool state or one sidecar's logs with:

```bash
./scripts/codex_pool.sh status 3
./scripts/codex_pool.sh logs 2
```

A successful per-slot verification prints:

```text
PASS direct Responses call: model='gpt-5.4'
PASS Claude Agent SDK call: model='gpt-5.4', tools=['Write']
```

## 4. Configure the experiment harness

Print the exact pool settings:

```bash
./scripts/codex_pool.sh env 3
```

Copy its output into the repository's git-ignored `.env`. It has this shape:

```dotenv
LITELLM_API_KEY=sk-codex-local-only
LITELLM_BASE_URL=http://olympiad-codex-litellm-1:4000
LITELLM_BASE_URL_2=http://olympiad-codex-litellm-2:4000
LITELLM_BASE_URL_3=http://olympiad-codex-litellm-3:4000
```

The proxy key authenticates only this local pool; it is not an OpenAI secret.
The numbered base URLs are the harness's provider pool.

`total_cost_usd` in harness output is LiteLLM/Agent SDK's API-equivalent
estimate. It is not a separate charge for ChatGPT-authenticated subscription
traffic. The experiment's compute metric is provider-reported output tokens;
plan consumption follows ChatGPT credits and usage limits. Check the official
[Codex usage dashboard](https://chatgpt.com/codex/settings/usage) for the
account-level balance.

Run an ordinary arm with the `litellm/` model prefix:

```bash
./run.sh run \
  --arm baseline \
  --model litellm/gpt-5.4 \
  --audit-model claude-opus-4-8 \
  --domain combinatorics \
  --seeds 1
```

For auditing GPT-generated proofs, select a different judge as usual:

```bash
./run.sh audit \
  --arm baseline \
  --model litellm/gpt-5.4 \
  --audit-model claude-opus-4-8 \
  --domain combinatorics \
  --seeds 1
```

The `litellm/` prefix is harness routing metadata. LiteLLM receives the actual
model alias `gpt-5.4`, while result paths use `litellm-gpt-5.4`.

GPT-5.4 mini, GPT-5.5, GPT-5.6 Luna, GPT-5.6 Terra, and GPT-5.6 Sol use the same frozen Responses
translation, tool policy, and output budgets. GPT-5.4 mini compacts at 300k to
fit its 400k context; the million-context models compact at 900k:

```bash
./run.sh run \
  --arm baseline \
  --model litellm/gpt-5.5 \
  --domain combinatorics \
  --seeds 1
```

To smoke-test a non-default model through each subscription before a run,
override only the verification model, for example:

```bash
CODEX_LITELLM_MODEL=gpt-5.5 ./scripts/codex_pool.sh verify 3
```

## Routing and recovery

- `LITELLM_BASE_URL`, `_2`, `_3`, and so on are deduplicated and assigned
  round-robin, using the same pool implementation as provider credentials.
- Several concurrent attempts may use one subscription; sidecar count is not a
  concurrency cap.
- A live session normally remains on its assigned sidecar. If the provider
  emits a rate-limit or spend-limit event, the existing recovery mechanism
  cools that sidecar, resumes the same local Claude transcript through another
  healthy sidecar, and preserves accumulated output-token accounting.
- Every transition is recorded using non-secret `credential_N` labels. OAuth
  material and full sidecar URLs are not written into result metadata.
- If every sidecar is cooling, attempts wait for the earliest reset instead of
  repeatedly retrying a limited account.
- GPT can spend a long time in hidden reasoning before emitting its first
  Anthropic-compatible stream event. For `litellm/` models the harness enables
  the third-party stream watchdog and sets both its idle threshold and the
  separate API-request timeout to one hour.
- Sidecars set `DISABLE_AIOHTTP_TRANSPORT=true`, selecting LiteLLM's HTTPX
  upstream transport. The pinned aiohttp transport produced truncated chunked
  streams under long concurrent GPT runs; this choice is recorded in result
  metadata and the checkpoint identity.
- The harness disables Claude Code's non-streaming fallback and automatic API
  retries, and the sidecar disables LiteLLM router retries. Transient failures
  are handled only by the harness's bounded same-transcript recovery, which
  deduplicates stable message IDs and preserves accumulated eligible-output
  accounting.
- The experimental budget counts output delivered into the persisted
  transcript: streamed usage plus completed per-query Result usage. Backend
  work that returned no usable transcript output is infrastructure overhead,
  not experimental output. Recovered attempts are valid and marked
  `recovered_eligible_output_accounted`; `process_resume_count` and
  `session_reconnects` retain the exact operational provenance.
- The non-secret LiteLLM transport policy is recorded in every result and in
  the GPT checkpoint identity. Changing it starts a new GPT checkpoint lineage
  instead of silently mixing sessions collected under different timeout or
  retry rules; other providers' checkpoint identities are unchanged.

## Stop or revoke

Stop the containers while retaining the copied OAuth:

```bash
./scripts/codex_pool.sh stop 3
```

Permanently delete one copied OAuth credential:

```bash
./scripts/codex_pool.sh delete-auth 2 --yes
```

That deletion is irreversible for the Docker copy; the original account login
or staged source file is required to add it again.

## Compatibility patch

Stock LiteLLM 1.97.0 can call `chatgpt/gpt-5.4` through Responses, but its
Anthropic translation assumes string-valued system content. Claude Agent SDK
sends content-block lists, and the ChatGPT subscription backend rejects system
messages.

`docker/Dockerfile.litellm-codex` pins the tested image and applies a narrow,
fail-closed patch from `docker/patch_litellm_chatgpt.py`. The patch safely folds
list-valued system content into the first user turn. The build fails if the
expected upstream source changes, forcing review before an upgrade.

This preserves the complete Agent SDK prompt and tool use, but not system-role
priority. Freeze the compatibility image for canonical paper runs and validate
long sessions, resume behavior, exact token accounting, and every enabled tool
before collecting final data.
