# Meta Muse Spark

The harness supports Meta's contributor-tier model under its provider model ID:

```text
muse-spark-1.2-contributor
```

Set the credential in the repository's `.env`:

```bash
META_API_KEY=
```

The harness passes that value as `ANTHROPIC_AUTH_TOKEN`, sets
`ANTHROPIC_BASE_URL=https://api.meta.ai`, and the Anthropic client calls Meta's
`https://api.meta.ai/v1` API. The run container permits egress only to
`api.meta.ai` for this provider.

Meta's adapter accepts standard Anthropic Messages fields but rejects Claude's
`output_config.task_budget` extension. The harness therefore omits that optional
field for Muse; the task prompt still states the allocation and the local token
tracker enforces the same cutoff as every other provider.

For example:

```bash
./run.sh run --arm baseline \
  --model muse-spark-1.2-contributor \
  --domain combinatorics \
  --seeds 1,2,3
```
