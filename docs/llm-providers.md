# LLM Providers

Generative LLMs propose structured strategies, explain results, and suggest reviewed improvements or semantic questions. Advisor responses are validated as `RecommendationSpec`, translated to `ExperimentSpec`, and checked against the same registries used by Expert mode. Generated code is never executed. Typed row judgments use the separate [Decision Provider](decision-providers.md) contract.

## Configure a provider

Set `LLM_PROVIDER`, `LLM_MODEL`, and optionally `LLM_BASE_URL` in `.env`.

### OpenRouter

```dotenv
LLM_PROVIDER=openrouter
LLM_MODEL=<structured-output-capable-model>
LLM_API_KEY=<secret>
REASONING_MODEL=false
OPENROUTER_PROVIDER=anthropic,openai
```

Start with `make up`. The key is sent only as an authorization header and must never be committed.

Set `REASONING_MODEL=false` for a conventional model. The adapter sends a deterministic
`temperature=0` and ignores `LLM_REASONING` and `LLM_VERBOSITY`, even when they are set. For a
reasoning-capable model, set `REASONING_MODEL=true`; the adapter then omits `temperature` and sends
both `LLM_REASONING` and `LLM_VERBOSITY`.

`OPENROUTER_PROVIDER` is an optional, comma-separated ordered allowlist of OpenRouter provider
slugs. When present, the request uses this order and disables fallback to providers outside the
list. Leave it empty to keep OpenRouter's default routing.

### Ollama

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=<installed-model>
```

Run `make ollama`, then install the model in the service if needed:

```bash
docker compose exec ollama ollama pull <installed-model>
```

### vLLM

```dotenv
LLM_PROVIDER=vllm
LLM_MODEL=<huggingface-model-id>
```

Run `make vllm`. The included profile expects an NVIDIA-compatible Docker runtime. Override `LLM_BASE_URL` when using an already-running OpenAI-compatible server.

## Privacy and failure behavior

The Advisor always receives the objective, dataset profile without its preview, and available registry metadata. Profiles include column names, statistics, and observed category/label values; disabling row sampling does not remove those values. Complete preview rows are included only when `LLM_SEND_SAMPLE=true`, limited by `LLM_SAMPLE_SIZE`; the response reports whether a sample was sent. UI language and Advisor response language are independent. The [configuration guide](configuration.md#data-sent-to-providers) describes sharing by workflow.

Result pages expose a separate, opt-in explanation flow. The overall action sends run metadata,
metrics, and the aggregate values represented by every visible chart. Its validated response must
contain exactly one explanation for each supplied `chart_id`; missing, duplicate, or invented chart
identifiers are rejected. Each chart also offers **Deepen with AI**, which sends the same run-level
context plus only that chart's aggregates. Histogram, residual, prediction, SHAP, and latency
contexts are summarized before transmission, so this flow never adds original dataset rows.

Experiment-level comparison ranks models deterministically by metric direction before an optional LLM request. Reviewed improvement proposals create a new draft and require a separate training action. See [results and AI improvements](results-and-improvements.md) for these contracts and the limits of generated explanations.

Timeouts, transport errors, invalid JSON, schema mismatches, unavailable configuration, and semantically incompatible recommendations return safe API errors. Automated tests use injected transports or the deterministic E2E mock, never paid/live providers. Expert mode remains available when the provider is unavailable.

To add a provider, implement `LLMProvider.generate_structured`, declare capabilities, register it in `llm/registry.py`, add its configuration literal, and test payload, schema validation, redaction-safe errors, and failure mapping.
