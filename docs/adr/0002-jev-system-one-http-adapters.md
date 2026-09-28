# ADR 0002: Use System One HTTP Adapters for Jev

Status: Accepted

## Context

TypeSafe and OpenRouter expose the same System One request and response shape. Their base URLs,
authentication credentials, model identifiers, and supplemental usage metadata differ. The Python
SDK is convenient but is not required for this repository's small, provider-neutral boundary.

## Decision

Use a shared asynchronous `DecisionHttpClient`, retry policy, serializer, and response normalizer.
Keep thin `TypeSafeJevProvider` and `OpenRouterJevProvider` adapters for provider identity, base URL,
and capabilities. The HTTP client retries only transient transport errors, HTTP 429, and HTTP 529.

## Consequences

The domain remains free of provider dependencies, tests can replace transport deterministically,
and provider-specific metadata is normalized in one place. Contract changes require adapter tests
and a review of the external documentation recorded in the Stage 21 specification.
