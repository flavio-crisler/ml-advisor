# ADR 0001: Keep DecisionProvider Separate from LLMProvider

Status: Accepted

## Context

The workbench uses generative LLMs to propose and explain strategies. Decision Models instead
produce narrow, typed probabilistic judgments that are evaluated, calibrated, cached, and reused
as numerical signals.

## Decision

`DecisionProvider` is a separate domain protocol with Noul, Choice, and Score contracts. It does
not inherit from `LLMProvider`, and concrete provider SDK or HTTP details remain in infrastructure.
Application services depend only on the provider protocol, registry, and versioned specs.

## Consequences

The system can replace Jev or add another Decision Model without changing ML registries,
`ExperimentEngine`, `PipelineCompiler`, `LLMProvider`, or frontend core. Provider responses need a
normalization boundary before they enter application workflows.
