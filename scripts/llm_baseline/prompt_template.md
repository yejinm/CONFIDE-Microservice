# Prompt Template - LLM Microservice Extraction Baseline

## Role
You are a **senior software architect** experienced in Java enterprise monolith modernization and microservice decomposition.

## Context
We are analyzing a Java monolithic system. The goal is to partition classes into a small set of microservices.

You must produce a clustering/partition of classes into services.

## Inputs
You will be given:

1) **Class list** (fully qualified class names, FQCN)
2) **Static dependency edges** among classes (caller -> callee or uses -> used)
3) (Optional) **Semantic hints** (short summaries or identifiers)

## Task
1) Partition the classes into microservices. Each microservice should have:
   - high internal cohesion (classes that belong together by domain responsibility)
   - low coupling to other services

2) Provide a brief explanation of the rationale.

3) Output *machine-readable JSON* mapping:

```json
{
  "com.example.Foo": 0,
  "com.example.Bar": 0,
  "com.example.Baz": 1
}
```

## Constraints
- Use **integer** service ids starting from 0.
- Every input class must appear exactly once as a key.
- Keep the number of services reasonable; prefer 3–10 unless the system is very large.
- Prefer grouping by business capabilities (e.g., account, catalog, order, trading, portfolio, admin).

## Notes
- If dependencies suggest a shared utility layer, you may keep a separate service id for it *only if necessary*.
- If uncertain, favor clearer modular boundaries and explain uncertainty.
