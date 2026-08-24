# 5 Continuum System Design

In Continuum, our design goal is a modular architecture that requires minimal changes to the core inference-engine scheduler loop. On the client side, we attach a program identifier (program\_id) to every inference request so the system can recognize multi-turn agent programs and reason about tool calls across steps.

Upon arrival at the serving engine, requests enter the existing scheduler loop. Continuum adds a thin Tool-Call Handler that is invoked on request arrival and completion. The handler parses tool calls from LLM outputs, tracks per-tool latency using observed inter-request intervals within the same program\_id, and returns TTL to the scheduler. The scheduler uses this hint to pin the request's KV cache for potential reuse by the next step, and later unpins it either when the TTL value expires or when the program terminates.

