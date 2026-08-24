# 1 Introduction

Large language models (LLMs) are rapidly evolving from single-turn chatbots into tool-augmented agents [\(Luo et al.,](#page-9-0) [2025a;](#page-9-0) [Yang et al.,](#page-10-0) [2025b;](#page-10-0) [Tran](#page-10-1) [et al.,](#page-10-1) [2025;](#page-10-1) [Wang et al.,](#page-10-2) [2025;](#page-10-2) [Xi et al.,](#page-10-3) [2025;](#page-10-3) [Mo](#page-9-1)[hammadi et al.,](#page-9-1) [2025\)](#page-9-1). LLM agents can interact with external tools and execute multi-step tasks across domains [\(Sapkota et al.,](#page-10-4) [2025\)](#page-10-4), and standardized agent–tool protocols such as the Model Context Protocol (MCP) are accelerating this integration [\(Ehtesham et al.,](#page-8-0) [2025;](#page-8-0) [Hou et al.,](#page-9-2) [2025;](#page-9-2)

[Hasan et al.,](#page-8-1) [2025;](#page-8-1) [Group,](#page-8-2) [2025;](#page-8-2) [Anthropic,](#page-8-3) [2024\)](#page-8-3). As agents are deployed at scale [\(Sun et al.,](#page-10-5) [2025\)](#page-10-5), operational reliability and cost stability emerge as primary concerns; under Unbounded Consumption [\(OWASP,](#page-9-3) [2025\)](#page-9-3), vulnerabilities can lead to severe resource exhaustion as Denial-of-Service (DoS) attacks [\(Xu and Parhi,](#page-10-6) [2025\)](#page-10-6).

Existing research on DoS attacks against LLMs largely forces models to generate excessively long outputs within a single interaction, typically triggered by a malicious user prompt or injected retrieval-augmented generation (RAG) context [\(Zhang et al.,](#page-10-7) [2025c;](#page-10-7) [Gao et al.,](#page-8-4) [2024b;](#page-8-4) [Dong](#page-8-5) [et al.,](#page-8-5) [2025;](#page-8-5) [Geiping et al.,](#page-8-6) [2024;](#page-8-6) [Gao et al.,](#page-8-7) [2024a\)](#page-8-7). For instance, Engorgio and Auto-DoS craft queries that elicit verbose responses that are often offtask [\(Dong et al.,](#page-8-5) [2025;](#page-8-5) [Zhang et al.,](#page-10-7) [2025c\)](#page-10-7), while Overthink [\(Kumar et al.,](#page-9-4) [2025\)](#page-9-4) injects decoy reasoning problems into retrieved context to inflate internal thought while keeping the final answer correct [\(Kumar et al.,](#page-9-4) [2025\)](#page-9-4). Despite their differences, a critical limitation unites these methods: they are fundamentally single-turn attacks operating at the user-query or RAG layer.

This single-turn focus limits their impact in the agentic paradigm for two reasons: costs are capped by the model's per-turn maximum completion, and many attacks (except Overthink) produce generic verbosity that is conspicuous in goal-oriented tool workflows [\(Zhang et al.,](#page-10-7) [2025c;](#page-10-7) [Gao et al.,](#page-8-4) [2024b;](#page-8-4) [Dong et al.,](#page-8-5) [2025;](#page-8-5) [Geiping et al.,](#page-8-6) [2024;](#page-8-6) [Gao et al.,](#page-8-7) [2024a;](#page-8-7) [Louck et al.,](#page-9-5) [2025\)](#page-9-5). Meanwhile, [Zhang et al.](#page-10-8) [\(2025a\)](#page-10-8) show malfunction amplification, where inputs induce repetitive or off-task action loops that often cause task failure. In contrast, the multiturn agent–tool communication loop remains a largely unexplored attack surface for correctnesspreserving economic DoS: the agent still completes the task, but the cost explodes. Table [1](#page-1-0) summarizes key differences between our tool-layer attack and prior single-turn methods.

<sup>\*</sup>Corresponding author.

<span id="page-1-0"></span>

| Aspect                  | Engorgio     | P-DoS        | Auto-DoS     | Overthink    | Ours              |
|-------------------------|--------------|--------------|--------------|--------------|-------------------|
| Correctness preserving  | ✗            | ✗            | ✗            | ✓            | ✓                 |
| Trigger layer           | Dialog query | Dialog query | Dialog query | RAG context  | MCP Tool server   |
| Turns & per-query bound | 1-turn (≤ M) | 1-turn (≤ M) | 1-turn (≤ M) | 1-turn (≤ M) | n-turn (≤ nM)     |
| Long-output site        | Answer step  | Answer step  | Answer step  | Think step   | Tool calling step |
| Access model            | White-box    | White-box    | Black-box    | Black-box    | Black-box         |

Table 1: DoS attack comparison. M is the per-turn max completion; our tool-layer attack enables n-turn amplification while preserving task success ([§4.5\)](#page-6-0).

To address these gaps, we introduce a tool-layer attack that targets the multi-turn agent–tool interaction loop. Our method transforms a benign, MCPcompliant tool server into a malicious variant that steers the agent to repeatedly call the same tool and generate long outputs at the tool-calling step, while still completing the user's task. Across six LLMs and two tool-use benchmarks (ToolBench and BFCL) [\(Fan et al.,](#page-8-8) [2025;](#page-8-8) [Patil et al.,](#page-9-6) [2025\)](#page-9-6), it consistently drives per-query completions beyond 60,000 tokens, inflates token budgets by up to 658×, increases total energy by up to 561×, and raises peak GPU KV-cache usage to over 73% [\(Pan](#page-9-7) [et al.,](#page-9-7) [2025\)](#page-9-7). Crucially, the attack preserves task success and is rarely flagged by representative defenses ([§4.5\)](#page-6-0), enabling substantial degradation of system throughput and OOM-safe concurrency.

Taken together, our findings make three critical contributions:

- This is the first work to devise the tool-calling layer as a first-class DoS attack surface in the agent era: even with correct tool use and correct final answers, representative prompt filters and output or trajectory monitors rarely flag the attack ([§4.5\)](#page-6-0).
- We propose a universal MCTS optimization method that transforms benign MCP servers into malicious variants under text-only, payload-preserving constraints.
- Extensive experiments via six LLMs on Tool-Bench and BFCL benchmarks show that our attack achieves unprecedented resource amplification while maintaining high task success.

