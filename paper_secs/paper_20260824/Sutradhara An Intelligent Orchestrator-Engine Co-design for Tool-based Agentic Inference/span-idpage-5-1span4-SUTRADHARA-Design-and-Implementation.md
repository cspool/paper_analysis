# <span id="page-5-1"></span>4 SUTRADHARA: Design and Implementation

SUTRADHARA extends the standard LLM serving architecture with a thin coordination layer that enables the orchestrator to communicate semantic hints about agentic request structure to the engine. As before, the orchestrator maintains knowledge of iteration boundaries, prompt composition, and tool dependencies, while the engine controls scheduling, batching, and KV cache management. However, through five new API calls shown in Table 1, the orchestrator guides engine decisions without requiring model-level modifications.

#### 4.1 Parallel execution via prompt splitting

Our analysis revealed that 50-80% of iteration i+1's prompt is available immediately when iteration i completes decode, yet current systems wait for all tool outputs before beginning any prefill computation. This sequential execution leaves parallelism unexploited. However, naively starting prefill before tool outputs arrive introduces two challenges: (1) correctness: the engine must know where to splice tool outputs into the partial prompt, and (2) efficiency: the engine must retain the prefilled KV cache while tools execute, potentially without premature eviction.

<span id="page-5-2"></span>> **[图片提取文字 (无描述)]:**
> Iter 1 Iter 2 Iter 3 LLM Ρ, D2  $D_3$  $\mathbf{D}_1$  $P_3$  $T_{21}$ T<sub>11</sub> Tool calls T22 T<sub>12</sub> T23 (a) Baseline LLM  $P_{2a}$  $D_2$  $P_{3b}$  $\mathbf{D}_3$  $P_1$  $\mathbf{D}_1$ P<sub>2b</sub>  $P_{3a}$ T11 T21 Tool calls T12 T22 T23 (b) Prompt splitting Latency  $P_{2h}$ LLM  $\mathbf{D}_2$  $P_{3a}$  $P_1$ P<sub>2a</sub>  $P_{3b}$ D,  $D_3$ Reduction  $T_{11}$ T21 Tool calls T<sub>12</sub> T22 T23 (c) Prompt splitting and Streaming tool dispatch Tool execution Prefill Decodes
![](_page_5_Figure_8.jpeg)

Figure 6. Intra-request parallel execution in SUTRADHARA

Figure 6 illustrates our approach across three iterations of an agentic request. Each iteration consists of prefill  $(P_i)$  and decodes  $(D_i)$ , followed by tool executions  $(T_{ij})$  for tool j in iteration i). Figure 6a is the baseline sequential execution done by systems today; iteration 1 completes decodes  $D_1$ , all the tool executions  $T_{11}$ ,  $T_{12}$  and then iteration 2 begins prefill  $P_2$ . This pattern repeats for subsequent iterations. The LLM engine schedules iterations from other agentic requests in the idle time between tool calls to keep GPU occupied, creating long intra-request sequential chains of execution.

Prompt splitting as demonstrated in Figure 6b breaks this sequential dependency by partitioning prompts into tool-independent and tool-dependent slices:

- Slice identification: When the iteration i completes decodes and generates tool calls, the orchestrator aware of prompt template, identifies the insertion point where tool outputs will be spliced, typically between system instructions/history and the tool results section.
- Eager prefill execution: The orchestrator submits the tool-independent prefix P\_2a using submit\_partial\_prefill(), which returns a continuation handle. The engine computes prefill while tools execute concurrently. The KV cache blocks from partial prefill are tagged with high priority via set\_reuse\_priority() to prevent eviction.
- 3. **Prompt extension:** Once tool outputs arrive, the orchestrator constructs the tool-dependent suffix and calls  $extend\_prefill()$  with the continuation handle. The engine splices the new content  $(P_{2b})$  onto the pinned KV cache from  $P_{2a}$ , completes the prefill, and proceeds to decode  $D_2$ . If tool execution fails or times out, the orchestrator discards the partial prefill and sets appropriate hints for its KV for the engine to release pinned resources.

This approach overlaps tool execution with a part of prefill, reducing the request's end-to-end latency.

#### 4.2 Streaming tool dispatch with decodes

Agentic systems generate tool calls as structured JSON arrays during the LLM decode phase. Figure 6c shows that current systems wait for complete decode output before parsing and dispatching tools, introducing unnecessary serialization. However, once a complete tool invocation structure is decoded (e.g., the closing } of "tool": "search", "query": "..."), that tool can execute immediately.

SUTRADHARA implements streaming tool dispatch through token-level callbacks from the engine to the orchestrator. When submitting a decode request that will generate tool calls (intermediate iterations), the orchestrator calls <code>register\_streaming\_callback()</code> with a handler function. The engine invokes this handler after each decoded token. The orchestrator maintains a streaming JSON parser that accumulates tokens and identifies complete objects. When a tool call object closes (final } token), the parser extracts the tool name and parameters. As soon as a complete tool invocation is identified, the orchestrator dispatches it for execution without waiting for remaining decode tokens. Subsequent tool calls in the JSON array are dispatched as they complete. The orchestrator tracks completion of all tools before proceeding to the next iteration.

