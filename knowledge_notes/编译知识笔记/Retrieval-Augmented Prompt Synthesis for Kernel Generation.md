## Retrieval-Augmented Prompt Synthesis for Kernel Generation

术语是什么？
Retrieval-Augmented Prompt Synthesis是KernelEvolve的动态prompt构建机制：不将完整历史知识保持在LLM working memory（会耗尽context window），而是通过targeted retrieval从持久化知识库按需加载相关context。由两级pipeline实现：Context Memory Sub-Agent分析runtime profiling artifacts诊断bottleneck → Deep Search Sub-Agent根据诊断结果检索知识库targeted content → 动态合成优化prompt注入LLM context window (64K-1M tokens)。这反映human-like pattern——人类通过外部文件系统按需检索，而非记忆整个信息体。

从编译框架角度拆解术语：
```
Retrieval Pipeline Architecture:

Context Memory Sub-Agent:
  Input: kernel source + profiling metrics + error diagnostics + correctness results
  Processing: LLM分析profiling data → bottleneck diagnosis
    (例: "30% H100 occupancy + high shared memory pressure → register spilling root cause")
  Output: structured optimization directives
           "Reduce register usage via value recomputation, optimize warp-level memory access"

Deep Search Sub-Agent:
  Input: bottleneck diagnosis reports (parameterize retrieval targets)
  Processing: 两级检索
    1. Query index.md → 定位相关modules
       (e.g., hardware/nvidia/optimization/{tma, shared_memory, on_device_tma}.md)
    2. Fetch targeted content from identified modules
    分层剪枝: platform(硬件平台) → concern(问题类型) → specificity(具体技术)
  Output: hardware constraints + optimization patterns + reference code samples

Progressive Specialization (iterative refinement):
  Iteration 1: 检索broad guidance (Triton basics, general optimization principles)
  Iteration 2-N: navigating progressively specialized content guided by profiling feedback
  例 (GEMM on H100):
    → hardware/nvidia/arch/tensor_cores.md (Tensor Core basics)
    → hardware/nvidia/tlx/{overview, warp_specialization, async_tensor_core}.md
    → code_samples/{hopper-gemm-pipelined, hopper-gemm-ws}.py
```

术语一般如何实现？如何使用？
Persistent knowledge base组织为hierarchical file system: constraints/ (correctness requirements) + guidance/ (platform-agnostic optimization) + hardware/{nvidia|amd|mtia}/ (≥100 documents per platform)。index.md提供structured navigation作为human-readable reference和machine-parseable retrieval signal。对于MTIA等proprietary accelerator (absent from LLM pretraining corpora)，knowledge injection通过在runtime context中educate LLM使其生成leveraging hardware-specific features的kernel——无需model retraining。Deep Search Sub-Agent通过MCP tools执行unified code search，自动dereference knowledge base references到production codebases。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
