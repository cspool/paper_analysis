# Abstract

This technical report presents QWENLONG-CPRS, a context compression framework designed for explicit long-context optimization, addressing prohibitive computation overhead during the prefill stage and the "lost in the middle" performance degradation of large language models (LLMs) during long sequence processing. Implemented through a novel *dynamic context optimization* mechanism, QWENLONG-CPRS enables multi-granularity context compression guided by natural language instructions, achieving both efficiency gains and improved performance.

Evolved from the Qwen architecture series, QWENLONG-CPRS introduces four key innovations: (1) Natural language-guided dynamic optimization, (2) Bidirectional reasoning layers for enhanced boundary awareness, (3) Token critic mechanisms with language modeling heads, and (4) Window-parallel inference.

Comprehensive evaluations across five benchmarks (4K-2M word contexts) demonstrate QWENLONG-CPRS's threefold effectiveness: (1) Consistent superiority over other context management methods like RAG and sparse attention in both accuracy and efficiency. (2) Architecture-agnostic integration with *all* flagship LLMs, including GPT-4o, Gemini2.0-pro, Claude3.7-sonnet, DeepSeek-v3, and Qwen2.5 max, achieves 21.59× context compression alongside 19.15-point average performance gains; (3) Deployed with Qwen2.5-32B-Instruct, QWENLONG-CPRS surpasses leading proprietary LLMs by 4.85 and 10.88 points on Ruler-128K and InfiniteBench, establishing new SOTA performance.

