# 4 Implementation

We implement Aragog as an orchestration layer in 6.8K lines of Python and Triton code that sits between agentic workflow frontends and LLM serving engines, without modifying either.

Frontend and Workflow Specification. As shown in Figure [11,](#page-7-0) Aragog converts any DSPy program into its internal workflow representation, enabling runtime model reconfiguration and asynchronous stage-wise execution. We choose DSPy for its lightweight design and ease of use, making it popular for building agentic workflows. However. Aragog can support other frameworks by implementing frameworkspecific workflow graph extraction.

Router Inference. We implement dynamic batching for shared embedding model serving. To accelerate router classifier inference, we develop custom Triton kernels that fuse classifiers layer by layer. We leverage CUDA graphs to further reduce the kernel launching overheads.

Runtime Scheduler. The scheduler runs asynchronously with workflow execution. It maintains: (1) FIFO queues with look-ahead scheduling, (2) model engine states tracking occupancy, (3) configuration caches mapping requests to viable sets, and (4) exponential moving averages of queue metrics. Beam search uses priority queues, where each state encodes partial assignments as bit vectors for efficient manipulation.

Serving Backend Integration. Aragog uses SGLang [\[62\]](#page-14-6), a widely used LLM serving engine as our serving backend. However, Aragog supports any serving backends (vLLM [\[21\]](#page-12-15), TensorRT-LLM [\[11\]](#page-12-16) and HuggingFace TGI [\[19\]](#page-12-17)) that provide OpenAI-compatible APIs needed for agentic workflow frontends.

