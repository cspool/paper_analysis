# 4 **SkyWalker** Implementation

We implemented SkyWalker (Figure [7\)](#page-8-0), a prototype system that leverages geo-distributed load balancers to achieve both high throughput and low latency for online LLM serving across multiple geographical regions. SkyWalker is built on top of SkyServe [\[35\]](#page-13-7), an open-source multi-region serving framework for AI models, which supports both on-premise and cloud-based replicas. SkyWalker extends SkyServe by adding geo-distributed load balancer with ≈ 3000 lines of Python code, and is compatible with any inference engine with OpenAI API, such as vLLM [\[33\]](#page-13-9) and SGLang [\[71\]](#page-15-0).

