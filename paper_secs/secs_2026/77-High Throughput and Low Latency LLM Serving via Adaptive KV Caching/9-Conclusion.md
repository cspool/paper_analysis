# 9 Conclusion

This paper presents eLLM, an innovative LLM serving system designed to maximize GPU utilization through an adaptive KV caching mechanism that stores only partial caches during decoding. eLLM applies dual-level optimization: at the request level, it adjusts batch sizes and tokenwise caching using latency quantification models; at the layer level, it enables adaptive kernel fusion and effective communication-computation overlap. These optimizations deliver high throughput and low latency while ensuring TPOT SLO compliance under dynamic request loads.

