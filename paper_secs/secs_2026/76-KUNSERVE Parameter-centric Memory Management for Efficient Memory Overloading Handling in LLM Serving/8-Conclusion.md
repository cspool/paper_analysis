# 8 Conclusion

In this paper, we are the first to demonstrate that parametercentric memory management can effectively address the latency spikes caused by memory overloading in LLM serving. We built KUNSERVE, an LLM serving system that cooperatively drops parameters to free up memory to eliminate queuing under overloading. We also proposed a set of techniques to ensure all requests execute efficiently after parameter dropping, including drop plan generation with local unified memory management, coordinated KVCache exchange and lookahead batch formulation. KUNSERVE reduces tail TTFT by up to 72.2 × compared to state-of-the-art systems like Llumnix, vLLM and InferCept.

