# 7 Conclusion

In this paper, we propose SDP4Bit, a communication reduction strategy for Sharded Data Parallelism. SDP4Bit reduces both weight and gradient communication to nearly 4 bits while maintaining model accuracy comparable to the baseline. We implemented SDP4Bit in Megatron-LM and optimized it to reduce quantization overhead. Specifically, our experimental results demonstrate a training speedup of up to 4.08 × on 128 GPUs. This paper focuses on LLM pre-training, but we plan to extend our work to other models and areas such as MoE, computer vision, and fine-tuning in the future.

