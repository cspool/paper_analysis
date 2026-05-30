# X. DISCUSSION

Impact of Hardware Advancements. SLINFER currently targets small- to mid-sized LLMs. For large models, SLIN-FER falls back to ServerlessLLM [26]'s exclusive allocation approach (recall §IX-E). Besides, current CPUs are still slow for tight SLOs and long inputs—decoding of Llama-3.1-8B takes at least 74 ms, and processing 32k inputs takes 84 s. However, CPU's capabilities are rapidly evolving: the 32-core 4th Gen Xeon we use delivers 105 TFLOPS (BF16) compared to 13 TFLOPS on a 32-core 3rd Gen Xeon, and the latest 96-core 6th Gen [12] reaches 297 TFLOPS. Meanwhile, GPU memory capacity is also increasing. These advancements offer further performance gains and greater model-sharing potential.

Serving Quantized Models. Applying quantization further enhances SLINFER's sharing capacity by reducing the memory footprint of each instance. When serving 32 22B-sized models [4], applying INT4 quantization [41] reduced GPU usage from 3.8 to 2.6. This improvement stems from the fact that the model weights alone consume 44GB, making quantization essential for sharing on a 80GB GPU.

