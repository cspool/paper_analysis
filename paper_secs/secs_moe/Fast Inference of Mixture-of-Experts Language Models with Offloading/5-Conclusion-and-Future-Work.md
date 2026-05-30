# 5 Conclusion and Future Work

In this work, we explore strategies for accelerating Mixture-of-Experts based language models on consumer hardware with limited GPU memory. We propose a MoE-centric approach to offloading and explore how mixed quantization affects perplexity and performance on language understanding tasks. We evaluate the proposed strategies and show that they produce a significant increase in generation speed compared to nave approaches on consumer-grade hardware, including free-tier ¨ Google Colab.

Our method provides a practical solution for inferencing large MoE language models on resourceconstricted hardware, enabling broader access to these powerful models for research and development. As future work, we plan to explore further offloading strategies, based on speculative expert prediction.

