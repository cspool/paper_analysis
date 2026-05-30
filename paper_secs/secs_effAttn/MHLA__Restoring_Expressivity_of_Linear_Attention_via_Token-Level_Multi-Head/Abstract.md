# **Abstract**

While the Transformer architecture dominates many fields, its quadratic self-attention complexity hinders its use in large-scale applications. **Linear attention** offers an efficient alternative, but its direct application often degrades performance, with existing fixes typically re-introducing computational overhead through extra modules (e.g., depthwise separable convolution) that defeat the original purpose. In this work, we identify a key failure mode in these methods: **global context collapse**, where the model loses representational diversity. To address this, we propose **Multi-Head Linear Attention (MHLA)**, which preserves this diversity by computing attention within divided heads along the token dimension. We prove that MHLA maintains linear complexity while recovering much of the expressive power of softmax attention, and verify its effectiveness across multiple domains, achieving a **3.6%** improvement on ImageNet classification, a **6.3%** gain on NLP, a **12.6%** improvement on image generation, and a **41%** enhancement on video generation under the same time complexity.

**Project Page:** <https://dagroup-pku.github.io/MHLA>

**GitHub Repo:** <https://github.com/DAGroup-PKU/MHLA>

**Huggingface Repo:** <https://huggingface.co/DAGroup-PKU/MHLA>

