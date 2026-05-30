# **6 Conclusion**

In this work, we show how to enable sparse MoE architectures for on-device inference use-cases along the three on-device dimensions: Quality, Memory and Latency. Specifically, we show that in a fair comparison, MoE-style models outperform their dense counterparts on language modeling tasks by over over +2.35%. Introducing our novel weight-decomposed experts, we show further performance gains of up to +1.1% compared to standard MoE models. To truly enable MoE-style models for on-device use-cases, we tackle the model offloading bottleneck by reducing expert offloads in the training stage and, in turn, reduce model inference latency. Our "grouped expert selection" loss term thereby improves expert offloading efficiency by 6x and increases generation speed by 50% compared to standard offloaded MoE models.

With the results presented in this paper, we effectively pave the way to unlock the potential of MoE-style architectures in on-device scenarios, supporting high quality, privacy preserving foundational models for edge devices.

