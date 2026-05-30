# **7 Conclusion**

We present LightTransfer, a lightweight framework for transforming standard transformers into hybrid models for more efficient generation by identifying *lazy* layers and replacing their full-attention modules with streaming attention. Extensive experiments show that even when half of the transformer layers are replaced with streaming attention, LightTransfer delivers up to a 2.17× increase in throughput while incurring less than a 1.5% performance drop on LongBench. For advanced long reasoning generation tasks like AIME24, our method achieves these gains without any performance degradation on QwQ-STILL.

