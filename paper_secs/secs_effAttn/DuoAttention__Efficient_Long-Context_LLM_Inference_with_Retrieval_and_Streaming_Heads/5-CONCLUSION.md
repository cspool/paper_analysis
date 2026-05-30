# 5 CONCLUSION

We introduce DuoAttention, a framework that optimizes memory and computational resources in LLMs by distinguishing between *Retrieval Heads* and *Streaming Heads*. By applying a full KV cache only to retrieval heads, DuoAttention significantly reduces memory usage and latency for both decoding and pre-filling in long-context applications. It achieves memory reductions of up to 2.55× for MHA and 1.67× for GQA models, with decoding speed improvements of up to 2.18× for MHA and 1.50× for GQA, and pre-filling accelerations of up to 1.73× and 1.63×, respectively, with minimal accuracy loss compared to full attention. When combined with quantization, DuoAttention further boosts KV cache capacity, supporting up to 3.30 million contextual tokens on a single A100 GPU. DuoAttention paves the way for LLMs to handle contexts with millions of tokens.

## ACKNOWLEDGMENTS

We thank MIT-IBM Watson AI Lab, MIT and Amazon Science Hub, MIT AI Hardware Program, National Science Foundation, Hyundai and Samsung for supporting this research. We thank NVIDIA for donating the DGX server.

