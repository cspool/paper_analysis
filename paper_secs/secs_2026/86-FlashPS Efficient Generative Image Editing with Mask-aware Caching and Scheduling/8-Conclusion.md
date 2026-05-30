# 8 Conclusion

We presented FlashPS, an efficient system for generative image editing. FlashPS effectively leverage the sparsity introduced by masks and proposes three novel designs: (1) an efficient pipeline of computing and cache loading to accelerate inference while maintaining image quality (2) a tailored continuous batching for diffusion models, and (3) a maskaware load balance policy to route requests. Collectively, these designs accelerate the inference of image editing and improve the cluster-level serving performance. Compared to existing systems, FlashPS achieves 3× higher throughput and reduces average request serving latency by up to 14.7× while maintaining image quality.

