# 6 Conclusion

AutoMoE is the first framework to design heterogeneous MoE's under computational constraints. It supports adaptive computation i.e. variable compute for different inputs with variable-size experts. It leverages NAS to explore a heterogeneous search space with variable number of experts, sizes, and placement choices; alongside other standard Transformer architectural hyper-parameters. AutoMoE generated MoE subnetworks reduce FLOPs and latency over both manually designed and NASsearched architectures on benchmark MT tasks.

