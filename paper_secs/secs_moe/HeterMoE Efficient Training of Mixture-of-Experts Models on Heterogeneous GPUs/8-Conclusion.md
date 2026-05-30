# 8 Conclusion

This paper presents HeterMoE, a system for efficient Mixtureof-Experts (MoE) models training on heterogeneous GPUs. HeterMoE disaggregates attention and expert modules to fully utilize each GPU's capability. HeterMoE introduces zebra parallelism (ZP), along with asymmetric expert assignment (Asym-EA), to enable computation overlapping and fine-grained load balancing. Our evaluations show that Heter-MoE consistently outperforms existing techniques, achieving up to 2.3x speedup over existing MoE training systems, while maintaining an average 95% throughput with half GPUs of newer generation in a homogeneous cluster replaced by older ones. We will open source HeterMoE.

