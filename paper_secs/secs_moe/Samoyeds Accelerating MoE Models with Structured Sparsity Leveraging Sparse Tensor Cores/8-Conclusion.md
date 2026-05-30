# 8 Conclusion

This paper presents Samoyeds, a novel acceleration system for MoE LLMs with software-hardware co-optimization. We introduce a new sparse format tailored to the dual-sided sparsity inherent in MoE LLMs and implement a bespoke sparsesparse multiplication kernel leveraging SpTC to eliminate redundant computation. Additionally, systematic optimizations specifically designed for this workload and memory access pattern are applied to the MoE execution flow, further enhancing overall efficiency. Evaluation results demonstrate that Samoyeds outperforms SOTA solutions in both computation speed and memory efficiency, while also providing superior model accuracy and hardware compatibility.

