# 7 Conclusion

In summary, this work provides a comprehensive revisit of the MoE-based structure for LoRA and analyzes its drawbacks regarding parameter interference and efficiency. Inspired by the fly olfactory circuit, we introduce FlyLoRA, a novel MoE-based LoRA variant that employs rank-wise expert activation in matrix B and a fixed sparse random projection for matrix A as an implicit router. Through the theoretical properties of these components, FlyLoRA achieves both intra-task and intertask decoupling, significantly improving decorrelation in single-domain instruction tuning and LoRA component fusion in multi-task settings. Additionally, the implicit routing strategy and inherent sparsity ensure computational efficiency.

