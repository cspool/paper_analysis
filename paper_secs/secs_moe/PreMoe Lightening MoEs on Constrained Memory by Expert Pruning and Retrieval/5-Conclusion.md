# **5 Conclusion**

This work introduced PreMoE, a training-free framework that enables proactive compilation of specialized MoE instances tailored to deployment scenarios. By extracting computational patterns through novel Predicted Expert Utility metric, which refines router logits via highconfidence filtering and logit transformation, PreMoE identifies critical experts for any domain. Our experiments on models from 30B to 718B parameters show that 50% sparsity can be achieved with nearly no performance loss.

