# 6 Conclusion

In this paper, we propose MoDSE, a novel structure for MoE layers. Inspired by the varying difficulties of next-token-generating tasks, we introduce the diverse size expert design, providing each expert with different prediction abilities. Our analysis of token routing distribution shows that MoDSE directs tokens to experts whose sizes are best suited for specific token generation tasks. This enhancement improves the MoE model's performance in auto-regression tasks and demonstrates superior results compared to the conventional MoE structure. Additionally, we present the expert-pair allocation method to address the issue of load imbalances in the diverse size expert design, making the MoDSE design more practical.

