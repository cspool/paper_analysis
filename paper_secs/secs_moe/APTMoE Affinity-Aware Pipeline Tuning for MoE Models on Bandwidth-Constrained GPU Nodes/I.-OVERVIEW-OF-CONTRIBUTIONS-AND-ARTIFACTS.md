# I. OVERVIEW OF CONTRIBUTIONS AND ARTIFACTS

#### *A. Paper's Main Contributions*

This paper presents APTMoE, an affinity-aware pipeline fine-tuning system for MoE models targeting at bandwidthconstrained GPU nodes. APTMoE enhances the computational efficiency and the model size for fine-tuning MoE models on limited number of bandwidth-constrained GPU nodes. APTMoE includes the affinity-aware offloading technique to enhance the pipeline parallelism, with the key idea to offload a portion of the affinity computation to the CPU, so as to better manage data across heterogeneous memory. Our contributions are summarized as follows:

- C<sup>1</sup> The hierarchical loading strategy. With the prior knowledge of expert popularity and computation affinity, it employs three loading phases to greedily allocate computation with the highest affinity and minimize data movement volume.
- C<sup>2</sup> The demand-priority scheduling strategy. It is used to alleviate the mutual interference among loading phases and maximize the bandwidth utilization by dynamically coordinating the loading order.
- C<sup>3</sup> Expert popularity simulator for evaluation. It proxies the gate and predictor for both generalized and real MoE models, so as to evaluate APTMoE on finetuning MoE models.

## *B. Computational Artifacts*

