# 5 Related Work

#### 5.1 Mixture of Experts (MoE) Models

Mixture of Experts (MoE) models [\[27,](#page-15-12) [1\]](#page-14-0) have gained significant attention in the field of large language models due to their ability to scale model capacity while maintaining computational efficiency. The MoE architecture employs a gating mechanism to selectively activate a subset of expert networks for each input token. This approach allows for an increased model capacity without a proportional increase in computational cost during inference and training.

Recent work has focused on improving the scalability and efficiency of MoE models. Switch Transformer [\[19\]](#page-15-4) simplified the MoE architecture by using a top-1 routing mechanism and demonstrated the ability to scale to trillion-parameter models. The GShard framework [\[2\]](#page-14-1) addressed challenges in training large-scale MoE models, introducing techniques such as expert capacity thresholds and local group dispatching to improve load balancing and training stability.

Research has explored various aspects of expert specialization and routing mechanisms in MoE models [\[28,](#page-15-13) [29\]](#page-15-14). Studies have investigated the impact of the number of experts on model performance, finding that increasing the number of experts leads to improved sample efficiency and faster training, albeit with diminishing returns beyond certain thresholds [\[20\]](#page-15-5). The choice of routing algorithm (e.g., top-1 vs top-2) and gating function (e.g., softmax vs sigmoid) has also been examined [\[30,](#page-15-15) [31,](#page-15-16) [32\]](#page-15-17), with Mixtral 8x7B switching softmax and topK in the router [\[3\]](#page-14-2).

#### 5.2 Upcycling and Model Expansion

The concept of upcycling in the context of MoE models refers to the practice of leveraging pre-trained dense models to initialize MoE architectures. This approach has gained traction as a way to efficiently create large-scale MoE models while benefiting from the knowledge captured in existing pre-trained checkpoints. Notable work in this area includes:

Sparse Upcycling: [\[15\]](#page-15-0) proposed a method for training MoE models from dense checkpoints, demonstrating the ability to expand model capacity while maintaining or improving performance. Recently, Qwen [\[17\]](#page-15-2) and Deepseek [\[14\]](#page-14-13) have adopted this approach. However, the recipe to scale upcycling beyond 1B parameters is not well known.

Network Growth: Research on model expansion techniques, such as those explored in the Gopher model [\[33\]](#page-16-1), has shown that it's possible to significantly increase model size while maintaining performance comparable to or better than models trained from scratch.

Progressive Expansion: Approaches like LLAMA PRO [\[10\]](#page-14-9) have investigated progressive expansion techniques, where model size is increased gradually during training [\[34,](#page-16-2) [35,](#page-16-3) [36,](#page-16-4) [37\]](#page-16-5).

## 5.3 Challenges and Ongoing Research

Despite the promising results in upcycling and MoE pretraining, several challenges remain active areas of research:

Expert Collapse: The phenomenon of expert collapse, where certain experts become underutilized or inactive, has been observed in MoE training. While some studies suggest that expert collapse may not necessarily harm model accuracy, addressing this issue remains an important consideration in MoE design [\[38,](#page-16-6) [31\]](#page-15-16).

Load Balancing: Ensuring an even distribution of work across experts continues to be a challenge, with various approaches proposed to improve load balancing, including auxiliary losses and specialized routing mechanisms. Skywork-MoE [\[26\]](#page-15-11) used aux loss to promote expert diversity during upcycling. BTX [\[39,](#page-16-7) [40\]](#page-16-8) trained experts on different tasks and then mixed them. DBRX [8](#page-13-0) added weight norm on experts' weights.

Efficient Training and Inference: Ongoing work focuses on optimizing the training and inference processes for MoE models, including techniques for reducing communication costs and improving parallelization strategies. Deepseek-MoE [\[16\]](#page-15-1) and Snowflake Arctic [9](#page-13-1) parallelized shared expert to increase utilization. MegaBlocks [\[41\]](#page-16-9) reformulated MoE computation in terms of block-sparse operations. Scattermoe [\[42\]](#page-16-10) optimized memory for finegrained MoE.

