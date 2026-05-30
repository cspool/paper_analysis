# *C. Motivations*

*1) Duplicate tokens with the hierarchical topology:* Since each GPU would hold E/G experts per MoE layer when the number of experts is larger than the number of GPUs (i.e., E/G > 1), tokens are required to be *redundantly* transmitted to multiple selected experts that are located on the same GPU to exploit the AlltoAll collective. As illustrated in Fig. [3a,](#page-2-0) each expert is assigned particular tokens, and a single token requires multiple experts (i.e., K in the top-K selection). Assume that each GPU (or group) holds two experts as shown in Fig [3b.](#page-2-0) Every group would have duplicate tokens, which results in redundant communication in the AlltoAll operation. *Thus, eliminating this duplication can reduce the communication traffic as shown in Fig. [3c](#page-2-0)*.

Moreover, the duplication rate is highly affected by the number of groups (say R) and the number of selected experts (i.e., K) per token. We conduct preliminary experiments with different R and K to measure the duplication rate at each group as shown in Table [II.](#page-2-1) The results indicate that lower R (The hierarchical topology can divide experts into different groups.) and higher K (which is very common in modern MoE models like DeepSeek-V3, Qwen-MoE, etc.) would result in a higher duplication rate. *Thus, how to eliminate the duplicated tokens by considering* K*,* R*, and the GPU topology becomes more challenging.*

*2) Unbalanced routing workloads with the hierarchical topology:* Since the selected experts for each token are determined by the routing function, it is easy to cause imbalanced workloads for each expert, which results in increased commu-

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

Fig. 4: Four types of hierarchical AlltoAll with different dimensions. The example has two nodes with eight GPUs per

node. We use "..." to omit some GPUs and nodes.

nication traffic [\[2,](#page-9-2)[7,](#page-9-1)[23\]](#page-9-12). Existing solutions like FlexMoE [\[38\]](#page-10-5) and SmartMoE [\[23\]](#page-9-12) dynamically adjust expert placement during training to balance token distribution across GPUs, *but they neither account for token deduplication nor adapt to the hierarchical topology of GPUs.* If the token duplications have been overlooked, simply swapping experts to balance the workload could result in a higher communication overhead. For example, as shown in Fig. [3d,](#page-2-0) we swap expert 1 and expert 3 such that the workload of each group is more balanced, but its communication traffic becomes higher than that of Fig. [3c.](#page-2-0)

*Therefore, it requires a new expert swap strategy taking into account token deduplication and hierarchical bandwidth constraints to achieve higher training performance.*

