# 6 CONCLUSION

In this paper we study how to compress MoE models by merging experts. We first analyze the theoretical essence of the expert merging in MoE models. Unlike the traditional view that focuses on merging expert parameters, we introduce a novel perspective that interprets expert merging as expert output merging. Under this perspective, the merging process can be formulated as inserting additional matrices into the forward computation. Building on this theoretical insight, we propose our solution, MergeMoE, which uses mathematical tools to optimize the design of the compression matrices in the expert-merging process. Our experiment results show that, compared with baseline algorithms, MergeMoE consistently achieves better performance at the same compression ratio.

