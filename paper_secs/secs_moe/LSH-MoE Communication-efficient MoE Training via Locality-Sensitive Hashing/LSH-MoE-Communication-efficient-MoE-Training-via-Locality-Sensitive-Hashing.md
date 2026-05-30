# LSH-MoE: Communication-efficient MoE Training via Locality-Sensitive Hashing

Xiaonan Nie<sup>1</sup> Qibin Liu<sup>1</sup> Fangcheng Fu<sup>1</sup> Shenhan Zhu<sup>1</sup> Xupeng Miao<sup>2</sup>
Xiaoyang Li<sup>3</sup> Yang Zhang<sup>3</sup> Shouda Liu<sup>3</sup> Bin Cui<sup>1</sup>

<sup>1</sup>Peking University <sup>2</sup>Purdue University <sup>3</sup>ByteDance

<sup>1</sup>{xiaonan.nie,2101212782,ccchengff,shenhan.zhu,bin.cui}@pku.edu.cn

<sup>2</sup>xupeng@purdue.edu <sup>3</sup>{lixiaoyang.x,zhangyang.elfin,liushouda}@bytedance.com

#### **Abstract**

Larger transformer models always perform better on various tasks but require more costs to scale up the model size. To efficiently enlarge models, the mixture-ofexperts (MoE) architecture is widely adopted, which consists of a gate network and a series of experts and keep the training cost constant by routing the input data to a fixed number of experts instead of all. In existing large-scale MoE training systems, experts would be distributed among different GPUs for parallelization, and thus input data requires additional all-to-all communications to access the target experts and conduct corresponding computations. However, upon evaluating the training process of three mainstream MoE models on commonly used GPU clusters, we found that the all-to-all communication ratio averaged around 45%, which significantly hinders the efficiency and scalability of training MoE models. In this paper, we propose LSH-MoE, a communication-efficient MoE training framework using locality-sensitive hashing (LSH). We first present the problems of scaling MoE training in existing systems and highlight the potential of exploiting token similarity to facilitate data compression. Then, we introduce an efficient LSH-based compression technique, which utilizes the cross-polytope hashing for rapid clustering and implements a residual-based error compensation scheme to alleviate the adverse impact of compression. To verify the effectiveness of our methods, we conduct experiments on both language models (e.g., RoBERTa, GPT, and T5) and vision models (e.g., Swin) for pre-training and fine-tuning tasks. The results demonstrate that our method substantially outperforms its counterparts across different tasks by  $1.28 \times$  -  $2.2 \times$  of speedup.

