# FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

Heming Zou<sup>1\*</sup> Yunliang Zang<sup>2\*</sup> Wutong Xu<sup>1</sup> Yao Zhu<sup>1</sup> Xiangyang Ji<sup>1†</sup>

Department of Automation, Tsinghua University

Academy of Medical Engineering and Translational Medicine, Tianjin University

{zouhm24, xwt22}@mails.tsinghua.edu.cn

yunliangzang@tju.edu.cn, ee\_zhuy@zju.edu.cn

xyji@tsinghua.edu.cn

#### **Abstract**

Low-Rank Adaptation (LoRA) is a widely used parameter-efficient fine-tuning method for foundation models, but it suffers from parameter interference, resulting in suboptimal performance. Although Mixture-of-Experts (MoE)-based LoRA variants show promise in mitigating intra-task correlations in single-task instruction tuning, they introduce additional router parameters and remain ineffective in multi-task model merging where inter-task interference arises. Inspired by the fly olfactory circuit, we propose FlyLoRA, an implicit MoE-based LoRA variant that introduces: (1) rank-wise expert activation in the up-projection matrix, and (2) an implicit router that unifies expert routing and down-projection, where a frozen sparse random projection matrix replaces the traditional dense trainable version. This design resolves the trade-off between intra-task decorrelation and computational efficiency by eliminating the need for an explicit router, while inherently mitigating intertask interference due to the orthogonality property of random matrices. Extensive experiments across four domains—general knowledge understanding, scientific question answering, mathematical reasoning, and code generation—demonstrate consistent performance improvements over existing methods. Beyond empirical gains, FlyLoRA highlights how biological structures can inspire innovations in AI technologies. Code is available at https://github.com/gfyddha/FlyLoRA.

