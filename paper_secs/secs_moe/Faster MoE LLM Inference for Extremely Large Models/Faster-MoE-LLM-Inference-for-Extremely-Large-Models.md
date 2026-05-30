# Faster MoE LLM Inference for Extremely Large Models

Haoqi Yang<sup>†</sup>, Luohe Shi<sup>†</sup>, Qiwei Li, Zuchao Li, Ping Wang, Bo Du Wuhan University Wuhan, 430072, P. R. China

### Mengjia Shen

Wuhan Second Ship Design And Research Institute Wuhan, 430010, P. R. China

#### Hai Zhao

Shanghai Jiao Tong University Shanghai, 200240, P. R. China

## **Abstract**

Sparse Mixture of Experts (MoE) large language models (LLMs) are gradually becoming the mainstream approach for ultra-large-scale models. Existing optimization efforts for MoE models have focused primarily on coarse-grained MoE architectures. With the emergence of DeepSeek Models, fine-grained MoE models are gaining popularity, yet research on them remains limited. Therefore, we want to discuss the efficiency dynamic under different service loads. Additionally, fine-grained models allow deployers to reduce the number of routed experts, both activated counts and total counts, raising the question of how this reduction affects the trade-off between MoE efficiency and performance. Our findings indicate that while deploying MoE models presents greater challenges, it also offers significant optimization opportunities. Reducing the number of activated experts can lead to substantial efficiency improvements in certain scenarios, with only minor performance degradation. Reducing the total number of experts provides limited efficiency gains but results in severe performance degradation. Our method can increase throughput by at least 10% without any performance degradation. Overall, we conclude that MoE inference optimization remains an area with substantial potential for exploration and improvement.

