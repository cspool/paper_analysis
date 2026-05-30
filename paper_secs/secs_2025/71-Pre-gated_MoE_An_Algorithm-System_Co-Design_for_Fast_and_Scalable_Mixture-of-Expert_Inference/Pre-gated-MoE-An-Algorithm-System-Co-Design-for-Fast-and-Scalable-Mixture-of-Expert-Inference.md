# Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

Ranggi Hwang∗† KAIST ranggi.hwang@kaist.ac.kr

Jianyu Wei∗† USTC / Microsoft Research noob@mail.ustc.edu.cn

Shijie Cao Microsoft Research shijiecao@microsoft.com

Changho Hwang Microsoft Research changhohwang@microsoft.com

Xiaohu Tang† Microsoft Research v-xiaohutang@microsoft.com

Ting Cao Microsoft Research ting.cao@microsoft.com

Mao Yang Microsoft Research maoyang@microsoft.com

*Abstract*—Large language models (LLMs) based on transformers have made significant strides in recent years, the success of which is driven by scaling up their model size. Despite their high algorithmic performance, the computational and memory requirements of LLMs present unprecedented challenges. To tackle the high compute requirements of LLMs, the Mixture-of-Experts (MoE) architecture was introduced which is able to scale its model size without proportionally scaling up its computational requirements. Unfortunately, MoE's high memory demands and dynamic activation of sparse experts restrict its applicability to real-world problems. Previous solutions that offload MoE's memory-hungry expert parameters to CPU memory fall short because the latency to migrate activated experts from CPU to GPU incurs high performance overhead. Our proposed Pre-gated MoE system effectively tackles the compute and memory challenges of conventional MoE architectures using our algorithm-system codesign. Pre-gated MoE employs our novel pre-gating function which alleviates the dynamic nature of sparse expert activation, allowing our proposed system to address the large memory footprint of MoEs while also achieving high performance. We demonstrate that Pre-gated MoE is able to improve performance, reduce GPU memory consumption, while also maintaining the same level of model quality. These features allow our Pre-gated MoE system to cost-effectively deploy large-scale LLMs using just a single GPU with high performance.

*Index Terms*—Mixture-of-expert, inference system, machine learning, large language model, memory offloading

