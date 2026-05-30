# BigMac: A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

Zewen Jin1 2\*, Shengnan Wang2\*, Jiaan Zhu1 3, Hongrui Zhan<sup>1</sup> , Youhui Bai<sup>2</sup> , Lin Zhang<sup>2</sup> , Zhenyu Ming<sup>2</sup> , Cheng Li1 3

> <sup>1</sup>University of Science and Technology of China <sup>2</sup>Huawei Technologies

3 Institute of Artificial Intelligence, Hefei Comprehensive National Science Center zevin@mail.ustc.edu.cn, wangshengnan12@huawei.com, andyzhu@mail.ustc.edu.cn, zhr2001@mail.ustc.edu.cn, baiyouhui@huawei.com, zhang.lin4@huawei.com, mingzhenyu1@huawei.com, chengli7@ustc.edu.cn

#### Abstract

The Mixture-of-Experts (MoE) structure scales the Transformer-based large language models (LLMs) and improves their performance with only the sub-linear increase in computation resources. Recently, a fine-grained DeepSeekMoE structure is proposed, which can further improve the computing efficiency of MoE without performance degradation. However, the All-to-All communication introduced by MoE has become a bottleneck, especially for the fine-grained structure, which typically involves and activates more experts, hence contributing to heavier communication overhead.

In this paper, we propose a novel MoE structure named Big-Mac, which is also fine-grained but with high communication efficiency. The innovation of BigMac is mainly due to that we abandon the communicate-descend-ascend-communicate (CDAC) manner used by fine-grained MoE, which leads to the All-to-All communication always taking place at the highest dimension. Instead, BigMac designs an efficient descend-communicate-communicate-ascend (DCCA) manner. Specifically, we add a descending and ascending projection at the entrance and exit of the expert, respectively, which enables the communication to perform at a very low dimension. Furthermore, to adapt to DCCA, we re-design the structure of small experts, ensuring that the expert in BigMac has enough complexity to address tokens. Experimental results show that BigMac achieves comparable or even better model quality than fine-grained MoEs with the same number of experts and a similar number of total parameters. Equally importantly, BigMac reduces the end-to-end latency by up to 3.09× for training and increases the throughput by up to 3.11× for inference on state-of-the-art AI computing frameworks including Megatron, Tutel, and DeepSpeed-Inference.

