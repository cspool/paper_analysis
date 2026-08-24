# OpenBA-V2: Reaching 77.3% High Compression Ratio with Fast Multi-Stage Pruning

Dan Qiao, Yi Su, Pinzheng Wang, Jing Ye, WenJing Xie, Yuechi Zhou, Yuyang Ding,

Zecheng Tang, Jikai Wang, Yixin Ji, Yue Wang, Pei Guo, Zechen Sun, Zikang Zhang, Juntao Li,

Pingfu Chao, Wenliang Chen, Guohong Fu, Guodong Zhou, Qiaoming Zhu, Min Zhang

**Soochow University** 

#### **Abstract**

Large Language Models (LLMs) have played an important role in many fields due to their powerful capabilities. However, their massive number of parameters leads to high deployment requirements and incurs significant inference costs, which impedes their practical applications. Training smaller models is an effective way to address this problem. Therefore, we introduce OpenBA-V2, a 3.4B model derived from multi-stage compression and continual pre-training from the original 15B OpenBA model. OpenBA-V2 utilizes more data, more flexible training objectives, and techniques such as layer pruning, neural pruning, and vocabulary pruning to achieve a compression rate of 77.3% with minimal performance loss. OpenBA-V2 demonstrates competitive performance compared to other open-source models of similar size, achieving results close to or on par with the 15B OpenBA model in downstream tasks such as common sense reasoning and Named Entity Recognition (NER). OpenBA-V2 illustrates that LLMs can be compressed into smaller ones with minimal performance loss by employing advanced training objectives and data strategies, which may help deploy LLMs in resource-limited scenarios.

## 1 Introduction

In recent years, Large Language Models (LLMs) have demonstrated powerful capabilities in natural language understanding and generation, leading to significant achievements in various tasks such as dialogue generation, code generation, text summarization, and machine translation (OpenAI, 2023; Touvron et al., 2023; Jiang et al., 2023; Bai et al., 2023; Li et al., 2023b). However, their extensive demand for computing resources makes them impractical in resource-limited scenarios, such as PCs or mobile phones (Thawakar et al., 2024). Furthermore, the high costs of inference and storage impede their widespread application across various industries (Bai et al., 2024).

To address these challenges, many researchers attempt to reduce the computational and storage requirements of LLMs by designing smaller models. These smaller models are usually obtained by training from scratch (Geng & Liu, 2023; Zhang et al., 2024; Li et al., 2023c; mic, 2024) or compressing larger models (Xia et al., 2024; Ma et al., 2023a). Some previous works (Li et al., 2023c; Bai et al., 2023) emphasize the importance of prioritizing data quality over quantity when training smaller models from scratch. They demonstrate that small models can potentially outperform

<sup>\*</sup> Equal Contribution.

<sup>†</sup> Corresponding author. ljt@suda.edu.cn

their larger counterparts with lower training costs. This insight offers a promising approach to training more powerful models with fewer resources. From another perspective, model compression, which includes pruning, distillation, and quantization, are presented as a method to strike a balance between efficiency and performance for existing LLMs. Pruning accelerates LLMs by removing non-essential parameters of the network with specialized hardware [\(Ma et al., 2023a;](#page-17-2) [Frantar &](#page-15-1) [Alistarh, 2023;](#page-15-1) [Sun et al., 2024\)](#page-17-3). Distillation enables the model to acquire knowledge rapidly from a teacher model by mimicking the teacher's behavior [\(Wu et al., 2023;](#page-18-3) [Hsieh et al., 2023\)](#page-15-2). Quantization can lower the costs of model storage and inference by converting the model to lower precision, more computationally efficient data types [\(Frantar et al., 2023;](#page-15-3) [Dettmers et al., 2024\)](#page-15-4).

To accommodate low-resource and low-cost requirements, we introduce OpenBA-V2, an encoderdecoder Transformer model with 3.4B parameters. OpenBA-V2 achieves a 77.3% compression ratio of OpenBA [\(Li et al., 2023b\)](#page-16-1), significantly lowering the resource requirements for deployment while maintaining high performance. OpenBA-V2 adopts a multi-stage compression strategy that employs layer pruning or neural pruning at each stage, followed by a period of fast and efficient recovery training to minimize performance loss due to model compression. After several compression stages, the model size has been reduced from 15B to 3.8B. Subsequently, we use 700B tokens to continually pre-train the compressed model with an optimized objective, further boosting training efficiency and enhancing the model's capabilities. Finally, we prune the model's vocabulary because of its redundancy, reducing the model size from 3.8B to 3.4B with almost no performance loss. In addition, we have compiled a more extensive and diverse dataset from various sources compared to OpenBA. These strategies have enabled OpenBA-V2 to achieve high performance with much fewer parameters. Through OpenBA-V2, we aim to demonstrate that smaller models can achieve comparable performance to larger models through better training objectives and data strategies, facilitating the deployment across various industries of LLMs. Our code and model weights are available at [https://github.com/OpenNLG/OpenBA-v2.](https://github.com/OpenNLG/OpenBA-v2)

