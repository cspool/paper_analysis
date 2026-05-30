# Yunji Chen<sup>∗</sup>

State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences Beijing, China University of Chinese Academy of Sciences Beijing, China cyj@ict.ac.cn

### Abstract

The rapid advancement of Large Language Models (LLMs) has established language as a core general-purpose cognitive substrate, driving the demand for specialized Language Processing Units (LPUs) tailored for LLM inference. To overcome the growing energy consumption of LLM inference systems, this paper proposes a Hardwired-Neurons Language Processing Unit (HNLPU), which physically hardwires LLM weight parameters into the computational fabric, achieving several orders of magnitude computational efficiency improvement by extreme specialization. However, a significant challenge

<sup>∗</sup>Corresponding Author

![](_page_1_Picture_26.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

ASPLOS '26, Pittsburgh, PA, USA. © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 <https://doi.org/10.1145/3779212.3790169>

still lies in the scale of modern LLMs. A straightforward hardwiring of gpt-oss 120 B would require fabricating photomask sets valued at over 6 billion dollars, rendering this straightforward solution economically impractical.

Addressing this challenge, we propose the novel Metal-Embedding methodology. Instead of embedding weights in a 2D grid of silicon device cells, Metal-Embedding embeds weight parameters into the 3D topology of metal wires. This brings two benefits: (1) a 15× increase in density, and (2) 60 out of 70 photomask layers are homogeneous across chips, including all EUV photomasks. In total, Metal-Embedding reduced the photomask cost by 112×, bringing the Non-Recurring Engineering (NRE) cost of HNLPU into an economically viable range. Experimental results show that HNLPU achieved 249,960 tokens/s (5,555×/85× that of GPU/WSE), 36 tokens/J (1,047×/283× that of GPU/WSE), 13,232 mm<sup>2</sup> total die area, \$ 59.46 M–123.5 M estimated NRE at 5 nm technology. Analysis shows that HNLPU achieved 41.7–80.4× improvement in cost-effectiveness and 357× reduction in carbon footprint compared to OpenAI-scale H100 clusters, under an annual weight updating assumption.

CCS Concepts: • Computer systems organization → Neural networks; • Hardware → Hardware accelerators.

Keywords: Large Language Models, Language Processing Unit, Hardwired-Neurons, Metal-Embedding, Sustainable AI

### ACM Reference Format:

Yang Liu, Yi Chen, Yongwei Zhao, Yifan Hao, Zifu Zheng, Weihao Kong, Zhangmai Li, Dongchen Jiang, Ruiyang Xia, Zhihong Ma, Zisheng Liu, Zhaoyong Wan, Yunqi Lu, Ximing Liu, Hongrui Guo, Zhihao Yang, Zhe Wang, Tianrui Ma, Mo Zou, Rui Zhang, Ling Li, Xing Hu, Zidong Du, Zhiwei Xu, Qi Guo, Tianshi Chen, and Yunji Chen. 2026. Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 21–26, 2026, Pittsburgh, PA, USA. ACM, New York, NY, USA, [20](#page-19-0) pages. <https://doi.org/10.1145/3779212.3790169>

