#### FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

Yunqi Gao<sup>1</sup>, Bing Hu<sup>1,\*</sup>, Mahdi Boloursaz Mashhadi<sup>2</sup>, A-Long Jin<sup>3</sup>, Yanfeng Zhang<sup>4</sup>, Pei Xiao<sup>2</sup>, Rahim Tafazolli<sup>2</sup>, Mérouane Debbah<sup>5</sup> 1-Zhejiang University · 2-University of Surrey · 3-Xi'an Jiaotong-Liverpool University · 4-Northeastern University · 5-Khalifa University

#### **Motivation**

- MoE scales LLM without increasing computational costs.
- Existing works pipeline only expert + all-to-all, ignoring MHA, gating, and all-reduce.
- Ignored tasks = 30-40% of iteration time  $\rightarrow$  huge inefficiency.

![](_page_0_Figure_6.jpeg)

Fig. 1: Training MoE model with expert parallelism

![](_page_0_Figure_8.jpeg)

Fig. 2: One iteration time breakdown per model

![](_page_0_Picture_10.jpeg)

