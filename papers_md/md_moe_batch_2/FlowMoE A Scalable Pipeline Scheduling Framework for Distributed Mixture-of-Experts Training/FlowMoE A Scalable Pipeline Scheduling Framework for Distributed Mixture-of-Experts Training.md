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

# **FlowMoE — Key Ideas**

![](_page_1_Figure_1.jpeg)

![](_page_1_Figure_2.jpeg)

- **Unified pipeline scheduling strategy:** Schedules MHA, gating, expert, and A2A together.
- **Priority scheduling mechanism for heterogeneous communication tasks:** Cut all-reduce chunks and execute them in all-to-all task gaps.
- **Lightweight adaptive optimizer and system integration:** Tiny Bayesian optimizer for automatic tuning. Deploying FlowMoE to the PyTorch engine.

# **Experimental Results**

### **Experimental Settings**

- 675 customized MoE layers, 4 real-world MoE models
- Clusters: RTX3090 (16 GPUs),RTX2080Ti (8 GPUs).

![](_page_2_Figure_4.jpeg)

![](_page_2_Figure_5.jpeg)

# **System Implementation & Conclusion**

### **System Implementation**

![](_page_3_Figure_2.jpeg)

- **Framework:** Implemented on PyTorch , leveraging Tutel for optimized communication.
- **Compatibility:** Supports multiple optimization frameworks and communication stacks.

### **Conclusions**

- The unified scheduling across all major MoE-related tasks.
- Enabling the optimal coexistence of heterogeneous communication tasks.
- Substantially advancing distributed pipeline training.
- Open-source: github.com/ZJU-CNLAB/FlowMoE

![](_page_3_Picture_10.jpeg)