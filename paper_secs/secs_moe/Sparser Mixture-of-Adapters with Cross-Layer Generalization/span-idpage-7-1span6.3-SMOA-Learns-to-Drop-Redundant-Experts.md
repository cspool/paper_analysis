# <span id="page-7-1"></span>6.3 SMOA Learns to Drop Redundant Experts

To address the issue of expert redundancy, as outlined in Section 3, SMoA introduces a shared adapter pool that enables efficient selection and reuse of adapters across multiple layers and target modules. This mechanism dramatically reduces the number of active experts required while maintaining strong task performance. As shown in Table 4, SMoA lowers expert utilization in Phi-2 to just 12.73%, a major improvement over the 100% utilization seen in baseline methods. This efficiency is achieved by allowing Phi-2 to adapt three target modules simultaneously, with shared adapters flexibly applied across these modules—whereas other pre-trained models are restricted to adapting just one (Appendix A). This modular sharing represents a pivotal advancement, enabling SMoA to optimize performance with fewer adapters and paving the way for more scalable, adaptable systems.

<span id="page-7-0"></span>Table 4: Average expert utilization rate (%) across all datasets. The baseline adopts layer-specific experts and requires to use all of them (100% utilization). In contrast, SMOA sparsely activates experts across layers.

|                       | Total<br>Experts | Avg.<br>Utilization         |
|-----------------------|------------------|-----------------------------|
| Phi-3 <sub>3.8B</sub> | 256              | <b>58.75%</b> ±0.46         |
| Phi- $2_{2.7B}$       | 768              | $12.73\%_{\pm0.25}$         |
| $Gemma_{2B}$          | 140              | <b>60.39%</b> $_{\pm 0.92}$ |
| $\mathrm{OLMo}_{1B}$  | 128              | <b>76.34%</b> $_{\pm 0.70}$ |

#### 6.4 Sensitivity Analysis and Ablation Study

Extreme sparsity: High performance with minimal activated experts. Our sensitivity analysis of  $n_l$  demonstrates that activating just 2 experts per layer yields performance nearly equivalent to using 8 experts (Table 7). This underscores SMoA's remarkable efficiency, as it maintains robust performance while significantly reducing the number of active experts, highlighting its capacity for sparse yet effective expert utilization.

Expert-redundancy regularization optimizes expert specialization without sacrificing per-

formance. The expert-redundancy regularization (Section [5.1\)](#page-4-0), promotes expert specialization, as detailed in Section [6.2,](#page-6-0) while maintaining or even improving overall model performance as shown in Appendix [E.](#page-13-3) This regularization strikes a crucial balance, fostering diversity among experts without compromising effectiveness.

