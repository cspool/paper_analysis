# <span id="page-6-0"></span>4.2 Search Space and Performance Model

Given a hardware configuration  $\mathcal{H}$ , a model configuration  $\mathcal{M}$ , and a workload configuration  $\mathcal{W}$ , we search for the optimal policy  $\mathcal{P}$  that minimizes per-layer latency  $T(\mathcal{M}, \mathcal{H}, \mathcal{W}, \mathcal{P})$  for the pipeline schedule in §4.1, without violating the CPU and GPU memory constraints, in order to reach the optimal balance point (Eq. (11)). Compared with FlexGen, we exclude disk-related variables from the search space and add two binaries to indicate whether to perform attention or MoE FFN on GPU.

<span id="page-7-1"></span>**Table 1.** Notations for the Performance Model Configuration

| Notation Description                               |                                       |  |  |  |  |  |
|----------------------------------------------------|---------------------------------------|--|--|--|--|--|
| Hardware Configurations, ${\cal H}$                |                                       |  |  |  |  |  |
| $m_q, m_c$                                         | GPU, CPU memory                       |  |  |  |  |  |
| $b_g, b_c, b_{cg}$                                 | GPU, CPU, CPU-GPU bandwidth           |  |  |  |  |  |
| $p_g, p_c$ GPU, CPU FLOPS                          |                                       |  |  |  |  |  |
| Model Configurations, $\mathcal{M}$                |                                       |  |  |  |  |  |
| l                                                  | Number of layers                      |  |  |  |  |  |
| $h_1, h_2$                                         | Model, Intermediate hidden dimensions |  |  |  |  |  |
| $n_q, n_{kv}$                                      | Query, Key/Value heads in attention   |  |  |  |  |  |
| $n_e, k$                                           | Number of experts, Top-k routing      |  |  |  |  |  |
| dt Data type (e.g., float32)                       |                                       |  |  |  |  |  |
| Workload Configurations, W                         |                                       |  |  |  |  |  |
| s Average Prompt Length                            |                                       |  |  |  |  |  |
| n Generation Length                                |                                       |  |  |  |  |  |
| Policy, $\mathcal P$                               |                                       |  |  |  |  |  |
| Ν, μ                                               | Batch, Micro Batch Size               |  |  |  |  |  |
| $F_g, A_g$                                         | GPU Attention/MoE FFN Indicator       |  |  |  |  |  |
| $r_w, r_c$ Ratio of Weights/KV Cache Stored on GPU |                                       |  |  |  |  |  |

The search space (Tab. 1) covers 2 integer values: the microbatch size ( $\mu$ ) and batch size (N), 2 binary indicators  $A_g$  to indicate whether to perform the attention on GPU and  $F_g$  to indicate whether to perform the MoE FFN on GPU. When  $F_g=1$ , we also need to decide the percent of weights  $r_w$  that can be statically stored on GPU and the percent of weights  $1-r_w$  that need to be transferred to GPU. Similarly, for  $A_g=1$ , we need to decide  $r_c$ . The generated policy will be a 6-tuple (N,  $\mu$ ,  $A_g$ ,  $F_g$ ,  $r_w$ ,  $r_c$ ). For our major setting, we always get  $A_g=0$  and  $F_g=1$ . However, we discuss in §6.3 different policies for various hardware settings. Notably, CGOPIPE is primarily designed for  $A_g=0$  and when  $A_g=1$ , MoE-LIGHTNING adopt  $S_4$ .

We then build the performance model based on Eq. (7) and Eq. (8) in HRM to estimate per-layer decode latency T by:

$$T(\mathcal{M}, \mathcal{H}, \mathcal{W}, \mathcal{P}) = \max(comm^{cpu\_to\_gpu}, T_{cpu}, T_{qpu})$$
 (12)

where  $comm^{cpu\_to\_gpu}$  can be computed as the number of bytes needed to be transferred from CPU to GPU for a layer's computation divided by the CPU to GPU memory bandwidth  $b_{cg}$ . Here, for simplicity, we only consider the attention computation and the MoE FFN computation in a transformer block, and therefore we have:

$$T_{gpu} = T_{attn}^g + T_{ffn}^g, T_{cpu} = T_{attn}^c + T_{ffn}^c$$
 (13)

To estimate the time to perform a computation x on GPU or CPU, we can use  $T_x = \max(comm_x, comp_x)$  according to Eq. (8) in HRM, resulting in:

$$T_{ffn}^g = \max(comm_{ffn}^g, comp_{ffn}^g)$$
 (14)

and similarly for  $T_{attn}^g$ ,  $T_{attn}^c$  and  $T_{ffn}^c$ .

For a given computation x, we can calculate their theoretical FLOPS and data transfer based on  $\mathcal{M}$  and then we have  $comm_x^g = bytes_x/b_g$  and  $comp_x^g = flops_x/p_g$  (same for CPU). While there are discrepancies between the theoretical performance estimation and the kernel's real performance,

such modeling can provide a reasonable estimation of the relative effectiveness of any two policies. In this paper, all the evaluation results of MoE-LIGHTNING follow policies generated by a performance model with theoretically calculated computation flops and bytes with profiled peak performance and memory bandwidth for the hardware.

