# B. Expert Parallelism

MPipeMoE distributes experts across different GPUs using the expert parallelism while running the remaining parts of the MoE model in data-parallel scheme. NCCL All-to-All collective operator is adopted to dispatch and collect tokens among GPUs.

#### C. Usability

It is easy for users to benefit from optimizations introduced by MPipeMoE using Python API, including adaptive pipeline parallelism and memory reusing, etc. As shown in the following code snippet, adding parameters such as *pipeline* and *memory\_reuse* can bring the end-to-end speedup.

```
import pmoe
moe_layer = pmoe.MoELayer(d_model=1024,
```

<sup>&</sup>lt;sup>1</sup>https://github.com/whuzhangzheng/MPipeMoE

TABLE III SPECIFICATIONS OF MOE LAYERS

| Model Name  | $d_{model}$ | $d_{hidden}$ | #experts |
|-------------|-------------|--------------|----------|
| MoE-GPT3-S  | 768         | 3072         | 64       |
| MoE-GPT3-XL | 2048        | 8192         | 64       |
| MoE-BERT-L  | 1024        | 4096         | 64       |

#### V. EVALUATION

#### A. Experimental Setup

- 1) Physical cluster: Experimental evaluation is performed on a physical cluster consisting of 8 NVIDIA DGX A100 servers. Each node is equipped with 8 NVIDIA A100 SXM 40GB GPUs and 200 Gbps HDR InfiniBand, backed by 96 × 2nd-generation AMD EPYC CPU cores and 1.9 TiB memory. GPUs are connected by the 3-rd generation NVLink and NVSwitch within each machine. GPUs across different machines are connected through a 1,600 Gbps InfiniBand network with adaptive routing.
- 2) Models and configurations: The significant discrepancy of the MoE layer between different models lies in the size of experts, which is determined by M and H, and the number of tokens that is denoted as B. Here, we aim to validate the efficiency of the proposed methods on different expert sizes as well as batch sizes. We configure the different expert sizes of FFN from BERT [7] and GPT-3 [10] as shown in Table III, in which  $d_{model}$  refers to the dimension of token embedding and  $d_{hidden}$  refers to the hidden dimension of FFN layer of models, respectively. Note that this paper focuses on the MoE structure and related performance. We create a dummy dataset by generating random tokens as input to different models. For all experiments, we adopt Adam [30] as the optimizer. We evaluate the efficiency of MPipeMoE in terms of the average training time and the peak memory footprint.

