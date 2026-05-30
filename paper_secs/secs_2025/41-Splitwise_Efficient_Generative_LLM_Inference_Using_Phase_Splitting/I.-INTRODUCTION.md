# I. INTRODUCTION

Recent advancements in generative large language models (LLMs) have significantly improved their response quality and accuracy [18], [71]. These trends have led to the widespread adoption of LLMs across various domains [6], [21]. Most modern LLMs are built using the transformer architecture [77], [78] and exhibit similar characteristics [63]. Transformer model sizes have grown steadily, from the early BERT models [36] having 340 million parameters, to GPT-3 [28] with a staggering 175 billion parameters, and GPT-4 rumored to have even more.

LLMs typically run on expensive and power-hungry GPUs [16]. The sudden and large-scale deployment of LLMs has led to a worldwide GPU capacity crunch [14]. The computational demand for LLM inference far exceeds that of training due to the vast number of applications leveraging LLMs. Furthermore, since training LLMs requires expensive and dedicated supercomputers [56], [60], a large number of inferences are necessary to amortize the high training costs. LLM inference jobs, although orders of magnitude smaller than training, are still expensive given the compute involved.

|                      | A100      | H100     | Ratio |
|----------------------|-----------|----------|-------|
| TFLOPs               | 19.5      | 66.9     | 3.43× |
| HBM capacity         | 80GB      | 80GB     | 1.00× |
| HBM bandwidth        | 2039GBps  | 3352GBps | 1.64× |
| Power                | 400W      | 700W     | 1.75× |
| NVLink               | 50Gbps    | 100Gbps  | 2.00× |
| Infiniband           | 200GBps   | 400GBps  | 2.00× |
| Cost per machine [5] | \$17.6/hr | \$38/hr  | 2.16× |

TABLE I: NVIDIA A100 vs. H100 specifications.

Generative LLM inference for a single request consists of several forward passes through the model, since the output tokens are generated one by one. This inherently has two contrasting phases of computation. First, the *prompt computation phase*, in which all the input prompt tokens run through the forward pass of the model in parallel to generate the first output token. This phase tends to be computationally intensive and requires the high FLOPs (floating point operations per second) of the latest GPUs today. Second, the *token generation phase*, in which subsequent output tokens are generated sequentially based on the forward pass of the last token and all the cached context from previous tokens in the sequence. Given the lack of compute parallelism, this phase tends to be more memory bandwidth and capacity bound, despite state-of-the-art batching. Running both phases on the same machine often leads to inconsistent end-to-end latencies due to the arbitrary batching of prompt and token phases. Due to these challenges, services need to over-provision expensive GPUs to meet tight inference service level objectives (SLOs) for interactive applications. At the same time, cloud service providers (CSPs) are having to build a lot of new datacenters to meet the GPU demand, and are running into a power wall [19].

The industry continues to release new computationally powerful GPUs, each much more power hungry and expensive than the last. However, as shown in Table I, the high-bandwidth memory (HBM) capacity and bandwidth on these GPUs has not scaled at the same rate recently. The latest NVIDIA H100 GPUs have 3.43× more compute and 1.75× more power compared to their predecessor A100 GPUs. However, their memory bandwidth only grew by 1.6×, with no increase in memory capacity.

Our work. Given the distinct properties of prompt computation and token generation phases, we propose splitting the inference

<sup>1</sup>Work partly done as an intern at Microsoft.

request and running them on separate machines. Doing so allows us to separately manage hardware resources for each phase, thereby increasing the GPU utilization and the overall efficiency of the system. It also enables using different, better-suited hardware for each phase. To realize such a setup, the cached context from the prompt computation needs to be communicated over from the prompt processing machine to the token generation machine at low latency. We implement these transfers in an optimized manner over the back-end Infiniband interconnects available in datacenters today, allowing us to increase efficiency without any perceived performance loss.

With Splitwise, we design clusters optimized for cost, throughput, and power, using production traces of LLM inference requests [4]. Given the diverging memory and compute scaling rates across GPU generations, we also evaluate different GPUs and power caps for the different inference phases. This allows us to target better performance per dollar (Perf/\$) for users, and better performance per watt (Perf/W) for CSPs. Additionally, users can target older GPUs, which are likely more readily available to them.

We show that Splitwise-based LLM inference clusters can achieve  $1.4 \times$  higher throughput at 20% lower cost than existing clusters. Alternatively, they can deliver  $2.35 \times$  more throughput with the same cost and power budgets.

Summary. We make the following contributions:

- An extensive characterization of the differences in the execution and utilization patterns of the prompt and token generation phases in LLM inference on the NVIDIA A100 and H100 GPUs using production traces.
- Splitwise, our technique for optimized utilization of available hardware, which splits the prompt computation and token generation phases onto separate machines.
- 3) A design exploration of homogeneous and heterogeneous cluster deployments with Splitwise to optimize the overall cost, request throughput, and provisioned power.
- 4) An evaluation of the systems designed with Splitwise using production traces.

