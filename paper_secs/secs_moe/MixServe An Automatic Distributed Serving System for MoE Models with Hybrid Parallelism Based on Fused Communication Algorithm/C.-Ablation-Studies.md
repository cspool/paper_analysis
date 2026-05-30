# C. Ablation Studies

To better understand the impact of different components in MixServe, we conduct ablation studies by systematically removing or modifying key features.

1) Trade-off between DP and EP: As §III-B3 describes, MixServe optimizes  $d_{\rm DP}$  and  $d_{\rm EP}$  by evaluating the modeled communication and computation costs across feasible TP/DP/EP tuples. We study three representative settings: (1)  $d_{\rm DP} = d_{\rm EP}$  (TP=8 + DP=4, TP=8 + EP=4), (2)  $d_{\rm DP} > d_{\rm EP}$ 

![](_page_9_Figure_7.jpeg)

Fig. 12: Impact of overlapping communication based on fused AR-A2A communication algorithm in MixServe. We evaluate that on the Ascend 910B cluster with DeepSeek-R1. (a) Gantt chart of Sync and Async communication. (b) Performance comparison of Sync and Async communication.

(TP=4 + DP=8, TP=8 + EP=4), and (3)  $d_{\rm DP} < d_{\rm EP}$  (TP=8 + DP=4, TP=4 + EP=8).

Fig. 11 summarizes the ablation results. On Ascend 910B, the balanced case attains the best latency/throughput for both DeepSeek-R1 and Qwen3 (e.g., 383.14ms TTFT and 150.08 tokens/s throughput for Qwen3 when  $d_{DP} = d_{EP}$ ), while skewing towards larger DP or larger EP degrades performance. However, on Nvidia H20, a different ordering holds:  $d_{DP} < d_{EP}$  yields the lowest TTFT (e.g., 228.99ms for Qwen3) and the highest throughput (40.00 tokens/s). These observations align with our analytical trade-off model-balancing DP and EP minimizes the dominant communication term—so the partitioner automatically selects this configuration under both high-bandwidth NVLink (H20) and RoCE/HCCS (910B) environments. When cluster bandwidth or node count changes, MixServe re-evaluates the cost model and picks the best feasible tuple, ensuring the serving system adapts its parallel strategy to the available network and compute resources.

2) Impact of Overlapping Communication: As §III-D describes, MixServe employs a fused AR-A2A communication algorithm to optimize the communication process. We evaluate the impact of this optimization on performance by whether asynchronous or synchronous communication is used.

Fig. 12a shows the Gantt chart of synchronous and asynchronous communication, where the asynchronous communication allows for overlapping of intra-node and inter-node communication. Specifically, the fused AG-Dispatch communication algorithm overlaps inter-node Dispatch communication with intra-node AG communication, while the fused RS-Combine algorithm overlaps inter-node Combine communication, intra-node RS communication, and the computation of top-*k* weights. The Gantt chart indicates that the asynchronous fused AR-A2A demonstrates a performance improvement compared to the total

latencies of the synchronous operators, which is approximately slightly greater than inter-node communication overhead.

Fig. 12b shows the performance comparison of synchronous and asynchronous communication. The results indicate that the asynchronous communication significantly reduces the overall latency, leading to improved TTFT and ITL. The throughput also increases due to the reduced communication overhead. This ablation study demonstrates the effectiveness of overlapping communication in enhancing performance.

#### V. RELATED WORK

#### A. Distributed MoE

In the early stages of research, various distributed methods facilitated parallel training of MoE models to improve throughput and efficiency. GShard [10] pioneering the use of all-to-all communication for large-scale sparsity. Subsequent frameworks like DeepSpeed-MoE [11] and Tutel [8] refined this via hybrid data/expert parallelism and fused kernels to enhance memory efficiency and multi-node scalability. To further mitigate communication bottlenecks, SmartMoE [12] introduced dynamic strategy selection, while Lina [13] optimized interleaved all-to-all operators. More recently, the field has moved toward high-dimensional parallelism; notably, MoE Parallel Folding [4] utilizes Megatron-Core [14] to integrate TP, EP, DP, PP, and context parallelism (CP) into a unified 5D scheme for heterogeneous clusters.

Our work leverages numerous methods and concepts from the training of distributed MoE models, focusing on their application in distributed MoE model serving.

## B. Distributed LLM Serving

Distributed serving systems prioritize maximizing throughput and minimizing latency for online inference. Early optimizations focused on request scheduling: Orca [15] pioneered iteration-level scheduling and selective batching, while Llumnix [16] introduced dynamic resource allocation based on workload characteristics. Regarding parallelism, Alpa [17] and AlpaServe [18] explored the synergy between intra/inter-operator parallelism and model multiplexing. To further optimize the distinct phases of inference, DistServe [19] proposed prefill/decode (P/D) disaggregation, and Sarathi-Serve [20] utilized chunked-prefills with stall-free scheduling to balance throughput-latency trade-offs. Most recently, MegaScale-Infer [21] extended disaggregation to Attention and MoE blocks, leveraging pingpong pipeline parallelism to hide communication overhead and maximize GPU utilization.

Our work focuses on parallel strategies and communication optimization, and can be effectively incorporated with various optimization methods of existing LLM serving systems, such as request scheduling, P/D disaggregation, etc.

#### VI. Conclusion

We introduce MixServe, a novel automatic distributed serving system that for efficient deployment of MoE models by hybrid TP-EP based on fused AR-A2A communication algorithm. MixServe automatically selects the optimal parallel strategy

based on model parameters and network configurations. It employs a hybrid TP-EP partitioner to optimize communication overhead and introduces a fused AR-A2A communication algorithm to enhance TTFT, ITL and throughput. MixServe's design is guided by theoretical analysis and practical considerations, ensuring efficient resource utilization and low latency. Our evaluation on mainstream MoE models such as DeepSeek-R1 and Qwen3 demonstrates that MixServe achieves significant performance improvements in MoE model serving, making it a valuable tool for deploying large-scale LLMs. We hope MixServe will contribute to the efficient deployment of MoE models in real-world applications.

#### REFERENCES

- [1] DeepSeek-AI *et al.*, "Deepseek-v3 technical report," 2025. [Online]. Available: https://arxiv.org/abs/2412.19437
- [2] —, "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning," 2025. [Online]. Available: https://arxiv.org/abs/2501.12948
- [3] A. Yang et al., "Qwen3 technical report," 2025. [Online]. Available: https://arxiv.org/abs/2505.09388
- [4] D. Liu et al., "Moe parallel folding: Heterogeneous parallelism mappings for efficient large-scale moe model training with megatron core," 2025. [Online]. Available: https://arxiv.org/abs/2504.14960
- [5] M. Shoeybi et al., "Megatron-lm: Training multi-billion parameter language models using model parallelism," 2020. [Online]. Available: https://arxiv.org/abs/1909.08053
- [6] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," 2022. [Online]. Available: https://arxiv.org/abs/2101.03961
- [7] W. Kwon et al., "Efficient memory management for large language model serving with pagedattention," in Proceedings of the 29th Symposium on Operating Systems Principles, ser. SOSP '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 611–626. [Online]. Available: https://doi.org/10.1145/3600006.3613165
- [8] C. Hwang et al., "Tutel: Adaptive mixture-of-experts at scale," 2023.[Online]. Available: https://arxiv.org/abs/2206.03382
- [9] OpenChat Team, "Openchat sharegpt v3," https://huggingface.co/datasets/ openchat/openchat\_sharegpt\_v3, 2023, shareGPT dataset for training OpenChat V3 series. Licensed under MIT. Accessed: 2025-08-20.
- [10] D. Lepikhin et al., "Gshard: Scaling giant models with conditional computation and automatic sharding," 2020. [Online]. Available: https://arxiv.org/abs/2006.16668
- [11] S. Rajbhandari et al., "Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale," 2022. [Online]. Available: https://arxiv.org/abs/2201.05596
- [12] M. Zhai et al., "SmartMoE: Efficiently training Sparsely-Activated models through combining offline and online parallelization," in 2023 USENIX Annual Technical Conference (USENIX ATC 23). Boston, MA: USENIX Association, Jul. 2023, pp. 961–975. [Online]. Available: https://www.usenix.org/conference/atc23/presentation/zhai
- [13] J. Li et al., "Accelerating distributed MoE training and inference with lina," in 2023 USENIX Annual Technical Conference (USENIX ATC 23). Boston, MA: USENIX Association, Jul. 2023, pp. 945–959. [Online]. Available: https://www.usenix.org/conference/atc23/ presentation/li-jiamin
- [14] NVIDIA Corporation, "Megatron-lm: Ongoing research training transformer models at scale," https://github.com/NVIDIA/Megatron-LM, 2024, accessed: 2025-08-20.
- [15] G.-I. Yu et al., "Orca: A distributed serving system for Transformer-Based generative models," in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). Carlsbad, CA: USENIX Association, Jul. 2022, pp. 521–538. [Online]. Available: https://www.usenix.org/conference/osdi22/presentation/yu
- [16] B. Sun et al., "Llumnix: Dynamic scheduling for large language model serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 173–191. [Online]. Available: https://www.usenix.org/conference/osdi24/presentation/sun-biao

- [17] L. Zheng et al., "Alpa: Automating inter- and Intra-Operator parallelism for distributed deep learning," in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). Carlsbad, CA: USENIX Association, Jul. 2022, pp. 559–578. [Online]. Available: https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin
- [18] Z. Li et al., "AlpaServe: Statistical multiplexing with model parallelism for deep learning serving," in 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). Boston, MA: USENIX Association, Jul. 2023, pp. 663–679. [Online]. Available: https://www.usenix.org/conference/osdi23/presentation/li-zhouhan
- [19] Y. Zhong et al., "DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 193–210. [Online]. Available: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- [20] A. Agrawal et al., "Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 117–134. [Online]. Available: https://www.usenix.org/conference/osdi24/presentation/agrawal
- [21] R. Zhu *et al.*, "Megascale-infer: Serving mixture-of-experts at scale with disaggregated expert parallelism," 2025. [Online]. Available: https://arxiv.org/abs/2504.02263