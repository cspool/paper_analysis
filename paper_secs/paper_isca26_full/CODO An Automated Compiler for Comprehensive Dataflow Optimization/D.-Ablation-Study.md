# D. Ablation Study

**Optimization Method Ablation.** To understand the impact of different optimization methods in CODO, we conduct an ablation study across five configurations (Opt1-Opt5), as

<span id="page-12-0"></span>TABLE VI: On-board comparison on GPT-2 model. TTFT measures the time to first token, the lower the better. Speed measures the decoding speed in token/s, the higher the better.

| [Input Len: | DFX             |           | Allo               |                 | StreamTensor |                    |                 | CODO      |                    |                 |           |                    |
|-------------|-----------------|-----------|--------------------|-----------------|--------------|--------------------|-----------------|-----------|--------------------|-----------------|-----------|--------------------|
| Output Len] | Latency<br>(ms) | TTFT (ms) | Speed<br>(token/s) | Latency<br>(ms) | TTFT (ms)    | Speed<br>(token/s) | Latency<br>(ms) | TTFT (ms) | Speed<br>(token/s) | Latency<br>(ms) | TTFT (ms) | Speed<br>(token/s) |
| [32:32]     | 350.00          | 177.20    | 185.19             | 238.32          | 81.50        | 204.05             | 194.99          | 34.59     | 199.51             | 158.64          | 20.40     | 231.48             |
| [64:64]     | 694.70          | 349.10    | 185.19             | 476.64          | 162.99       | 204.05             | 358.24          | 61.27     | 215.51             | 313.44          | 32.64     | 231.48             |
| [128:128]   | 1384.00         | 692.80    | 185.19             | 953.28          | 325.98       | 204.05             | 696.65          | 125.35    | 224.05             | 663.36          | 110.40    | 231.48             |

<span id="page-12-1"></span>TABLE VII: Five configurations of optimization methods.

| Optimization                         | Opt1 | Opt2 | Opt3 | Opt4 | Opt5 |
|--------------------------------------|------|------|------|------|------|
| Coarse-grained Violation Elimination | ×    | /    | /    | /    | 1    |
| Fine-grained Violation Elimination   | /    | X    | X    | 1    | ✓    |
| Efficient Data Communication         | X    | X    | 1    | 1    | ✓    |
| Automated Dataflow Scheduling        | X    | X    | X    | X    | ✓    |

<span id="page-12-2"></span>![](_page_12_Figure_4.jpeg)

Fig. 12: On-board execution time breakdown.

defined in Table VII. The performance speedups and resource utilization from synthesis results are detailed in Fig. 10. Starting with Opt1, we observe that enabling fine-grained optimizations in isolation yields negligible speedup. This is because unresolved coarse-grained violations invalidate dataflow optimization, leading to sequential execution. In contrast, Opt2 resolves these violations and enables basic ping-pong buffer-based dataflow execution, achieving initial performance gains ranging from  $2.5 \times$  to  $9.7 \times$ . Next, Opt 3 enables efficient data communication. For models with high data reuse potential, such as ResNet-18 and YOLO, the generation of line and window buffers significantly improves on-chip data reuse and communication efficiency, delivering higher speedups compared to Opt 2. Building on this, Opt 4 addresses fine-grained violations to enable efficient FIFO-based dataflow, boosting performance up to 105.8× Finally, Opt5 delivers the highest performance improvements by leveraging resource-aware parallelism exploration and inter-task optimization. Notably, for computation- and communication-intensive workloads like GPT-2, applying Opt2-Opt4 yields limited gains due to the extremely imbalanced dataflow, which severely hinders overall performance. Opt5 addresses this by enforcing resourceaware parallelism exploration and inter-task optimization, enabling a high percentage of efficient FIFO implementations

TABLE VIII: Percentage of FIFO usage.

<span id="page-12-3"></span>

| Application | Gesummv | Residual<br>Block | Multi-Head<br>Attention | MobileNet | ResNet-18 | GPT-2 |
|-------------|---------|-------------------|-------------------------|-----------|-----------|-------|
| Percentage  | 100%    | 100%              | 84%                     | 100%      | 100%      | 89%   |

with a balanced dataflow. This demonstrates that the superior performance of CODO results from the joint co-optimization of correctness, communication, and parallelism.

Resource-Performance Trade-off Evaluation. To evaluate CODO's ability to generate efficient designs under various resource budgets, we conducted a resource-performance trade-off experiment by adjusting the parallelism degree to simulate different resource budgets, as shown in Fig. 11. The results show that performance speedup increases nearly linearly with higher parallelism degrees, accompanied by a steady rise in DSP utilization. This indicates that even on resource-constrained FPGA boards, CODO can generate efficient dataflow accelerators by appropriately tuning parallelism.

On-board Execution Time Breakdown. Figure 12 shows a detailed breakdown of execution time for GPT-2 with different prefill lengths and DNNs with different input sizes. Overall, data transfer time remains low since CODO effectively utilizes HBM bandwidth. For GPT-2, the data transfer portion is relatively high at short prefill lengths but drops quickly as the sequence length grows. This is because computation in self-attention increases much faster than data movement, causing computation latency to dominate at larger prefill lengths. With efficient FIFO implementation and communication optimizations, CODO consistently delivers satisfying performance gains across different input lengths.

FIFO Percentage Quantification. To quantitatively evaluate the effectiveness of our approach, Table VIII reports the proportion of FIFOs used across benchmarks. Except for a few cases in attention and GPT-2, where detected optimization strategy conflicts trigger a fallback to ping-pong buffers, all other tasks achieve a 100% FIFO implementation. This demonstrates the strong scalability and effectiveness of CODO's dataflow violation elimination.

# D. Ablation Study

**Optimization Method Ablation.** To understand the impact of different optimization methods in CODO, we conduct an ablation study across five configurations (Opt1-Opt5), as

<span id="page-12-0"></span>TABLE VI: On-board comparison on GPT-2 model. TTFT measures the time to first token, the lower the better. Speed measures the decoding speed in token/s, the higher the better.

| [Input Len: | DFX             |           | Allo               |                 | StreamTensor |                    |                 | CODO      |                    |                 |           |                    |
|-------------|-----------------|-----------|--------------------|-----------------|--------------|--------------------|-----------------|-----------|--------------------|-----------------|-----------|--------------------|
| Output Len] | Latency<br>(ms) | TTFT (ms) | Speed<br>(token/s) | Latency<br>(ms) | TTFT (ms)    | Speed<br>(token/s) | Latency<br>(ms) | TTFT (ms) | Speed<br>(token/s) | Latency<br>(ms) | TTFT (ms) | Speed<br>(token/s) |
| [32:32]     | 350.00          | 177.20    | 185.19             | 238.32          | 81.50        | 204.05             | 194.99          | 34.59     | 199.51             | 158.64          | 20.40     | 231.48             |
| [64:64]     | 694.70          | 349.10    | 185.19             | 476.64          | 162.99       | 204.05             | 358.24          | 61.27     | 215.51             | 313.44          | 32.64     | 231.48             |
| [128:128]   | 1384.00         | 692.80    | 185.19             | 953.28          | 325.98       | 204.05             | 696.65          | 125.35    | 224.05             | 663.36          | 110.40    | 231.48             |

<span id="page-12-1"></span>TABLE VII: Five configurations of optimization methods.

| Optimization                         | Opt1 | Opt2 | Opt3 | Opt4 | Opt5 |
|--------------------------------------|------|------|------|------|------|
| Coarse-grained Violation Elimination | ×    | /    | /    | /    | 1    |
| Fine-grained Violation Elimination   | /    | X    | X    | 1    | ✓    |
| Efficient Data Communication         | X    | X    | 1    | 1    | ✓    |
| Automated Dataflow Scheduling        | X    | X    | X    | X    | ✓    |

<span id="page-12-2"></span>![](_page_12_Figure_4.jpeg)

Fig. 12: On-board execution time breakdown.

defined in Table VII. The performance speedups and resource utilization from synthesis results are detailed in Fig. 10. Starting with Opt1, we observe that enabling fine-grained optimizations in isolation yields negligible speedup. This is because unresolved coarse-grained violations invalidate dataflow optimization, leading to sequential execution. In contrast, Opt2 resolves these violations and enables basic ping-pong buffer-based dataflow execution, achieving initial performance gains ranging from  $2.5 \times$  to  $9.7 \times$ . Next, Opt 3 enables efficient data communication. For models with high data reuse potential, such as ResNet-18 and YOLO, the generation of line and window buffers significantly improves on-chip data reuse and communication efficiency, delivering higher speedups compared to Opt 2. Building on this, Opt 4 addresses fine-grained violations to enable efficient FIFO-based dataflow, boosting performance up to 105.8× Finally, Opt5 delivers the highest performance improvements by leveraging resource-aware parallelism exploration and inter-task optimization. Notably, for computation- and communication-intensive workloads like GPT-2, applying Opt2-Opt4 yields limited gains due to the extremely imbalanced dataflow, which severely hinders overall performance. Opt5 addresses this by enforcing resourceaware parallelism exploration and inter-task optimization, enabling a high percentage of efficient FIFO implementations

TABLE VIII: Percentage of FIFO usage.

<span id="page-12-3"></span>

| Application | Gesummv | Residual<br>Block | Multi-Head<br>Attention | MobileNet | ResNet-18 | GPT-2 |
|-------------|---------|-------------------|-------------------------|-----------|-----------|-------|
| Percentage  | 100%    | 100%              | 84%                     | 100%      | 100%      | 89%   |

with a balanced dataflow. This demonstrates that the superior performance of CODO results from the joint co-optimization of correctness, communication, and parallelism.

Resource-Performance Trade-off Evaluation. To evaluate CODO's ability to generate efficient designs under various resource budgets, we conducted a resource-performance trade-off experiment by adjusting the parallelism degree to simulate different resource budgets, as shown in Fig. 11. The results show that performance speedup increases nearly linearly with higher parallelism degrees, accompanied by a steady rise in DSP utilization. This indicates that even on resource-constrained FPGA boards, CODO can generate efficient dataflow accelerators by appropriately tuning parallelism.

On-board Execution Time Breakdown. Figure 12 shows a detailed breakdown of execution time for GPT-2 with different prefill lengths and DNNs with different input sizes. Overall, data transfer time remains low since CODO effectively utilizes HBM bandwidth. For GPT-2, the data transfer portion is relatively high at short prefill lengths but drops quickly as the sequence length grows. This is because computation in self-attention increases much faster than data movement, causing computation latency to dominate at larger prefill lengths. With efficient FIFO implementation and communication optimizations, CODO consistently delivers satisfying performance gains across different input lengths.

FIFO Percentage Quantification. To quantitatively evaluate the effectiveness of our approach, Table VIII reports the proportion of FIFOs used across benchmarks. Except for a few cases in attention and GPT-2, where detected optimization strategy conflicts trigger a fallback to ping-pong buffers, all other tasks achieve a 100% FIFO implementation. This demonstrates the strong scalability and effectiveness of CODO's dataflow violation elimination.

