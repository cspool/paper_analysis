# CCS Concepts: • Computing methodologies $\rightarrow$ Machine learning.

Keywords: LLM Inference, Scheduling Optimization

#### **ACM Reference Format:**

Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, Junyeol Lee, Du-seong Chang, Jiwon Seo. 2024. ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24), April 27-May 1, 2024, La Jolla, CA, USA. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3620665.3640383

 $^*$ Corresponding author and principal investigator

![](_page_0_Picture_10.jpeg)

This work is licensed under a Creative Commons Attribution International 4.0 License.

ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0385-0/24/04.

https://doi.org/10.1145/3620665.3640383

#### 1 Introduction

Large language models (LLMs) have significantly advanced the field of natural language processing (NLP) and enabled a wide range of NLP applications. However, their high computational costs make it challenging to run LLMs efficiently, limiting their full potential. For example, generating a single token in LLMs can require hundreds of billions of FLOPs, demonstrating the need for efficient execution.

Compared to other neural networks, LLM inference is challenging due to their large size and irregular executions. LLMs can have hundreds of billions of parameters, requiring model parallel execution on multiple GPUs. The autoregressive nature of LLM inference also complicates its execution, as each model execution generates a single token, which is fed back to generate the next one, requiring multiple iterations for completion. However, when executing for a batch of data, each input may need a different number of iterations, leading to much reduced batch sizes in later iterations. This workload variability makes it hard to maintain high throughput.

To mitigate this problem, ORCA proposed *iteration-level* scheduling that adds a new input data into a running batch to replace a completed one [47]. Although this ensures consistent batch sizes throughout execution, it does not consider that the cost of input encoding is orders of magnitude higher than that of output decoding, potentially causing large pipeline bubbles. Moreover, executing input encoding at any decoding iteration may result in highly variable and thus uncontrollable latency.

We suppose that NLP applications have diverse requirements for inference latency, with some needing immediate responses while others can tolerate longer delays. To achieve efficient inference serving under a given latency bound, it is essential to optimize resource utilization and computation throughput. However, the irregular workload between inputs makes it difficult to efficiently run LLM inference.

This paper introduces ExeGPT, a system designed for constraint-aware LLM inference. ExeGPT finds and runs with an optimal execution schedule that maximizes the inference throughput while satisfying a given latency constraint. It leverages the distribution of sequence lengths to efficiently allocate resources between GPUs and optimize the parallel configuration. The system provides two novel scheduling strategies with four control variables, such as

![](_page_1_Figure_2.jpeg)

Figure 1. Illustration of a Transformer decoder (left) and inference timelines of three LLM inference systems (right).

decoding batches or partial tensor parallelism. The variables are designed to have a monotonic property, which we exploit in our scheduling algorithm to find an optimal execution schedule. This schedule is enforced by our distributed runner based on FasterTransformer.

We evaluated ExeGPT and other cutting-edge inference systems with six LLM configurations and five NLP tasks. ExeGPT shows significant improvements in throughput, up to 15.2×, and latency, up to 6×, when compared to Faster-Transformer. Our contributions are summarized as follows. Scheduling Strategies and Latency/Throughput Tradeoff. We propose novel scheduling strategies based on two resource allocation policies, i.e., Round-Robin Allocation (RRA) and Workload-Aware Allocation (WAA). These strategies decouple the execution of encoding and decoding, allowing for efficient optimization of each phase. The strategies offer four control variables, enabling flexible trade-offs between throughput and latency for different workloads and applications. The scheduling strategies and the trade-off mechanism form the basis of our optimal scheduling algorithm.

Scheduling with Sequence Length Distribution. Our scheduler incorporates input and output sequence distributions, which can be obtained by observing NLP services over time, to determine the optimal schedule for real-world NLP workloads. We conducted probabilistic analysis to estimate the number of completed queries in decoding iterations and also their execution times using the profiling results. With the estimated execution times, we accurately simulate the timeline for the two strategies and find the optimal execution schedule.

Optimizing Scheduling Algorithm. We formulated an optimization problem for LLM inference to maximize throughput under a given latency bound by finding the optimal values of the control variables for RRA and WAA Scheduling. The optimization problem is monotonic, as the control variables are designed to be monotonic with respect to

the throughput and latency. This allows us to implement an efficient optimization algorithm based on the standard branch-and-bound method. Our scheduling algorithm shows significant improvements in throughput and latency over existing methods.

Extensive Evaluation. We evaluated ExeGPT with six LLM instances of T5, OPT, and GPT-3 and five NLP tasks of summarization, translation, code generation, and conversational Q/A. For each task, we evaluated with four distinct latency constraints and measured the throughput. Compared to Faster-Transformer, the state-of-the-art inference system, ExeGPT achieves up to 15.2× improvement in throughput and 6× in latency. On average, ExeGPT achieves 2.9× higher throughput across all the LLMs and tasks. Also, we observed that when adapting to changing sequence distributions, the cost of adjusting the schedule in ExeGPT is reasonably modest.

The paper is organized as follows. Section 2 provides background on LLM inference. Section 3 presents an overview of ExeGPT and its key components. Section 4 introduces our proposed scheduling strategies and their latency/throughput trade-off mechanism. In Section 5, we formulate the optimization problem for LLM inference and propose an efficient scheduling algorithm. Section 6 describes our simulation of execution schedules using the sequence length distributions. Section 7 evaluates ExeGPT and compares it with state-ofthe-art systems. Finally, Section 8 concludes.

