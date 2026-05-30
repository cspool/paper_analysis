# <span id="page-8-0"></span>6 Adaptive Memory Management

#### 6.1 Motivation: Performance Degradation

As mentioned in Section 1, most existing works designed for long-context input scenario determines the KV cache management strategy that whether to store the entire cache in GPU HBM or offload it to CPU DRAM before LLM inference. However, unlike the long-context input scenario, the sequence length exhibits is dynamic and unpredictable in the long-context reasoning scenario because the inference termination is completely determined by the LLM itself. Consequently, as shown in Figure 2(a), even a tiny increase in task workload (e.g., the longer context length and more requests) can trigger a complete offload of the entire KV cache to the CPU, leading to > 80% performance degradation.

#### 6.2 Approach: Adaptive Memory Management

**Theoretical Model and Analysis.** Inspired by a series of works on adaptive scheduling for resource allocation [9, 47], we develop a theoretical memory overhead model based on LLM architecture, hardware specifications and inference workload detailed in Table 3, and further propose a novel adaptive memory management system.

During LLM inference, additional memory, called runtime memory, is required to serve as a temporary buffer to store intermediate values (*e.g.*, activations). We checked some references and found that runtime memory typically amounts to around 20% to 30% of the model size [5, 9, 15, 53]. This buffer is dynamically used and released during the LLM inference. As a result, we select 30% of the model size as the runtime memory. As mentioned in Section 4, the decode layer of DLM is consistent with the original LLM architecture and has only one layer, and due to the repeat\_kv operation in GQA or MQA, an additional buffer( $S\alpha HD$ ) must allocated for computation. So the total number of layers in the KV cache is  $(L+1+\alpha)$ . Moreover, Due to the Key and Value with FP16 precision which is 2 byte per value, the coefficient of KV cache is 4. Therefore, we can calculate the total memory

#### <span id="page-9-3"></span>**Algorithm 1** Adaptive memory management in inference

 Input: The sequence length threshold list:  $S^T = [S_0^T, ...S_L^T]$ ,

 Prompt:  $input\_id$ , LLM Layers:  $Layer = [Layer_0, ...Layer_{L-1}]$ .

 1:  $L_{CPU} \leftarrow 0$ ;  $L_{GPU} \leftarrow L$  ▶ Initiate  $L_{CPU}$  and  $L_{GPU}$  

 2:  $S \leftarrow len(input\_id)$  ▶ Initiate Sequence length S 

```
▶ Initiate Sequence length S
 3: while True do
       while S \ge S_{L_{CPU}}^T and L_{CPU} < L do
 4:
          KV\_Cache\_Offload(L - L_{CPU} - 1) \Rightarrow Offload the KV
          cache of Layer_{L-LCPU-1}
          L_{CPU} \leftarrow L_{CPU} + 1
 6:
       end while
 7:
       input id = LLM(input id)
 8:
       S \leftarrow S + 1
 9:
10:
       if stop_id in input_id then
          Break
11:
12:
       end if
13: end while
```

requirements for placing all KV cache the GPU as follows.

<span id="page-9-0"></span>
$$M_{all} = M_{Model} + M_{KV} = 1.3(M_O + M_D) + 4R(L + 1 + \alpha)SHD$$
 (6)

Based on the Equation 6, if we keep all the data on the GPU, we need to make sure that  $M_{all} < Mem_{GPU}$ .

For the resource-constrained environment (e.g., low-end GPU with limited memory (i.e.,  $Mem_{GPU}$  is insufficient) and high-end GPU with multi-requests (i.e.,  $M_{all}$  is too large)), it is necessary to split the KV cache across different memory tiers. Specifically, the KV cache of some layers ( $L_{GPU}$ ) should be stored on the GPU, while the KV cache of others ( $L_{CPU}$ ) are offloaded to the CPU. However, for the layers where the KV cache is offloaded to the CPU, it is still necessary to reserve a small GPU buffer to store KV cache budget (B) loaded from the CPU for the computation. Therefore, the total memory requirements in this case can be calculated as follows:

$$M_{part} = 1.3(M_O + M_D) + 4R[(L_{GPU} + 1 + \alpha)S + (L_{CPU}B)]HD$$
(7)

To maximize the utilization of the GPU memory, the theoretical optimization model thus can be abstracted as follows.

<span id="page-9-1"></span>
$$Max(L_{GPU});$$
  
 $s.t. \ M_{part} \le Mem_{GPU}$  (8)

**6.2.1 Implementation Details.** As mentioned above, the sequence length grows continuously during long-context reasoning. Following the objective of maximizing  $L_{GPU}$  in Equation 8, we propose an adaptive memory management system that progressively offloads the KV cache of each LLM layer to the CPU as the context length increases during reasoning, thereby freeing additional GPU memory to store more KV cache in other layers. Our analysis indicates that during the inference, the primary factor influencing memory overhead is the sequence length. Capitalizing on this, we can pre-calculate the sequence length thresholds in compilation detailed in Algorithm 2. The threshold  $S_0^T$  represents that

<span id="page-9-2"></span>**Algorithm 2** Sequence length threshold calculation in compilation

```
Input: The Symbols in Table 3.

Output: The sequence length threshold list S^T = [S_0^T, ... S_L^T].

1: S_0^T \leftarrow \lfloor \frac{Mem_{GPU} - 1.3 \times (M_O + M_D)}{4 \times R \times H \times D \times (L + 1 + \alpha)} \rfloor \Rightarrow \text{Place all KV cache on GPU}

2: \mathbf{for} \ i \leftarrow 1 \ \text{to} \ L \ \mathbf{do}

3: S_i^T = \lfloor \frac{Mem_{GPU} - 1.3 \times (M_O + M_D) - (i \times B) \times R \times H \times D}{4 \times (L + 1 + \alpha - i) \times R \times H \times D} \rfloor \Rightarrow \text{Place last}
\ni layers of KV cache on GPU

4: \mathbf{end} \ \mathbf{for}

5: \mathbf{return} \ S^T
```

<span id="page-9-4"></span>if we want to place all the KV cache on GPU, the current sequence length(S) must be smaller than  $S_0^T$ . If  $S > S_0^T$ , we need to offload the KV cache of the final layer to CPU for more GPU memory.

During LLM inference, the adaptive memory management system will offload the KV cache of an additional layer to CPU DRAM at the exact time point based on these thresholds to maintain optimal memory usage. Algorithm 1 shows the details of LLM inference with adaptive memory management. For example, if the prompt length is between  $S_2^T$  and  $S_3^T$  at the beginning of LLM inference, the system will offload the KV cache of last two layers(e.g., the 31st and 32nd layer in Llama3-8B) to CPU DRAM and keep the KV cache of the left layers on GPU. As sequence length increases during inference(i.e., line 9 in Algorithm 1) and exceeds  $S_3^T$ , the management system will offload the KV cache of the 30th layer to CPU. With the adaptive memory management, we maximize the utilization of GPU HBM for better performance and convenient deployment.

#### 7 Evaluation

#### 7.1 Environmental Setup

We evaluate the performance of *SpeContext* with various LLMs in two resource-constrained environments, a low-end GPU with limited memory in edge and a high-end GPU with multi-requests in cloud, targeting the long-context input and reasoning scenarios. We compare the performance with several LLM inference engines and some latest works on KV cache optimization in these two environments.

Table 4. Hardware Platforms

<span id="page-9-5"></span>

|     | High-end GPU                            | Low-end GPU                             |
|-----|-----------------------------------------|-----------------------------------------|
| GPU | A800, 80GB HBM<br>CUDA 12.1             | RTX 4060 Laptop<br>8GB GDDR6, CUDA 12.6 |
| CPU | Intel Xeon Platinum 8358<br>1008GB DRAM | Intel i7-13650HX<br>24GB DRAM           |

Hardware Platforms. For the scenario of the high-end GPU with multi-requests in cloud, we choose a workstation with an NVIDIA A100-80GB GPU. For the scenario of the low-end GPU with limited memory in edge, we select the Lenovo Legion Y7000 PC with NVIDIA RTX 4060 Laptop

<span id="page-10-0"></span>![](_page_10_Figure_2.jpeg)

Figure 8. Accuracy in LongBench on Llama3.1-8B.

GPU(8GB) and Intel i7-13650HX CPU. Table [4](#page-9-5) shows the detailed hardware specification.

Baselines. To evaluate the performance, we select the typical LLM framework Huggingface [\[45\]](#page-14-11) and the famous LLM inference fast engine, FlashInfer [\[51\]](#page-14-3), as the baselines for full attention. We also select three latest open-sourced works on KV cache optimization, Quest [\[40\]](#page-13-12), ClusterKV [\[30\]](#page-13-10) and ShadowKV [\[39\]](#page-13-11) as the baselines for sparse attention.

Models and Benchmarks. We select three LLMs, Llama3.1- 8B [\[17\]](#page-13-8), DeepSeek-R1-Distill-Llama-8B [\[29\]](#page-13-25) and Qwen3-8B [\[49\]](#page-14-9) for evaluation in the cloud environment with multiple requests on single GPU. We also select a larger LLM, Llama3.1- 70B [\[17\]](#page-13-8) for evaluation on multi-GPUs in cloud with multiple requests. For the edge environment, we select the Reasoning-Llama-3.2-1B [\[12\]](#page-13-29), which is the reasoning model finetuned on Llama3.2-1B [\[17\]](#page-13-8), due to the limited memory. To evaluate the accuracy of SpeContext in two scenarios mentioned above, we select the four tasks(i.e., 2WiKiMQA, TriviaQA, HotpotQA, and Passage count) from LongBench [\[2\]](#page-12-5) for the longcontext input scenario, the LongWriter benchmark [\[3\]](#page-12-6) for the long-context reasoning scenario and the ultrachat [\[11\]](#page-13-30) dataset for multi-round dialogue scenario.

