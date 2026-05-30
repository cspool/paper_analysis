# MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

Dennis Liu<sup>∗</sup> Zijie Yan<sup>∗</sup> Xin Yao Tong Liu Vijay Korthikanti Evan Wu Shiqing Fan Gao Deng Hongxiao Bai Jianbin Chang Ashwath Aithal Michael Andersch Mohammad Shoeybi Jiajie Yao Chandler Zhou David Wu Xipeng Li June Yang † NVIDIA

{denliu, zijiey, xiny, tongliu, vkorthikanti, evwu, shiqingf, gdeng, hongxiaob, jianbinc, aaithal, mandersch, mshoeybi, jiajiey, chandlerz, davidwu, xipengl, juney}@nvidia.com

### Abstract

Mixture of Experts (MoE) models enhance neural network scalability by dynamically selecting relevant experts per input token, enabling larger model sizes while maintaining manageable computation costs. However, efficient training of largescale MoE models across thousands of GPUs presents significant challenges due to limitations in existing parallelism strategies. We introduce an end-to-end training framework for large-scale MoE models that utilizes five-dimensional hybrid parallelism: Tensor Parallelism, Expert Parallelism, Context Parallelism, Data Parallelism, and Pipeline Parallelism. Central to our approach is MoE Parallel Folding, a novel strategy that decouples the parallelization of attention and MoE layers in Transformer models, allowing each layer type to adopt optimal parallel configurations. Additionally, we develop a flexible token-level dispatcher that supports both token-dropping and token-dropless MoE training across all five dimensions of parallelism. This dispatcher accommodates dynamic tensor shapes and coordinates different parallelism schemes for Attention and MoE layers, facilitating complex parallelism implementations. Our experiments demonstrate significant improvements in training efficiency and scalability. We achieve up to 49.3% Model Flops Utilization (MFU) for the Mixtral 8x22B model and 39.0% MFU for the Qwen2-57B-A14B model on H100 GPUs, outperforming existing methods. The framework scales efficiently up to 1,024 GPUs and maintains high performance with sequence lengths up to 128K tokens, validating its effectiveness for large-scale MoE model training. The code is available in Megatron-Core [1](#page-0-0) .

# 1 Introduction

In recent years, Mixture of Experts (MoE) models have emerged as a powerful architecture for scaling neural networks to unprecedented sizes. By leveraging multiple experts in one model and dynamically selecting the most relevant experts for each input token, MoE models can accommodate larger parameter counts while maintaining manageable computational costs. This approach not only enhances model capacity but also improves performance on a variety of tasks compared to their dense counterparts. Recent MoE models scale to hundreds of billions or even trillions of parameters and achieve state-of-the-art performance [\[12;](#page-12-0) [28;](#page-13-0) [38;](#page-13-1) [37;](#page-13-2) [3;](#page-11-0) [33\]](#page-13-3).

<sup>∗</sup>These authors contributed equally to this work. Authors are listed in alphabetical order.

<sup>†</sup>Corresponding author: juney@nvidia.com

<span id="page-0-0"></span><sup>1</sup> https://github.com/NVIDIA/Megatron-LM

Training large-scale MoE models, however, presents significant challenges. As the model size increases, efficient distributed training across thousands of GPUs becomes essential. Different parallelism strategies have been proposed in recent years for distributed LLM training, including model parallelism, data parallelism, and pipeline parallelism [\[31;](#page-13-4) [27;](#page-13-5) [19\]](#page-12-1). However, a single parallelism strategy has limitations regarding scalability. For example, the performance of data parallelism with ZeRO-3 will decrease dramatically when the number of GPUs increases to several thousands [\[21\]](#page-12-2).

To address the growing computational and memory requirements of increasingly large models, hybrid parallelism—which integrates multiple parallelization strategies—has become essential. While 3D parallelism is widely adopted for training large-scale dense models, optimizing training efficiency for MoE models using hybrid parallelism presents greater complexity. This is primarily due to the inherent sparsity of MoE models, which results in a significantly lower computation-to-parameter ratio compared to dense models. Employing a small degree of model parallelism often leads to out-of-memory issues for MoE models, whereas a large degree introduces substantial communication overhead and diminishes computational efficiency.

Previous approaches have predominantly relied on expert parallelism coupled with data parallelism to scale the training of MoE models. However, the diversity in the number of experts, the size of individual experts, and the sequence length of training samples across different MoE models necessitates an adaptive parallelism strategy tailored to each scenario. Moreover, the distinct computational characteristics of the Attention and Feed-Forward Network (FFN) layers in MoE models render a uniform parallelism strategy across these layers suboptimal. The dynamic nature of MoE models, including on-the-fly token routing and variable tensor shapes, further exacerbates the complexity of designing efficient training algorithms. These challenges collectively underscore the need for a coordinated integration of multiple parallelism strategies to optimize the training of MoE models.

To address these challenges, we propose an end-to-end training framework for large-scale MoE models based on 5-D hybrid parallelism, which integrates five key parallelism dimensions: Tensor Parallelism(TP), Expert Parallelism(EP), Context Parallelism(CP), Data Parallelism(DP), and Pipeline Parallelism(PP). At the core of our framework are two innovations: MoE Parallel Folding and an efficient token-level dispatcher. MoE Parallel Folding is a novel hybrid parallelism strategy that disentangles the parallel mappings of the Attention and MoE components in Transformer-based models. Our key insight is that enabling flexible and distinct parallelism configurations for these components unlocks a comprehensive parallelism space, ensuring optimal performance. Additionally, to support arbitrary parallelism combinations while maintaining numerical correctness, we designed a highly flexible and efficient token-level MoE dispatcher. This dispatcher accommodates both token-dropping and token-dropless training paradigms, eliminates sequence length dependencies, and enables dynamic tensor shapes, thereby facilitating the implementation of complex parallelism schemes.

The contributions of this paper are as follows:

- 1. MoE Parallel Folding: We introduce MoE Parallel Folding, the first approach that decouples parallelization strategies for attention and MoE layers, enabling each layer to adopt its own optimal configurations. This method enables the folding of communication-intensive parallel dimensions to fit within high-bandwidth intra-node networks, reducing communication overhead.
- 2. Flexible and efficient token-level dispatcher: We develop a novel dispatcher that supports both token-dropping and token-dropless MoE training with five-dimensional hybrid parallelism, including TP, EP, CP, DP, and PP.
- 3. Performance enhancements: Through MoE Parallel Folding, we demonstrate significant improvements in training efficiency and scalability for large-scale MoE models. By optimizing the utilization of network resources based on model characteristics, we achieve 49.3% MFU for Mixtral 8x22B and 39.0% MFU for Qwen2-57B-A14B on H100 GPUs.

# 2 Related Work

#### 2.1 Mixture of Experts

The Mixture of Experts (MoE) architecture [\[29\]](#page-13-6) enhances neural network capacity and efficiency by integrating multiple specialized sub-networks, known as experts, under the supervision of a routing mechanism. Each expert specializes in different regions of the input space or captures distinct features of the data. The router dynamically selects the most relevant experts for each input, enabling the model to process a diverse range of patterns more effectively than traditional monolithic architectures. This selective activation allows MoE models to scale significantly without a proportional increase in computational complexity, as only a subset of experts is engaged per input.

Incorporating MoE architectures into Transformer models [\[36\]](#page-13-7) has led to substantial advancements, achieving superior performance compared to dense counterparts. Notable examples include GShard [\[15\]](#page-12-3), which scaled models to trillions of parameters using MoE layers, and the Switch Transformer [\[7\]](#page-11-1), which improved scalability and efficiency with a streamlined routing mechanism. Similarly, GLaM [\[4\]](#page-11-2) demonstrated the effectiveness of MoE in large-scale language modeling tasks. These models leverage the sparsity introduced by MoE to maintain manageable computational demands, activating only a subset of experts per input to reduce overall computation and memory requirements.

Addressing challenges such as load balancing among experts and managing dynamic computation graphs is critical for MoE architectures. Traditional methods often employ token-dropping training [\[7\]](#page-11-1), setting a capacity factor for each expert to prevent overloading. While this mitigates performance bottlenecks, it can result in some tokens being dropped or not fully processed, potentially affecting model quality. In contrast, Megablocks [\[8\]](#page-11-3) utilizes token-dropless training to ensure all input tokens are processed, demonstrating better performance for models of equivalent size and training data by avoiding the loss of information inherent in token dropping.

Recent developments focus on fine-grained MoE architectures to further enhance performance [\[14;](#page-12-4) [9\]](#page-11-4). Approaches like DeepSeek-MoE [\[2;](#page-11-5) [28\]](#page-13-0) segment experts into smaller sub-experts and activate a greater number of experts per token, achieving higher degrees of specialization. This fine-grained specialization allows models to capture complex patterns and relationships within the data more effectively.

### 2.2 Distributed MoE Training

The substantial size of Large Language Models (LLMs) often exceeds the memory and computational capacity of a single GPU, necessitating distributed training strategies to manage resource constraints effectively. Conventional distributed training methods include TP, DP, CP and PP. TP divides the computations of neural network layers across multiple devices, allowing for parallel processing of tensors within layers[\[31\]](#page-13-4). TP can significantly reduce the memory consumption of each model rank but introduces some intra-layer communication overhead. DP distributes batches of data across replicas of the model on different devices, aggregating gradients during training[\[35\]](#page-13-8). Zero Redundancy Optimizer(ZeRO) further splits optimizer states, model weights and gradients across DP group to trade memory with communication [\[16;](#page-12-5) [27\]](#page-13-5). CP splits the input sequences into small segments for each device, allowing for very long sequence length training[\[11;](#page-11-6) [17\]](#page-12-6). PP splits[\[6;](#page-11-7) [21;](#page-12-2) [19;](#page-12-1) [20\]](#page-12-7) the model layers across devices, enabling different stages of the model to process data concurrently in a pipelined fashion.

In the context of MoE models, EP is employed to optimize MoE training by assigning different experts to different devices[\[4;](#page-11-2) [7;](#page-11-1) [1;](#page-11-8) [15;](#page-12-3) [30\]](#page-13-9). During training, the routing mechanism directs inputs to the appropriate experts across devices. EP efficiently utilizes hardware resources by balancing the computational load and reducing inter-device communication overhead associated with expert data exchanges. To further enhance the efficiency of distributed MoE training, hybrid parallelism strategies are leveraged which combines EP with other parallelism methods, like FSDP and TP[\[32;](#page-13-10) [26;](#page-12-8) [10;](#page-11-9) [8\]](#page-11-3).

# 3 Method

#### 3.1 Preliminary

#### 3.1.1 Mixture of Experts

The Mixture of Experts (MoE) is a neural network architecture that dynamically selects the most relevant experts to process each individual token. The MoE layer consists of E expert networks and a gating network, which determines the routing by computing a selection probability for each expert. The output of the MoE layer is computed as a weighted aggregation of the outputs from the selected experts, based on their gating probabilities:

$$\mathbf{y} = \sum_{e=1}^{E} g_e(\mathbf{x}) \cdot f_e(\mathbf{x}), \tag{1}$$

where ge(x) denotes the gating weight for expert e, and fe(x) represents the output of expert e. Among the various gating strategies, the Top-K gating method is the most widely used:

$$g_e(\mathbf{x}_i) = \begin{cases} s_i & \text{if } i \in \text{TopK}(\mathbf{s}, K), \\ 0 & \text{otherwise,} \end{cases}$$
 (2)

where x<sup>i</sup> is the i-th token input to the current expert, and s is computed as follows:

$$\mathbf{s} = G(W_g \cdot \mathbf{x}). \tag{3}$$

Here, W<sup>g</sup> represents the weight matrix of the gating network, and G denotes a non-linear activation function.

The learnable gating network enables dynamic routing, which can lead to load imbalance issues. To mitigate this, in addition to the auxiliary loss, a capacity factor (CF) is introduced to regulate load balancing among experts. The capacity factor defines the maximum capacity of each expert relative to the average expected load, ensuring that computational resources are evenly distributed and preventing bottlenecks caused by uneven workloads. The capacity per expert is calculated as:

Capacity per Expert = 
$$CF \cdot \frac{L}{E}$$
, (4)

where L is the total number of tokens to process, E is the number of experts, and CF ≥ 1 is the capacity factor. Tokens that exceed the capacity of a given expert are dropped.

#### 3.1.2 Expert Parallelism

In large-scale distributed training of MoE models, EP efficiently distributes computation across multiple devices by assigning distinct experts to each device. This parallelization strategy reduces communication overhead while optimizing hardware utilization. The EP process comprises three key stages:

Token Dispatching Initially, input tokens are grouped according to their assigned experts through data permutation, ensuring tokens destined for the same expert are stored contiguous in memory. An *All-to-All* collective communication operation then exchanges token data between devices, allowing each device to receive only the tokens required by its locally hosted experts.

Expert Computation Each device processes its local batch of tokens through its designated experts in parallel. Since experts operate independently, this stage requires no inter-device communication, allowing for efficient concurrent computation across the distributed system.

Token Restore After expert processing, the output tokens are rearranged to restore their original sequence order through an inverse permutation operation. This restoration step ensures proper alignment for subsequent layer operations while maintaining the model's sequential processing requirements. The restored tokens can then flow into the next layer of the network.

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 1: Illustration of parallelism mappings with MoE Parallel Folding.

### 3.2 MoE Parallel Folding

Attention layers and MoE layers in Transformers exhibit distinct computation and communication patterns. Attention operations are performed at the whole-sequence level with dense computation, requiring information exchange between devices holding sub-sequences when using TP and CP. In contrast, MoE layers process individual tokens rather than whole sequences, and their inherent sparsity makes them more suitable for EP with lower communication overhead.

Consequently, forcing MoE layers to follow the same parallelism mapping as Attention layers is sub-optimal. To achieve optimal hybrid parallelism for MoE models, we propose MoE Parallel Folding, which disentangles the parallel mappings between Attention and MoE layers.

As shown in Figure [1,](#page-4-0) previous methods place the EP group in a sub-group of DP, which greatly restricts the scalability of MoE. The maximum degree of expert parallelism is bounded by the degree of data parallelism. Instead, we flatten the parallelism mappings of the attention layer and allow model parallelism in the MoE layer to be folded with arbitrary sub-groups of attention, making the parallelism mappings of MoE layer more flexible and efficient.

Specifically, for the attention layers, we form a four-dimensional parallel group comprising T P × CP × DP × P P. For the MoE layers, we define another four-dimensional group consisting of T P × EP × DP × P P. For convenience, we name the TP and DP group for MoE layer as Expert-TP(ETP) and Expert-DP(EDP). The only restriction is that the number of PP groups and members of each PP group for the Attention and MoE layer must be consistent. This separation allows us to set flexible and independent parallelism configurations for attention and MoE layers.

MoE Parallel Folding provides two main benefits. First, it allows selecting the optimal parallelism mapping for the MoE layer independently of the Attention layer. For example, EP is more communication-efficient than ETP. We can replace ETP with EP and fold it with TP in the Attention layer. Second, the folded parallelism mappings enable communication within more compact groups. By folding model parallelism across attention and MoE layers, the scope of intra-layer communication is reduced, allowing it to fit within high-bandwidth intra-node connections more effectively.

### 3.3 Flexible and Efficient Token Dispatcher

Arbitrary hybrid parallelism with MoE Parallel Folding necessitates a flexible and scalable token dispatcher. The dispatcher is responsible for routing tokens to their assigned experts across various parallelism dimensions. To ensure numerical correctness while maintaining high performance under different parallelism strategies, we have designed a unified token dispatcher that handles both ETP and EP within the MoE layer.

With MoE Parallel Folding, the inputs fed into the MoE layer from the attention layer are split either along the batch dimension (DP) or the sequence dimensions (CP and TP). In both scenarios, different ranks contain different chunks of tokens. Since the expert layer computes the features of each token individually, we can employ the same workflow for the token dispatcher regardless of the parallelism mappings of the attention layer.

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 2: Workflow of token dispatcher with Tensor Parallelism and Expert Parallelism.

In Figure [2,](#page-5-0) we illustrate the workflow of an MoE layer distributed across four GPUs, where the degrees of TP and ETP are both 2. GPU pairs (0, 1) and (2, 3) form the ETP group. GPU pairs (0, 2) and (1, 3) form the EP group.

The forward computation workflow proceeds as follows. First, the router determines the mapping of each token to its designated expert based on the local input and reorganizes the tokens assigned to the same expert into contiguous memory regions through a permutation operation. Next, an All-to-All-V communication is executed across the EP groups to exchange tokens, ensuring that each token is delivered to its corresponding expert. Following this, an AllGather-V communication is performed within the ETP groups to guarantee that all members within an ETP group share identical activations. Once the AllGather-V communication is complete, each GPU computes its assigned partition of the expert feed-forward networks. A subsequent ReduceScatter-V communication within the ETP groups aggregates and distributes the output hidden states, effectively reversing the AllGather operation. Another All-to-All-V communication is then employed to return the tokens to their original GPUs. Finally, an un-permutation operation restores the tokens to their initial order, preparing them for further processing in the attention layer. The backward workflow mirrors the forward process, with the AllGather/ReduceScatter (AG/RS) operations in the TP groups replaced by ReduceScatter/AllGather (RS/AG).

We now elaborate on the design of the router to support both token-dropping and token-dropless training paradigms. The router assigns tokens to experts by selecting the top-k tokens based on their softmax probabilities. In token-dropless training, ensuring numerical correctness is straightforward, as token assignments remain consistent across different parallelism configurations. For tokendropping training, two potential strategies can be employed: full-sequence-based dropping and sub-sequence-based dropping.

- Full-sequence dropping ensures consistency by gathering logits from all ranks that collectively represent the entire sequence. However, this approach incurs significant communication overhead, particularly when sequences are distributed across multiple nodes.
- Sub-sequence dropping, on the other hand, makes dropping decisions based solely on the logits from the current sub-sequence. This strategy eliminates the need for gathering logits across ranks, thereby reducing communication overhead and alleviating load imbalance issues during token communication.

Empirically, we observe that sub-sequence dropping does not adversely affect model convergence compared to full-sequence dropping. Consequently, we adopt the sub-sequence dropping approach as the default strategy in this work.

# 4 Experiments

#### 4.1 Experimental Setup

All experiments in this work were conducted on the Eos [\[34\]](#page-13-11) cluster. The Eos cluster consists of NVIDIA DGX H100 nodes, each equipped with eight NVIDIA H100 GPUs [\[24\]](#page-12-9) and two 56-core Intel Sapphire Rapids CPUs. Each GPU achieves a peak half-precision throughput of 989.5 TFLOP/s, and all GPUs are interconnected via NVLink 4th Generation [\[22\]](#page-12-10) and InfiniBand [\[23\]](#page-12-11). The peak uni-directional communication bandwidths are 450 GB/s for intra-node (NVLink) and 400Gbps for inter-node (InfiniBand) connections. We utilize PyTorch 2.5.0 and CUDA 12.6 for our experiments. All performance measurements reported in TFLOPS and MFU are conducted using BF16 precision. Up to 1024 GPUs are utilized in the scaling experiments.

We select two types of MoE models for our experiments, coarse-grained and fine-grained MoE, each type containing models of two different sizes. Compared to coarse-grained MoE, fine-grained MoE has a larger number of experts and more activated experts per token, but each expert has a reduced hidden size. For the coarse-grained MoE, we select the Mixtral 8x22B [\[18\]](#page-12-12) model and design a larger MoE named Llama3-8x70B by upcycling Llama3-70B [\[5\]](#page-11-10) to 8 experts [\[13\]](#page-12-13). For the fine-grained MoE, we choose Qwen2-57B-A14B [\[38\]](#page-13-1), which has 64 experts and 8 active experts per token, totaling 57 billion parameters with 14 billion active parameters. To obtain a larger fine-grained MoE model, we reparameterized the Mixtral 8x22B model to 64 experts and 8 active experts per token called Mixtral-8x22B-G8T8, with each expert possessing a hidden size that is one-eighth of the original model, by applying fine-grained upcycling [\[9\]](#page-11-4).

#### 4.2 Performance Comparison

To evaluate the performance of our proposed MoE Parallel Folding technique compared to existing parallelism strategies, we conducted comparative experiments using the four models previously described. The primary metric for assessment was the Model TFLOPS Utilization (MFU) during training, which measures the efficiency of computational resource utilization by comparing theoretical peak performance with the actual achieved performance in BF16 precision. To alleviate the performance jitter caused by load imbalance issues in dropless training, we use token drop training with a capacity factor equal to 1 for benchmarking.

For baseline comparisons, we chose four representative baseline parallelism strategies:

- 1. FSDP [\[39\]](#page-13-12): A data parallelism method that shards model parameters, gradients, and optimizer states across workers.
- 2. FSDP + EP [\[8\]](#page-11-3): An extension of FSDP that incorporates EP.
- 3. TP+EP+DP [\[32\]](#page-13-10): An framework combining TP and EP to fit larger MoE models across multiple GPUs.
- 4. MCore with 5D-parallelism[\[21\]](#page-12-2): The state-of-the-art training framework for large scale LLM models, supporting TP,EP,CP,DP and PP.

All baseline methods were implemented using the NVIDIA Megatron-Core framework[2](#page-6-0) . For each method, we report the MFU achieved with the optimal parallelism configuration found by tuning its supported parallelism dimensions.

Table [1](#page-7-0) presents the comparison results of different parallelism strategies on the selected MoE models. The observed MFU values highlight several key insights into the performance implications of each strategy: (1) FSDP exhibits poor performance(<10% MFU) due to their sparse computations and large parameter counts. In FSDP, the communication of parameters and gradients cannot be effectively overlapped with computation. Additionally, FSDP fails to train larger models like Llama3-8x70B due to out-of-memory (OOM) issues. (2) FSDP + EP improves performance, by parallelizing expert across GPUs, thereby reducing communication of expert parameters and gradients. However, this strategy still suffers from communication overhead that cannot be fully overlapped with computation, limiting further performance gains. (3)TP + EP + DP [\[32\]](#page-13-10) further uses TP to split the model weights to multiple GPUs and use ZeRO-1 instead of ZeRO-3 to reduce communication overhead

<span id="page-6-0"></span><https://github.com/NVIDIA/Megatron-LM>

<span id="page-7-0"></span>Table 1: Performance comparison of different parallelism strategies by MFU. The global batch size for experiment is 256.

|                  |                      | Coarse-grained      | Fine-grained         |                           |  |  |  |
|------------------|----------------------|---------------------|----------------------|---------------------------|--|--|--|
| GPUs             | Mixtral-8x22B<br>128 | Llama3-8x70B<br>256 | Qwen2-57B-A14B<br>64 | Mixtral-8x22b-G8T8<br>128 |  |  |  |
| FSDP             | 4.3%                 | OOM                 | 9.9%                 | 2.2%                      |  |  |  |
| FSDP + EP        | 23.4%                | 19.6%               | 25.4%                | 9.0%                      |  |  |  |
| TP+EP+DP         | 36.6%                | OOM                 | 23.1%                | 8.7%                      |  |  |  |
| MCore            | 46.3%                | 38.8%               | 35.3%                | 17.1%                     |  |  |  |
| MCore w/ Folding | 49.3%                | 41.6%               | 39.0%                | 28.8%                     |  |  |  |

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 3: Strong scaling experiments for various parallelism strategies by increasing number of GPUs up to 1024.

of parameters, resulting in better performance. But a large TP also introduces significant activation communication overhead. And the largest model Llama3-8x70B could not be trained using only TP+EP due to memory constraints. (4)MCore framework leverages pipeline parallelism (PP) in addition to TP, EP, and DP, achieves a better balance between communication and computation. This results in higher MFU values, reaching 46.3% on Mixtral-8x22B and 35.3% on Qwen-2-57B. By effectively partitioning the model across pipeline stages, MCore reduces the memory footprint per GPU and overlaps communication with computation more efficiently. However, the coupling of parallelism strategies between the Attention and MoE layers renders the mappings sub-optimal for MoE models. (5)MCore with MoE Parallel Folding: further enhances training efficiency, achieving the highest MFU values across all models: 49.3% for Mixtral-8x22B, 41.6% on Llama3-8x70B, 39.0% on Qwen-2-57B, and 28.8% on Mixtral-8x22B-G8T8. The flexible parallelism provided by MoE Parallel Folding allows for a more optimal parallelism strategy tailored to the characteristics of MoE models. By folding MoE parallel groups with Attention and effectively utilizing available hardware resources, it minimizes communication overhead and maximizes computational efficiency. This leads to significant performance improvements over existing strategies.

The experiments also reveal that fine-grained MoE models achieve lower training efficiency compared to coarse-grained MoE models across all parallelism strategies. This performance gap stems from two key factors: (1) Fine-grained MoE models generate higher communication volume due to their architecture - they employ more experts and activate more experts per token, increasing communication overhead during the token dispatching process. Additionally, the smaller hidden sizes decrease GEMM efficiency. (2) Fine-grained MoE models typically incorporate a larger number of

local and active experts, leading to significant memory overhead for storing activations. The memory requirements for managing numerous experts force the use of larger model parallelism sizes, which introduces additional communication costs and further reduces training efficiency.

#### 4.3 Scaling Experiments

Strong Scaling To evaluate the scalability of our methods, we conduct strong scaling experiments by increasing the number of GPUs up to 1,024. The global batch size is set to 1024 in the scaling experiments. As shown in Figure [3,](#page-7-1) our framework maintains consistently higher MFU compared to baseline approaches as the GPU count increases across all model types. The results show the scalability of MoE parallel folding up to 16x nodes with little MFU drops, especially for large-scale models like Llama3-8x70B, where the MFU only drops from 43.7% to 41.5%.

Scaling with Context Length To evaluate the capability of our framework to train large scale MoE models with very long context lengths, we conducted context scaling experiments by increasing the sequence length while keeping the total number of tokens per global batch constant. As shown in Figure [4,](#page-8-0) our framework can train MoE models with high efficiency up to a context length of 128K tokens, and the MFU only drops from 38.7% to 35.9% for Qwen-57B14A and 47.6% to 42.9% for Mixtral-8x22B. With MoE parallel folding, MCore can achieve higher performance by folding the parallelism groups of attention and MoE layers to better utilize the intra-node communication bandwidth.

#### 4.4 Ablation Study

To systematically evaluate the performance characteristics of MoE layers and quantitatively assess the advantages of MoE parallel folding, we conducted comprehensive ablation studies. Our methodology involves varying the parallelism mappings of the MoE layer while maintaining fixed parallelism configurations for the Attention layer. Specifically, we examine the Attention layer's parallelism mappings across TP and CP, while the MoE layer's parallelism mappings are analyzed with respect to EP and ETP.

In the first experimental setup, we configure the Attention layer with TP=4 and CP=1 (no context parallelism). We evaluate parallelism mappings for the MoE layer with EPxETP=8 and EPxETP=16, which enables us to examine both intra-node and inter-node communication patterns. Notably, the memory utilization remains consistent across different configurations when the product ETPxEP is held constant.

Figure [5](#page-9-0) presents detailed latency breakdowns for the MoE layer in both the standard Mixtral 8x22B model and its fine-grained variant Mixtral 8x22B G8T8. Configurations enabled by MoE parallel folding are denoted with an asterisk (\*). Our analysis reveals several key findings: (1) MoE Parallel Folding significantly expands the available parallelism configuration space, enabling the discovery of optimal parallelism mappings. The configurations utilizing MoE parallel folding consistently achieve superior performance. (2) ETP in the MoE layer introduces substantially higher communication overhead compared to EP, with this effect being particularly pronounced in fine-grained MoE models. (3) Fine-grained MoE models exhibit notably lower computation-to-communication ratios. When ETPxEP exceeds 8, necessitating inter-node communication, communication overhead dominates,

<span id="page-8-0"></span>![](_page_8_Figure_8.jpeg)

Figure 4: Context-scaling experiments by increasing context length and number of GPUs up to 128K and 1024.

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 5: MoE layer breakdown with different parallelism mappings. Marker \* means the new parallelism mappings supported by MoE Parallel Folding.

<span id="page-9-1"></span>![](_page_9_Figure_2.jpeg)

Figure 6: MoE layer breakdown with different parallelism mappings. Marker \* means the new parallelism mappings supported by MoE Parallel Folding.

accounting for over 70% of the total latency. (4) Maintaining minimal model parallelism while favoring EP over ETP emerges as an effective strategy for optimizing MoE layer performance.

In the second experimental setup, we configure the Attention layer with various CP sizes and sequence lengths, and compare the performance of the MoE layer with and without parallel folding. Figure [6](#page-9-1) shows the breakdown results. As we can see, when the size of the CPxEP group exceeds 8 and spans beyond the NVLINK domain, the latency without MoE Parallel Folding increases significantly. Without MoE Parallel Folding, the EP group spans across multiple context parallelism groups, causing All-to-All communications within the EP group to traverse the lower-bandwidth inter-node network fabric. The MoE Parallel Folding technique allows the CP and EP groups to be folded together, maximizing the use of high-bandwidth NVLink connections whenever possible.

### 4.5 FP8 Training Performance

To further evaluate the capabilities of our framework, we investigated the performance benefits of utilizing FP8 precision, particularly relevant for newer hardware architectures like NVIDIA Hopper and NVIDIA Blackwell. We conducted experiments employing FP8 delayed scaling [\[25\]](#page-12-14) with the Mixtral 8x22B model on 128 H100 GPUs. The results demonstrate substantial throughput improvements compared to BF16 training.

Specifically, we observed the following performance in model TFLOPS:

These results, summarized in Table [2,](#page-10-0) indicate that FP8 training provides a significant performance uplift over BF16 (approximately 1.26x speedup without folding and 1.30x with folding). Furthermore, MoE Parallel Folding continues to enhance performance within the FP8 regime, yielding the highest throughput of 631.7 TFLOPS.

Table 2: Mixtral 8x22B Performance Comparison

<span id="page-10-0"></span>

| Configuration    | Precision | TFLOPS | Speedup vs BF16 | Speedup w/ Folding |
|------------------|-----------|--------|-----------------|--------------------|
| MCore            | BF16      | 458.3  | -               | -                  |
| MCore w/ Folding | BF16      | 487.7  | -               | 1.06x              |
| MCore            | FP8       | 575.1  | 1.26x           | -                  |
| MCore w/ Folding | FP8       | 631.7  | 1.30x           | 1.10x              |

### 5 Conclusion

In this paper, we introduce a novel framework for efficient large-scale MoE model training that addresses key challenges in distributed training through two main innovations. First, we propose MoE Parallel Folding, a technique that decouples the parallelization strategies of attention and MoE layers, enabling more flexible and efficient parallel configurations. This approach allows for optimal resource utilization by adapting to the distinct computational characteristics of each layer. Second, we develop an efficient token-level dispatcher that supports both token-dropping and token-dropless training across five dimensions of parallelism, providing a robust foundation for complex hybrid parallelism schemes. Our experimental results demonstrate significant performance improvements across different MoE architectures, achieving up to 49.3% MFU for Mixtral 8x22B and 39.0% MFU for Qwen2-57B-A14B on H100 GPUs. The framework shows strong scaling efficiency up to 1024 GPUs and maintains high performance with sequence lengths up to 128K tokens. These results validate the effectiveness of our approach in addressing the scalability challenges of large-scale MoE model training.

# References

- <span id="page-11-8"></span>[1] Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts. *arXiv preprint arXiv:2112.10684*, 2021.
- <span id="page-11-5"></span>[2] Damai Dai, Chengqi Deng, Chenggang Zhao, Runxin Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. In *Annual Meeting of the Association for Computational Linguistics*, 2024.
- <span id="page-11-0"></span>[3] Databricks. Introducing dbrx: A new state-of-the-art open llm, 2024. Accessed: December 11, 2024.
- <span id="page-11-2"></span>[4] Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen S. Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Z. Chen, and Claire Cui. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*, 2021.
- <span id="page-11-10"></span>[5] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, and Kevin Stone et al. The llama 3 herd of models. *ArXiv*, abs/2407.21783, 2024.
- <span id="page-11-7"></span>[6] Shiqing Fan, Yi Rong, Chen Meng, Zongyan Cao, Siyu Wang, Zhen Zheng, Chuan Wu, Guoping Long, Jun Yang, Lixue Xia, et al. Dapple: A pipelined data parallel approach for training large models. In *Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, pages 431–445, 2021.
- <span id="page-11-1"></span>[7] William Fedus, Barret Zoph, and Noam M. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *ArXiv*, abs/2101.03961, 2021.
- <span id="page-11-3"></span>[8] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei A. Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. *ArXiv*, abs/2211.15841, 2022.
- <span id="page-11-4"></span>[9] Ethan He, Abhinav Khattar, Ryan Prenger, Vijay Korthikanti, Zijie Yan, Tong Liu, Shiqing Fan, Ashwath Aithal, Mohammad Shoeybi, and Bryan Catanzaro. Upcycling large language models into mixture of experts. *ArXiv*, abs/2410.07524, 2024.
- <span id="page-11-9"></span>[10] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, Joe Chau, Peng Cheng, Fan Yang, Mao Yang, and Yongqiang Xiong. Tutel: Adaptive mixture-of-experts at scale. *ArXiv*, abs/2206.03382, 2022.
- <span id="page-11-6"></span>[11] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. *arXiv preprint arXiv:2309.14509*, 2023.

- <span id="page-12-0"></span>[12] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, L'elio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. *ArXiv*, abs/2401.04088, 2024.
- <span id="page-12-13"></span>[13] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. *ArXiv*, abs/2212.05055, 2022.
- <span id="page-12-4"></span>[14] Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pi'oro, Michal Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Kr'ol, Tomasz Odrzyg'o'zd'z, Piotr Sankowski, Marek Cygan, and Sebastian Jaszczur. Scaling laws for fine-grained mixture of experts. *ArXiv*, abs/2402.07871, 2024.
- <span id="page-12-3"></span>[15] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam M. Shazeer, and Z. Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *ArXiv*, abs/2006.16668, 2020.
- <span id="page-12-5"></span>[16] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. Pytorch distributed: Experiences on accelerating data parallel training. *arXiv preprint arXiv:2006.15704*, 2020.
- <span id="page-12-6"></span>[17] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. *ArXiv*, abs/2310.01889, 2023.
- <span id="page-12-12"></span>[18] Mistral AI. Introducing mixtral-8×22b. <https://mistral.ai/news/mixtral-8x22b/>, 2023.
- <span id="page-12-1"></span>[19] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R Devanur, Gregory R Ganger, Phillip B Gibbons, and Matei Zaharia. Pipedream: Generalized pipeline parallelism for dnn training. In *Proceedings of the 27th ACM symposium on operating systems principles*, pages 1–15, 2019.
- <span id="page-12-7"></span>[20] Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, and Matei Zaharia. Memoryefficient pipeline-parallel dnn training. In *International Conference on Machine Learning*, pages 7937–7947. PMLR, 2021.
- <span id="page-12-2"></span>[21] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei A. Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm. *SC21: International Conference for High Performance Computing, Networking, Storage and Analysis*, pages 1–14, 2021.
- <span id="page-12-10"></span>[22] NVIDIA Corporation. NVLink, 2016. [https://www.nvidia.com/en-us/data-center/](https://www.nvidia.com/en-us/data-center/nvlink/) [nvlink/](https://www.nvidia.com/en-us/data-center/nvlink/).
- <span id="page-12-11"></span>[23] NVIDIA Corporation. InfiniBand, 2020. [https://www.nvidia.com/en-us/networking/](https://www.nvidia.com/en-us/networking/products/infiniband/) [products/infiniband/](https://www.nvidia.com/en-us/networking/products/infiniband/).
- <span id="page-12-9"></span>[24] NVIDIA Corporation. NVIDIA H100 GPU, 2022. [https://www.nvidia.com/en-us/](https://www.nvidia.com/en-us/data-center/h100/) [data-center/h100/](https://www.nvidia.com/en-us/data-center/h100/).
- <span id="page-12-14"></span>[25] NVIDIA Corporation. Using FP8 with Transformer Engine, 2024. [https:](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) [//docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) [fp8\\_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
- <span id="page-12-8"></span>[26] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixtureof-experts inference and training to power next-generation ai scale. *ArXiv*, abs/2201.05596, 2022.

- <span id="page-13-5"></span>[27] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pages 1–16, 2019.
- <span id="page-13-0"></span>[28] Zhihong Shao, Damai Dai, Daya Guo, Bo Liu (Benjamin Liu), and Zihan Wang. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. *ArXiv*, abs/2405.04434, 2024.
- <span id="page-13-6"></span>[29] Noam M. Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer. *ArXiv*, abs/1701.06538, 2017.
- <span id="page-13-9"></span>[30] Liang Shen, Zhihua Wu, Weibao Gong, Hongxiang Hao, Yangfan Bai, Huachao Wu, Xinxuan Wu, Haoyi Xiong, Dianhai Yu, and Yanjun Ma. Moesys: A distributed and efficient mixtureof-experts training and inference system for internet services. *IEEE Transactions on Services Computing*, 17:2626–2639, 2022.
- <span id="page-13-4"></span>[31] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *ArXiv*, abs/1909.08053, 2019.
- <span id="page-13-10"></span>[32] Siddharth Singh, Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He, and Abhinav Bhatele. A hybrid tensor-expert-data parallelism approach to optimize mixture-ofexperts training. In *Proceedings of the 37th ACM International Conference on Supercomputing*, ICS '23, page 203–214, New York, NY, USA, 2023. Association for Computing Machinery.
- <span id="page-13-3"></span>[33] Snowflake AI Research Team. Snowflake arctic: The best llm for enterprise ai — efficiently intelligent, truly open, 2024. Accessed: December 11, 2024.
- <span id="page-13-11"></span>[34] TOP500. Eos, 2024. <https://www.top500.org/system/180239/>.
- <span id="page-13-8"></span>[35] L G Valiant. A bridging model for parallel computation. *Communications of the ACM*, 1990.
- <span id="page-13-7"></span>[36] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Neural Information Processing Systems*, 2017.
- <span id="page-13-2"></span>[37] xAI. Grok-1. GitHub repository, 2024. Accessed: December 11, 2024.
- <span id="page-13-1"></span>[38] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Ke-Yang Chen, Kexin Yang, Mei Li, Min Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yunyang Wan, Yunfei Chu, Zeyu Cui, Zhenru Zhang, and Zhi-Wei Fan. Qwen2 technical report. *ArXiv*, abs/2407.10671, 2024.
- <span id="page-13-12"></span>[39] Yanli Zhao, Andrew Gu, Rohan Varma, Liangchen Luo, Chien chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel. *Proc. VLDB Endow.*, 16:3848–3860, 2023.

# 6 Appendix

#### 6.1 Accuracy Validation

To validate the accuracy of our implementation, we train a Mixtral 8x7B model with MoE Parallel Folding compared to MCore v0.9 in a token-dropless manner up to 40B tokens. We set TP=2, CP=2, PP=2, EP=8, ETP=1, this allows us to verify the correctness of MoE Parallel Folding where EP in MoE layer are folded with all of TP,CP,DP in Attention. As shown in Figures [7](#page-14-0) and [8,](#page-14-1) MCore with MoE Parallel Folding is able to successfully train the model to convergence, and the training and validation loss curves align well with MCore v0.9.

<span id="page-14-0"></span>![](_page_14_Figure_3.jpeg)

Figure 7: Training loss of MCore with MoE Parallel Folding compared to MCore v0.9.

<span id="page-14-1"></span>![](_page_14_Figure_5.jpeg)

Figure 8: Validation loss of MCore with MoE Parallel Folding compared to MCore v0.9.

#### 6.2 Workflow for Transformer Layer with MoE parallel folding

Figure [9](#page-15-0) illustrates the overall workflow of a Transformer Layer in an MoE model with Parallel Folding. In the Attention component, the parallelism mapping is TP2CP2DP2, where each sequence is split across 4 GPUs. For the MoE layer, the parallelism mapping is TP1EP8DP1, with each GPU handling a different expert FFN. The transformation between the Attention and MoE layers requires only a reshape operation that flattens the sequence/subsequence into a batch of tokens, introducing no explicit communication overhead.

<span id="page-15-0"></span>![](_page_15_Figure_2.jpeg)

Figure 9: Workflow of the Transformer layer with MoE Parallel Folding.

#### 6.3 Parallel Groups Initialization

In the code of Listing [1,](#page-16-0) we give an example to show how the parallelism groups with MoE Parallel Folding for each device are generated. The code demonstrates the initialization of parallel groups for both attention and MoE components, handling different parallelism dimensions including TP, EP, PP and DP.

The function generate\_mappings takes the total number of devices (world\_size) and parallelism dimensions as input parameters. It first calculates the effective data parallelism degrees for attention and MoE components separately. Then, it creates two sets of parallel groups: one for attention layers with TP, CP, PP, and DP dimensions, and another for MoE layers with TP, EP, PP, and DP dimensions.

```
1 from einops import rearrange
2 import torch
4 def generate_mappings ( world_size , tp , cp , ep , etp , pp ) :
5 ranks = torch . arange ( world_size )
6 attn_dp = world_size // tp // cp // pp
7 moe_dp = world_size // etp // ep // pp
8
9 # Parallel groups for attention
10 attn_ranks = ranks . reshape ( attn_dp , pp , cp , tp )
11 attention_groups = {
12 "TP": rearrange ( attn_ranks , " attn_dp pp cp tp -> ( attn_dp pp
     cp) tp",
13 tp =tp , cp = cp , pp = pp , attn_dp = attn_dp ) . tolist () ,
14 "CP": rearrange ( attn_ranks , " attn_dp pp cp tp -> ( attn_dp pp
     tp) cp",
15 tp =tp , cp = cp , pp = pp , attn_dp = attn_dp ) . tolist () ,
16 "PP": rearrange ( attn_ranks , " attn_dp pp cp tp -> ( attn_dp cp
     tp) pp",
17 tp =tp , cp = cp , pp = pp , attn_dp = attn_dp ) . tolist () ,
18 "DP": rearrange ( attn_ranks , " attn_dp pp cp tp -> (pp cp tp)
     attn_dp ",
19 tp =tp , cp = cp , pp = pp , attn_dp = attn_dp ) . tolist ()
20 }
22 # Parallel groups for MoE
23 moe_ranks = ranks . reshape ( moe_dp , pp , ep , tp )
24 moe_groups = {
25 "TP": rearrange ( moe_ranks , " moe_dp pp ep tp -> ( moe_dp pp ep)
     tp",
26 tp = etp , ep = ep , pp = pp , moe_dp = moe_dp ) . tolist () ,
27 "EP": rearrange ( moe_ranks , " moe_dp pp ep tp -> ( moe_dp pp tp)
     ep",
28 tp = etp , ep = ep , pp = pp , moe_dp = moe_dp ) . tolist () ,
29 "PP": rearrange ( moe_ranks , " moe_dp pp ep tp -> ( moe_dp ep tp)
     pp",
30 tp = etp , ep = ep , pp = pp , moe_dp = moe_dp ) . tolist () ,
31 "DP": rearrange ( moe_ranks , " moe_dp pp ep tp -> (pp ep tp)
     moe_dp ",
32 tp = etp , ep = ep , pp = pp , moe_dp = moe_dp ) . tolist ()
33 }
34
35 return attention_groups , moe_groups
36
37 attn_groups , moe_groups = generate_mappings (64 , 2 , 2 , 2 , 2 , 2)
```

Listing 1: Python implementation of parallel group generation for MoE Parallel Folding

#### 6.4 Details of Parallelism Mappings in Experiments

We conducted numerous experiments to find the optimal training parallel configurations. The optimal settings found and their corresponding performance metrics are presented in Table [3.](#page-17-0) In these experiments, the global batch size was fixed at 256, and the sequence length was fixed at 4096.

To investigate the scalability of various methods, we fixed the parallel configuration and increased the number of GPUs. The detailed benchmark numbers are presented in Table [4.](#page-17-1) All parallel configurations are the same as those identified in the performance experiments.

In the context scaling experiment, influenced by the long sequence length, the optimal parallel configuration might differ. The parallel configurations found and the detailed performance results are presented in Table [5.](#page-19-0)

Table 3: Detailed parallel mapping of models with optimal configurations.

<span id="page-17-0"></span>

| Model              | Methods          | GPUs | CP | TP | EP | PP | ETP | MFU   |
|--------------------|------------------|------|----|----|----|----|-----|-------|
|                    | FSDP             | 128  | 1  | 8  |    |    |     | 4.3%  |
|                    | FSDP + EP        | 128  | 1  | 2  | 8  |    |     | 23.4% |
| Mixtral-8x22B      | TP + EP + DP     | 128  | 1  | 4  | 8  |    |     | 36.6% |
|                    | MCore            | 128  | 1  | 2  | 4  | 8  |     | 46.3% |
|                    | MCore w/ Folding | 128  | 1  | 2  | 8  | 8  | 1   | 49.3% |
|                    | FSDP             | 64   | 1  | 2  | 1  |    |     | 9.9%  |
|                    | FSDP + EP        | 64   | 1  | 1  | 8  |    |     | 25.4% |
| Qwen2-57B-A14B     | TP + EP + DP     | 64   | 1  | 4  | 4  |    |     | 23.1% |
|                    | MCore            | 64   | 1  | 2  | 4  | 4  |     | 35.3% |
|                    | MCore w/ Folding | 64   | 1  | 2  | 4  | 4  | 1   | 39.0% |
|                    | FSDP             | 128  | 1  | 8  | 1  |    |     | 2.2%  |
|                    | FSDP + EP        | 128  | 1  | 4  | 8  |    |     | 9.0%  |
| Mixtral-8x22B-G8T8 | TP + EP + DP     | 128  | 1  | 8  | 8  |    |     | 8.7%  |
|                    | MCore            | 128  | 1  | 2  | 8  | 8  |     | 17.1% |
|                    | MCore w/ Folding | 128  | 1  | 4  | 8  | 8  | 1   | 28.8% |
|                    | FSDP             | 256  | 8  | 8  | 1  |    |     | OOM   |
|                    | FSDP + EP        | 256  | 1  | 8  | 8  |    |     | 19.6% |
| Llama3-8x70B       | TP + EP + DP     | 256  | 1  | 8  | 8  |    |     | OOM   |
|                    | MCore            | 256  | 1  | 8  | 4  | 8  |     | 38.8% |
|                    | MCore w/ Folding | 256  | 1  | 8  | 8  | 16 |     | 41.6% |

<span id="page-17-1"></span>Table 4: The detailed parallel mapping of scaling experiments for the numbers of GPUs

| Model         | Methods          | GPUs | MFU   |
|---------------|------------------|------|-------|
|               |                  | 128  | 49.4% |
|               | MCore            | 256  | 48.0% |
|               |                  | 512  | 45.5% |
|               |                  | 1024 | 42.3% |
|               |                  | 128  | 52.2% |
|               |                  | 256  | 50.7% |
|               | MCore w/ Folding | 512  | 48.9% |
| Mixtral 8x22B |                  | 1024 | 44.9% |
|               |                  | 128  | 23.9% |
|               |                  | 256  | 25.5% |
|               | FSDP + EP        | 512  | 24.4% |
|               |                  | 1024 | 23.8% |
|               |                  | 128  | 40.4% |
|               | TP + EP + DP     | 256  | 39.1% |

| Model              | Methods          | GPUs        | MFU            |
|--------------------|------------------|-------------|----------------|
|                    |                  | 512         | 36.0%          |
|                    |                  | 1024        | 36.2%          |
|                    |                  | 64          | 36.2%          |
|                    | MCore            | 128         | 36.0%          |
|                    |                  | 256         | 34.8%          |
|                    |                  | 512<br>1024 | 32.5%<br>29.8% |
|                    |                  | 64          | 39.9%          |
|                    | MCore w/ Folding | 128         | 39.7%          |
|                    |                  | 256         | 38.1%          |
|                    |                  | 512         | 36.6%          |
|                    |                  | 1024        | 33.4%          |
|                    |                  | 64          | 26.3%          |
| Qwen2 57B-A14B     | FSDP + EP        | 128         | 25.6%          |
|                    |                  | 256<br>512  | 23.4%<br>22.8% |
|                    |                  | 1024        | 21.6%          |
|                    |                  | 64          | 22.4 %         |
|                    | TP + EP + DP     | 128         | 20.9 %         |
|                    |                  | 256         | 19.8 %         |
|                    |                  | 512         | 19.7 %         |
|                    |                  | 1024        | 17.9 %         |
|                    |                  | 128         | 19.8%          |
|                    | MCore            | 256<br>512  | 18,4%<br>16.3% |
|                    |                  | 1024        | 13.4%          |
|                    |                  | 128         | 30.0%          |
|                    | MCore w/ Folding | 256         | 29.3%          |
|                    |                  | 512         | 26.7%          |
| Mixtral 8x22B G8T8 |                  | 1024        | 25.5%          |
|                    |                  | 128         | 9.0%           |
|                    | FSDP + EP        | 256         | 8.5%           |
|                    |                  | 512         | 8.9%           |
|                    |                  | 1024        | 8.6%           |
|                    |                  | 128<br>256  | 8.6%<br>8.5%   |
|                    | TP + EP + DP     | 512         | 8.5%           |
|                    |                  | 1024        | 8.1%           |
|                    |                  | 256         | 40.1%          |
|                    | MCore            | 512         | 39.5%          |
|                    |                  | 1024        | 39.1%          |
| Llama3 8x70B       |                  | 128         | 43.7%          |
|                    | MCore w/ Folding | 512         | 42.7%          |
|                    |                  | 1024        | 41.5%          |
|                    |                  | 128         | 18.9%          |
|                    | FSDP + EP        | 512<br>1024 | 17.3%<br>17.1% |
|                    |                  |             |                |

Table 5: The performance of scaling experiments for the sequence length

<span id="page-19-0"></span>

| model          | methods          | #GPUs | SeqLen | CP | TP | EP | PP | ETP | GBS  | MFU      |
|----------------|------------------|-------|--------|----|----|----|----|-----|------|----------|
|                |                  | 128   | 16384  | 4  | 2  | 4  | 8  |     | 1024 | 45.30%   |
|                |                  | 256   | 32768  | 8  | 2  | 4  | 8  |     | 512  | 43.20%   |
|                | Mcore            | 512   | 65536  | 16 | 2  | 4  | 8  |     | 256  | 42.60%   |
| Mixtral-8x22B  |                  | 1024  | 131072 | 16 | 4  | 8  | 8  |     | 128  | 38.20%   |
|                | Mcore w/ Folding | 128   | 16384  | 4  | 2  | 8  | 8  | 1   | 1024 | 47.60%   |
|                |                  | 256   | 32768  | 8  | 2  | 8  | 8  | 1   | 512  | 45.10%   |
|                |                  | 512   | 65536  | 8  | 4  | 8  | 8  | 1   | 256  | 44.50%   |
|                |                  | 1024  | 131072 | 8  | 8  | 8  | 8  | 1   | 128  | 42.90%   |
|                | Mcore            | 128   | 16384  | 4  | 2  | 4  | 8  |     | 1024 | 45.30%   |
|                |                  | 256   | 32768  | 8  | 2  | 4  | 8  |     | 512  | 43.20%   |
| Qwen2-57B-A14B |                  | 512   | 65536  | 16 | 2  | 4  | 8  |     | 256  | 42.60%   |
|                |                  | 1024  | 131072 | 16 | 4  | 8  | 8  |     | 128  | 38.20%   |
|                | Mcore w/ Folding | 128   | 16384  | 4  | 2  | 8  | 8  | 1   | 1024 | 47.60%   |
|                |                  | 256   | 32768  | 8  | 2  | 8  | 8  | 1   | 512  | 45.10%   |
|                |                  | 512   | 65536  | 8  | 4  | 8  | 8  | 1   | 256  | 44.50%   |
|                |                  | 1024  | 131072 | 8  | 8  | 8  | 8  | 1   | 128  | 42.90% . |