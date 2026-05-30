# <span id="page-3-0"></span>3.3 Expensive All-to-All on HPC Platform with Heterogeneous and Hierarchical Network

Most prior MoE training systems were designed for clusters composed of NVIDIA DGX nodes [16, 31], where nodes are connected via high-bandwidth, low-latency InfiniBand. These clusters exhibit relatively balanced GPU-to-GPU inter/intra-node communication bandwidths, with intra-node bandwidths only 3× faster than internode bandwidths. As a result, existing MoE systems often take advantage of this balanced network and treat all GPUs in a cluster equivalently. For example, in DeepSpeed-MoE [31], each collective involves all GPUs within an expert-parallel group, regardless of physical placement.

However, many HPC platforms differ from DGX-style clusters. For example, Frontier [5] adopts a Dragonfly topology, where GPUs within a node are connected via Infinity Fabric (up to 200 GB/s) while inter-node communication happens via Slingshot (25 GB/s). This introduces a significant bandwidth asymmetry. In such hierarchical interconnects, network-aware communication is essential. Unfortunately, existing MoE systems do not exploit this hierarchy and instead route tokens indiscriminately, often leading to sub-optimal bandwidth utilization.

This problem is exacerbated in expert-specialized MoEs, which rely on large top-k routing where each token is sent to a large number of experts. If multiple selected experts reside on the same node, existing systems send multiple copies of the same token activation across inter-node links, one for each destination expert, even though only one copy is actually needed (§ 4.2). To quantify this redundancy, we evaluate a DeepSeek-style configuration (256 experts, k=8 routing) using DeepSpeed-MoE. Fig. 4 shows that the redundancy rate ranges can be up to 75.1%, depending on the EP size. This leads to redundant inter-node communication.

<span id="page-3-4"></span>![](_page_3_Figure_12.jpeg)

Figure 4: Redundancy rate of all dispatched tokens.

**Takeaway-3:** Expert-specialized MoEs increase the number of routed expert per token, leading to significant duplication in communication. On HPC platforms with hierarchical and heterogeneous networks, this results in inefficient use of inter-node bandwidth and causes communication to become a major training efficiency bottleneck as the expert granularity increases.

## <span id="page-3-5"></span>4 X-MoE Design

To address the training inefficiencies of emerging MoE architectures on non-NVIDIA platforms, X-MoE introduces a set of system optimization techniques to address the unique challenges posed by fine-grained experts with large top-k routing on hierarchical HPC platforms. Fig. 5 provides an overview of X-MoE. First, we propose a new sparse data layout PFT and rework the MoE pipeline to eliminate zero-padding across different MoE stages via padding-free sparse MoE training with cross-platform kernels (§ 4.1). Second, we reduce communication redundancy by leveraging topological awareness of HPC systems via a hierarchical two-stage redundancybypassing dispatch algorithm (§ 4.2). Third, X-MoE incorporates a hybrid prallelism strategy that enables sequence-sharding in MoE blocks to address the activation memory bottleneck, with topologyaware planning and device-mapping. Together, these optimizations form an integrated and cross-platform training system that scales emerging MoEs on large scale HPC clusters. The following sections provide an in-depth description of each design.

## <span id="page-3-1"></span>4.1 Padding-Free Sparse MoE Training with Cross-Platform Kernels

We introduce the truly padding-free MoE training pipeline in X-MoE. First, we design a novel sparse data-structure: PFT (**P**adding-Free **T**oken buffers). The PFT is designed to eliminate zero padding through the MoE computation and communication stages, including

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 5: Overview of X-MoE. At a high level, X-MoE enables efficient and scalable training for expert-specialized MoEs through a set of targeted system-level optimizations, including a fully padding-free MoE training pipeline with cross-platform kernels (§ 4.1), redundancy-bypassing dispatch (§ 4.2), and hybrid parallelism with sequence-sharded MoE blocks (§ 4.3).

the dispatch, MLP, and combine stages (§ 4.1.1). Instead of allocating fixed-capacity expert buffers padded with zeroed-vectors, PFT stores only valid routed tokens. However, in introducing the PFT to MoE training, each of the dispatch, MLP, and combine stages needs to be modified to operate on the PFT. We also detail our modifications to each stage (§ 4.1.1). Second, to efficiently implement the PFT-based pipeline, we design a suite of Triton-based kernels to handle the corresponding sparse and irregular workloads (§ 4.1.2). These kernels are designed to be hardware-agnostic, support coalesced memory accesses and avoid vendor-specific constraints like CUDA-only fused kernels. As a result, our padding-free MoE training pipeline improves memory efficiency and reduces communication volume, which serve as key enablers of scalable training of expert-specialized MoEs on diverse hardware.

<span id="page-4-1"></span>4.1.1 Padding-Free Token Storage and Pipeline To eliminate the inefficiencies introduced by zero-padding in existing MoE pipelines, we introduce the PFT data-structure. Unlike standard expert input buffers that reserve fixed-capacity slots per expert, PFT consists of a token-buffer, x, which stores only the routed tokens, along with expert routing information arrays (ERI-arrays) that track how each token should be processed. The ERI-arrays consists of the following data: (1) array token\_ids (a [B]-sized array; B is the number of routed tokens in x) where ti = token\_ids[i] is an index that maps the  $ti^{th}$  input-token to the  $i^{th}$  position in the dispatch matrix (see figure Fig. 6 for an example), (2) array expert\_ids (a [B]-sized array) where expert\_ids[i] represents the expert that x[i] is routed to, (3) array tokens\_per\_expert (a [E]-sized array; E is the expert count) where tokens\_per\_expert[i] represents the number of tokens in x routed to expert i, (4) array combine\_weights (a [B]-sized array) where combine\_weights[i] represents the value that combinein[i] (an intermediate matrix assembled after the last alltoall, described later in § 4.1.1) is scaled by in the combine phase. We first show how PFT is constructed and then demonstrate how this representation allows each MoE stage to operate without any zero padding. Fig. 6 depicts the PFT structure with ERI-arrays and how ERI-arrays drive token dispatching.

**PFT Construction.** PFT is constructed after the MoE gating function and before token dispatching (we describe how the gating, dispatch, MLP and combine stages are modified later). Listing 1 illustrates the pseudo code for PFT construction. It takes as input the outputs of the gating function: (1) the top\_experts array, a [S,

<span id="page-4-2"></span>![](_page_4_Figure_6.jpeg)

Figure 6: PFT with sparse structure and ERI-arrays.

E]-sized array that contains the token to expert mapping and (2) corresponding combine\_weights array, a [S, E]-sized array used in the combine stage containing each token's probability score that reflects the confidence of the gating function. It also takes as input the max\_token\_count variable, indicating the expert-capacity. The PFT construction routine returns a PFT whose ERI-arrays are correctly instantiated.

The PFT construction routine proceeds in two stages. In the first stage, we flatten and sort the incoming top\_experts array (lines 20-21), which contains the token to expert assignments generated by the gating function. In the second stage, we determine which tokens are dropped (lines 24-33); using this information, we construct the ERI-arrays by pruning out the dropped tokens from the unfiltered token\_ids, expert\_ids, and combine\_weights (lines 34-36).

Padding-free Gating, Dispatch, MLP and Combine. Listing 1 (lines 67-72) illustrate our padding-free pipeline where the modified dispatch, MLP and combine stages operate on the PFT. First, during gating, we transform the input tokens to logits and select the top-k experts per token, returning their respective expert indices (top\_expert) and probability confidence scores of the assignment (combine\_weights) (lines 6-8). Second, we construct the PFT structure using these data. Third, during dispatch, we consume the pft and gateout tokens and route tokens to the correct worker. This occurs by: (1) reordering the tokens locally using a custom gatherkernel (described in § 4.1.2) producing the dispatch<sub>in</sub> buffer, (2) exchanging the tokens between devices via an uneven alltoall, routing each token to the correct device its expert resides on (lines 43-47) producing the dispatchout buffer. No zero-padding is communicated in this stage. Fourth, the MLP layer processes tokens; we launch a custom sequential-GeMM (described in § 4.1.2) to

```
1 def gating(k, tokens):
       tokens: input tokens to MoE-layer [S, H]-sized.
       logits = softmax(FFN(tokens), axis=-1)
       combine_weights, top_experts = topk(logits)
return top_expert, combine_weights, tokens
10 \ \mathsf{def} \ \mathsf{PFT\_construction} \\ (\mathsf{max\_token\_count}\,, \mathsf{top\_experts}\,,
        combine_weights):
11
12
       top_experts: token to expert assignments [S, E]-sized combine weights: gating probabilities [S, E]-sized
13
       combine_weights: gating probabilities [S, E]-sized
max_token_count: maximum tokens for a single expert
15
16
       E = get_expert_count()
       ## Step 1: Generate the expert & token ids ##
# shape (lines 20-21): [S*K]
18
       flat_top_experts = flatten(top_experts)
20
       ## Step 2: Identify filter dropped-tokens ##

# shape (lines 24-26): [S*K]

flat_combine_weights = flatten(combine_weights)
21
22
23
24
25
       sorted_indices = argsort(flat_combine_weights)
sorted_top_experts = flat_top_experts[sorted_indices]
27
                    (lines 28-30)
       one_hot_enc=one_hot(sorted_top_experts,num_classes=E)
29
       rank_in_expert = cumsum(one_hot_enc, axis=0)
30
       weight_mask = rank_in_expert <= max_token_count</pre>
                                                     token-count
       # shape (lines 32-36): [B], token-count post d
filtered_indices = sorted_indices[weight_mask]
retained_token_ids = isin(flat_top_experts,
                                                                                dropping
32
33
        filtered_indices)
       token_ids = token_ids[retained_token_ids]\nexpert_ids = expert_ids[retained_token_ids]
combine_weights = combine_weights[retained_token_ids]
34
35
36
37
38
       tokens_per_expert = histogram(expert_ids, bins=E)
       # Return PFT(token_ids,expert_ids,tokens_per_expert,
39
40
        combine_weights)
41
       er dispatch(pft, gate<sub>out</sub>):
dispatch<sub>in</sub> = gather_kernel(gate<sub>out</sub>,pft.token_ids,pft.
expert_ids)
   def dispatch(pft.
43
44
       pft.tokens_per_expert=alltoall(pft.tokens_per_expert)
45
       dispatch<sub>out</sub> = alltoallv(dispatch<sub>in</sub>, pft.
tokens_per_expert)
46
       pft.x = dispatch_{out}
       return pft
48
49
   def mlp(pft. w1. w2):
       inter_activ = sequential_gemm(pft.x,w1)
51
       {\rm mlp}_{out} = {\rm sequential\_gemm(inter\_activ,w2)} pft.x = {\rm mlp}_{out}
52
53
54
       return pft
55
   def combine(pft):
       combine_{in} = alltoallv(pft.x,pft.tokens_per_expert) combine_{out} = scatter_kernel(combine_{in},pft.token_ids,
56
57
        pft.expert_ids,pft.combine_weights)
58
       return combineout
59
60
   def call(tokens, k, max_token_count, w1, w2):
61
       tokens: tokens input to the MoE layer, [S. H]-sized
62
       k: topk value, int.
max_token_count: expert capacity, int
64
       w1 & w2: weights of first and second layer of MLP.
       \label{eq:continuous} \begin{array}{ll} \texttt{top\_expert}\,, \texttt{combine\_weights}\,, \texttt{gate}_{out} = \texttt{gating}\,(\texttt{k}\,, \texttt{tokens})\\ \texttt{pft} = \texttt{PFT\_construction}\,(\texttt{max\_token\_count}\,, \texttt{top\_expert}\,, \end{array}
67
68
        combine_weights)
69
       pft = dispatch(pft,gate<sub>out</sub>)
70
       pft = mlp(pft,w1,w2)
       pft = combine(pft)
       return pft.x
```

Listing 1: Padding-free MoE-layer

implement each MLP layer enabling different token-counts to be multiplied by the MLP weights of different experts without the need for zero-padding (lines 50-53). Fifth, during combine, tokens are re-routed back to their original device and a custom scatter kernel (described in § 4.1.2) locally reorders the inbound tokens to their

<span id="page-5-3"></span>![](_page_5_Figure_4.jpeg)

Figure 7: Hierarchical Redundancy-Bypassing Dispatch: Multi-stage token routing across inter- and intra-node networks to reduce communication duplication.

original position in the sequence, multiplying each token with its respective value in combine\_weights (lines 56-58).

<span id="page-5-1"></span>4.1.2 Highly-Optimized Cross-Platform Sparse and Irregular Kernels The PFT format helps improve the memory-efficiency of emerging MoE training by eliminating the need for any zero-padding in the dispatch, MLP and combine stages. However, certain operators in this modified pipeline, such as the gather, scatter and sequential GeMM, introduce sparse and irregular access patterns to the PFT ERI-arrays, which can be expensive to implement in Pytorch and requires specialized kernels for efficiency. To address these issues, we introduce Triton-based gather, scatter as well as (non Triton-based) sequential GeMM implementations that are high-performance and platform agnostic. The gather and scatter kernels are responsible for computing: dispatch<sub>in</sub>[i,:] = gate<sub>out</sub>[ $token_ids[i]$ ,:] and combine<sub>in</sub>[ $token_ids[i]$ ,:] = mlp<sub>out</sub>[i,:]

 $\times$  combine\_weights[token\_ids[i], :], respectively. However, the irregular memory access patterns in reading and writing to tensors results in uncoalesced memory requests and poses a unique performance challenge. We circumvent this by scheduling a single thread-block to operate (read and write) on one vector, assigning contiguous threads to operate on the model-hidden dimension (outer-dimensions of the gateout and combinein tensors). On the other hand, our sequential GeMM operates on the dispatchout matrix. It extracts the correct tokens each expert is assigned to in dispatchout with a python for-loop launching  $E_{local}$  (number of experts assigned to the device) GeMMs.

#### <span id="page-5-0"></span>4.2 Redundancy-Bypassing Dispatch

We propose Hierarchical Redundancy-Bypassing Dispatch (RBD) to eliminate redundant inter-node communication by introducing a multi-stage dispatching process with two groups of tokens: *Pilot tokens*, which are the minimal set of distinct tokens that must be communicated across nodes; and *local replica*, which are local duplicates of pilot tokens routed to additional experts on the same destination node. Instead of sending all token data through one alltoall, RBD only sends pilot tokens through inter-node communication and propagates local replica using fast intra-node connects. We now illustrate RBD's multi-stage process using Fig. 7.

Stage 0 (S0): Pilot Selection and Instantiation. The first step of RBD is pilot tokens selection within x, which is generated through PFT in  $\S$  4.1. Based on token\_ids and expert\_ids, RBD extracts the node destination information for each token. Then for each token's k destinations, RBD identifies the group of experts that share the same destination node. Among tokens with the same

source and destination node, RBD randomly selects one as the pilot token and marks the rest as local replica. This randomized strategy helps avoid a biased distribution and creates a balanced workload for alltoall communication. For example, always routing tokens to the smallest expert ID within a node will significantly increase the alltoall latency.

Meanwhile, we create separate ERI-arrays for pilot tokens and local replicas, respectively. Each of them contains the routing information for those tokens. This process is represented by ① in Fig. 7. In addition, we construct a mapping array s1\_mapping\_indices (used in Stage 1 for local replica reconstruction), where each local replica token records the index of its corresponding pilot token. To ensure the correctness of this mapping index before and after the uneven alltoall exchange (④), we use the relative index starting from 0 for each target expert. This is allowed because the pilot ERI-arrays is sorted by expert IDs. We re-encode it to the absolute index after the alltoall exchange. Finally, we instantiate the pilot token buffer (④) using a Triton gather kernel. The local replica tokens are not instantiated yet. They are reconstructed from their associated pilot tokens after the pilot tokens arrive their destination.

Stage 1 (S1): Inter-Node Exchange (Pilot Only) and Local Replica Reconstruction. In S1, RBD sends only pilot tokens across nodes using an uneven alltoall (❸). This is the only stage that uses inter-node bandwidth. Additionally, RBD also sends local replica tokens' metadata (ERI-arrays and s1\_mapping\_indices) (④), alongside their corresponding pilot tokens. This is lightweight given that metadata has small message size. Once the pilot tokens arrive at their destination node, local replica tokens are reconstructed by copying data from pilot tokens to a local exchange buffer based on s1\_mapping\_indices (⑤). Note that the local exchange buffer serves as the input of the intra-node alltoall, RBD ensures token data is contiguous and ordered by destination (e.g., the ascending order of expert IDs).

Stage 2 (S2): Intra-Node Exchange (Local Replica Only) and Expert Input Reconstruction. The newly reconstructed local replica tokens are exchanged among GPUs within the same node using a fast intra-node uneven alltoall **6**, which helps to save expensive inter-node traffic. After pilot tokens and local replica tokens all arrive at their target GPUs, RBD reconstructs the each expert's local input by merging the two groups and correctly orders them based on their expert indices.

The combine stage reverses the above described RBD process. Specifically, local replica tokens are first gathered via intra-node communication, followed by pilot tokens through inter-node transfer. To ensure the correctness of combining weight scaling on expert outputs, we exchange the original combine\_weights ERI-array for all tokens in advance through a small inter-node alltoall, along with ②. During combine, the weight scaling is performed in *stage 1*, before merging local replica tokens into pilot tokens. Finally, the full results are reconstructed from the pilot tokens on the original device using the ERI-arrays preserved during dispatch.

## <span id="page-6-0"></span>4.3 Hybrid Parallelism with Sequence-Sharded MoE Blocks

Training MoEs at scale requires hybrid parallelism that carefully balances memory, compute, and communication. A common

approach is to apply tensor parallelism (TP) to dense blocks (e.g., attention, non-MoE MLPs) as in Megatron-LM [26] and switch to expert-parallelism (EP) for MoE blocks, sharding expert weights across devices. This TP + EP combination, used in systems such as DeepSpeed-TED [34], allows scaling conventional MoE parameters across large clusters. However, the naive transition from TP to EP fails to address the key bottleneck in expert-specialized MoEs: the activation memory, especially for  $A_{\rm dispatch}$  and  $A_{\rm combine}$ . As described in § 3.2, these tensors scale linearly with sequence length s, routing factor k, hidden dimension h, and fine-grained factor m.

Tensor parallelism works by duplicating input tokens across all TP ranks and computing partial results, which are later reduced via all-reduce. In the context of MoE training, this means that each TP worker holds a full copy of the input sequence, and even if we switch to EP within the MoE block, each EP worker begins the MoE computation with the same *duplicated activations*. Consequently, the heavy activations ( $A_{dispatch}$  and  $A_{combine}$  as described in § 3.2) are not reduced at all, as they are still duplicated across all EP workers that originate from TP ranks, which leads to poor scaling.

To address this challenge, we propose a new hybrid parallelism strategy that combines tensor-slicing parallelism with sequence-sharded execution for MoE Block (SSMB). This strategy is motivated by a key insight: all operations in an MoE block (gating, dispatch, expert FNNs, and combine) are applied token-wise and do not require inter-token dependencies. This allows us to shard the input sequence of the MoE block across EP ranks, so each rank only retains and processes a segment of the sequence and later recover the full sequence using all-gather, reintroducing the duplicated inputs expected by the next TP block. This strategy reduces the activation footprint of  $A_{\rm dispatch}$  and  $A_{\rm combine}$  by a factor of the TP group size, while preserving compatibility with standard MoE routing and communication.

Fig. 8 illustrates how SSMB works in practice. In this setup, we use TP=2 and DP=2 (TP and DP parallel-group size) for the dense (non-MoE) block, and EP=4 for the MoE blocks. In the TP + DP phase, each TP worker holds a full copy of the input sequence: device 0 and 1 each have a copy of sequence  $A_0$ , while device 2 and 3 have  $A_1$ . In existing MoE training, duplicated activations like  $A_{\text{dispatch}}$  and  $A_{\text{combine}}$  would be store on both devices, increasing memory cost. Instead, SSMB drops a fraction of the tokens on each device ( $\mathbf{0}$ ), partitioning the sequence across TP ranks (e.g.,  $A_0^0$  and  $A_0^1$ ). After entering the MoE block, SSMB reassigns each TP+DP worker to act as an EP rank and performs MoE-Gating, dispatch, expert FNNs, and combine on the partitioned tokens (2), using the padding-free pipeline introduced in § 4.1. After the combine op, SSMB issues an all-gather (3) to reconstruct the full output sequence (e.g.,  $A'_0$ ) across all EP ranks, effectively restoring the replicated data layout for the next TP-based non-MoE block.

In the backward pass, SSMB follows a reversed sequence of operations. Upon entering the MoE block, it first drops the gradients corresponding to the partial sequences retained during forward. It then performs expert-specific gradient computation and alltoall communications, mirroring the forward process. Finally, SSMB uses an all-gather operation to reconstruct the full input gradient across TP ranks, allowing the propagation to continue in the TP phase.

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

Figure 8: Illustration of X-MoE's hybrid parallelism with sequence-sharded MoE blocks.

Can existing parallel strategies handle the shifted memory bottleneck? The careful reader may think of an alternative approach that uses TP + EP for MoE blocks, as opposed to EP with sequence sharding. After all, these schemes also shard model states across devices and reduce memory. However, in expert-specialized MoEs, experts already have small intermediate dimensions, making TP's benefits marginal. Moreover, neither TP nor ZeRO-style DP reduce the expensive activations like  $A_{\tt dispatch}$  and  $A_{\tt combine}$ . We compare the overhead and gain of using SSMB with tensor-expert-data (TED) parallelism by calculating both model states and activation memory saving of each approach. In short, the benefit of SSMB over TED depends on the ratio:  $r = \frac{k}{H_{FFN}}$ . Under the usage of ZeRO-1 DP, when this ratio satisfies:  $r > \frac{2}{c \cdot S}$ , SSMB offers more memory savings than TED. For expert-specialized MoEs with fine-grained factor m, we have  $H_{FFN} \propto \frac{1}{m}$  and  $k \propto m$ . Thus, generally speaking, under identical sequence length choice, the more fine-grained the MoE model is, the more benefits SSMB provides over TED.

Why not activation checkpointing? Another approach to reducing activation memory is activation checkpointing [8], which trades memory for recomputation. However, in MoE training with expert parallelism,  $A_{\text{dispatch}}$  and  $A_{\text{combine}}$  are outputs of alltoall communication during token routing. In regular MoE training, 4 alltoalls are needed per layer in each step. SSMB follows this, requiring 4 alltoalls per step. However, applying checkpointing to these tensors would require two extra alltoall communications during the backward pass, resulting in a total of 6 alltoalls per layer, which incurs expensive communication overhead in addition to the recomputation overhead.

Why not use pipeline parallelism (PP)? While PP is effective for reducing memory by splitting the model across devices, it requires significant code refactoring and careful scheduling to balance pipeline stages, especially with sparse MoE layers. In contrast, our solution requires minimal code changes. We leave the integration with PP as future work.

#### 5 Evaluation

In this section, we evaluate X-MoE in comparison with state-ofthe-art large-scale MoE training approaches, demonstrating that it achieves significantly improved training efficiency and scalability for emerging MoEs. We also show the impact of different technologies within X-MoE on performance.

#### 5.1 Evaluation Methodology

Hardware. We conduct evaluation on the *Frontier* supercomputer [5]. Each cluster node is equipped with 4×AMD MI250X GPUs with dual Graphics Compute Dies (GCDs) and one EPYC CPU. A GCD is viewed as an effective GPU. The 2 GCDs on the same MI250X are connected with Infinity Fabric with a peak bandwidth of 200 GB/s. The GCDs on different MI250X are connected with Infinity Fabric where the peak bandwidth ranges from 50-100 GB/s. The Frontier nodes are connected with four Slingshot 25 GB/s NICs. We use up to 128 nodes (1024 MI250X GCDs) for experiments.

**Evaluation setup.** We implement X-MoE in DeepSpeed [32], a widely used open-source DL training library. We include the implementation and environment details in Appendix. If not specified, we choose the maximum micro-batch size of power of 2 under the memory limitation and a global batch size of 1024. We choose the capacity factor c = 1.25 for all experiments, as suggested by [24].

#### <span id="page-7-2"></span>5.2 Main Results

We first demonstrate that X-MoE scales effectively across a wide range of expert-specialized MoE models. We use model configurations from DeepSeek-MoE [10], DeepSeek-v2 [25], and DeepSeek-v3 [11], as shown in Table 3. We compare X-MoE against three large-scale MoE training frameworks: DeepSpeed-MoE [31], DeepSpeed-TED [34], and Tutel [16] as baselines. For DeepSpeed-MoE and Tutel, we sweep EP size in {32/64/128/256} and ZeRO stages 1/2. For DeepSpeed-TED, we additionally sweep TP in {1, 2, 4, 8} and choose the best performing configuration.

<span id="page-7-1"></span>

| Models                     | Small           | Medium          | Large             | Super             |
|----------------------------|-----------------|-----------------|-------------------|-------------------|
| seq. length                | 2048            | 4096            | 4096              | 4096              |
| $H_{model}$                | 2048            | 5120            | 7168              | 7168              |
| $H_{FFN}$                  | 1408            | 1536            | 2048              | 2560              |
| num. experts               | 64              | 128             | 256               | 256               |
| top-k                      | 6               | 6               | 8                 | 8                 |
| num. layers                | 28              | 28              | 28                | 61                |
| Param.<br>Activated Param. | 10.1 B<br>1.3 B | 55.2 B<br>5.2 B | 201.4 B<br>11.5 B | 545.4 B<br>28.7 B |

Table 3: The model configs used for evaluation.

Trainability and throughput. We evaluate X-MoE on Small (10.1B), Medium (55.2B), and Large (201B) model configurations using 256 GPUs. As shown in Fig. 9, while existing systems such as DeepSpeed-MoE, DeepSpeed-TED, and Tutel run out of memory on medium and large models, X-MoE successfully enables their training, effectively changing the status from non-trainable to trainable under the same hardware budget. When multiple systems can train a model, e.g., on the Medium model, X-MoE achieves higher throughput, with 5.15x and 1.42x speedup over DeepSpeed-TED and Tutel respectively. X-MoE achieves these results through a set of targeted system-level optimizations. Its padding-free training pipeline eliminates zero-padding overhead in both memory and communication. RBD reduces communication redundancy in high top-k routing scenarios by minimizing cross-node token duplication. SSMB effectively mitigates the shifted memory bottleneck. Together, these innovations enable efficient and scalable training of emerging expert-specialized MoEs.

Pushing the model scale limit. X-MoE further enables the Super 545B model on 1024 GPUs, achieving an aggregated throughput of 10.44 PetaFLOPs while all prior systems fail due to OOM errors. At this scale, training becomes sensitive to system dynamics beyond memory and communication volume optimizations. On Frontier, we observe that scaling beyond 256 GPUs results in significantly higher alltoall latencies (> 10× higher than average), likely due to increased cross-rack communication and network congestion from concurrently running jobs on the shared cluster. Despite this, X-MoE successfully sustains large-scale MoE training across racks, demonstrating its robustness and extending the boundary of what is trainable on today's HPC clusters.

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Figure 9: Results on training Small, Medium, and Large models on 256 GPUs; training Super model on 1024 GPUs. The dashed lines show the aggregated throughput.

#### 5.3 Scalability Evaluation

We evaluate both weak and strong scaling to demonstrate that X-MoE not only enables training of expert-specialized MoEs but also scales efficiently with increasing compute resources. We compare against only Tutel, as it is the best performing baseline as shown in Fig. 9.

*Weak scaling.* To evaluate weak scaling behavior, we train the 10.1B *Small* model from 16 to 256 GPUs, proportionally increasing the global batch size from 256 to 4096. We use EP=8 and scale out via ZeRO-DP. Our results are in figure Fig. 10(a). The results show that X-MoE consistently maintains higher TFLOPs compared to Tutel with a comparatively smaller drop in throughput as the number of GPUs increases.

Strong scaling. We evaluate strong scaling using the 55.2B Medium model on 128, 256, 512, and 1024 GPUs, keeping the global batch size fixed at 2048. This setup tests how well the system reduces iteration time as more GPUs are used. Since DeepSpeed-MoE fails to run due to OOM errors, we compare X-MoE (EP=64) with Tutel (EP=128). Fig. 10b shows that Tutel cannot run on 128 GPUs even with EP=128, while X-MoE scales effectively and achieves lower iteration time as GPU count grows. At 1024 GPUs, both systems converge to similar performance, as the increasing alltoall latency (as in § 5.2) becomes the dominant bottleneck at this scale.

#### 5.4 Analysis Results

<span id="page-8-2"></span><span id="page-8-1"></span>![](_page_8_Figure_9.jpeg)

Figure 10: Scalability results. (a) Weak scaling results: Training the 10.1B MoE on 16-256 GPUs with increasing batch size. (b) Strong scaling result: Training the 55.2B MoE across 128, 256, 512, 1024 GPUs while fixing the batch size.

<span id="page-8-3"></span>![](_page_8_Figure_11.jpeg)

Figure 11: Forward MoE layer time breakdown comparison between DeepSpeed-MoE and X-MoE of training *Small* model and *Large* Model.

5.4.1 How does PFT and the padding-free pipeline bring benefits? We evaluate the benefits of PFT format and the associated padding-free pipeline in two dimensions: (1) reduced layer-wise execution time, and (2) improved memory efficiency.

MoE layer time breakdown. To evaluate the impact of PFT, we compare the MoE layer time breakdown of X-MoE and DeepSpeed-MoE when training the *Small* model (EP=8) and *Large* model (EP=64) on 256 GPUs. We disable other optimizations such as RBD to isolate the contribution of PFT. Fig. 11 shows the time comparison in an MoE layer: gating, buffer dispatching, dispatch alltoall, expert computation, combine alltoall, and buffer combining.

The latency reduction arises for different reasons in the *Small* and *Large* models. For the *Small* model, a major inefficiency of the baseline comes from the inefficient gating and buffer dispatch/buffer combine. X-MoE improves on these stages due to the PFT sparse structure and efficient Triton kernels. Specifically, the gating, buffer dispatch and buffer combine stages are accelerated by 5.7×, 35.7× and 8.1× respectively. Note that the expert computation time is slightly increased in X-MoE at this scale. The reason is that X-MoE applies a sequential GEMM on the uneven token buffer, which requires extra data transformations to get the expert input. Despite this overhead, the overall layer time is reduced by 62.3%. For the *Large* model, the largest latency reduction comes from the alltoall. X-MoE significantly reduces this time by 50.7% by eliminating zero-padding. The gating, buffer dispatch and combine time are also negligible after X-MoE's optimizations.

<span id="page-9-1"></span>![](_page_9_Figure_1.jpeg)

Figure 12: Dispatching time breakdown: With and without RBD under PFT-based dispatching.

Activation memory savings. We compare the per-layer activation memory usage of DeepSpeed-MoE, Tutel, and X-MoE when training the Large model on 256 GPUs, using EP=64 and ZeRO-style data parallelism. We report the maximum memory usage across all ranks. As shown in Table 4, X-MoE achieves significantly lower memory consumption than both DeepSpeed-MoE and Tutel, because it reduces memory wastage on dispatching metadata as well as the unused tokens through its padding-free pipeline. Besides, another reason for Tutel's high memory usage is that the Tutel kernel forces the use of float32 on Acombine on AMD GPUs.

<span id="page-9-0"></span>

|             | DS-MoE | Tutel | X-MoE | Theoretical |
|-------------|--------|-------|-------|-------------|
| Memory (GB) | 2.81   | 1.95  | 1.21  | 1.125       |

Table 4: The activation memory consumption per-MoE-layer.

5.4.2 How does RBD reduce MoE dispatching latency? We evaluate the performance impact of dispatching with and without RBD, with PFT format and padding-free pipeline enabled. We conduct the experiment using a single MoE layer from the *Large* model on 32 GPUs with EP=32. In this setting, the measured redundancy rate is 54.8%. Fig. 12 shows that inter-node alltoall communication (shadowed area) dominates the total dispatch time in the padding-free training pipeline. RBD reduces the inter-node communication time by 52.5% by bypassing redundant tokens transferred through the low-bandwidth inter-node links. Although RBD introduces extra overhead, such as intra-node alltoall (yellow) and data transformation costs, they are relatively minor compared to the savings from reduced inter-node data transfer volume, resulting in an overall performance speedup of 1.55x.

5.4.3 How does SSMB save memory? We evaluate the memory saving benefits of X-MoE by comparing X-MoE with SSMB enabled and X-MoE that uses the conventional tensor-expert-data parallelism (TP+EP+DP) without sequence sharding in MoE blocks, on the *Large* model across 256 GPUs. We enable ZeRO-1 DP and set EP=64 while varying the TP degree from 1 to 4. Fig. 13 shows that enabling SSMB leads to significantly lower memory usage, and the benefit grows as the TP degree increases. This is because SSMB shards sequences within MoE blocks, effectively addressing the shifted memory bottleneck in expert-specialized MoEs. As model size increases, the TP degree naturally grows for non-MoE blocks, making SSMB increasingly important for high memory efficiency for MoE training at scale.

5.4.4 How does SSMB compare to activation checkpointing? One may ask how SSMB compares to activation checkpointing, a technique that reduces memory usage. As shown in Fig. 14, under similar memory savings, X-MoE with SSMB achieves higher throughput. This is because SSMB reduces activation memory without the cost of recomputation and extra alltoall during backward pass.

<span id="page-9-2"></span>![](_page_9_Figure_9.jpeg)

Figure 13: X-MoE's maximum allocated memory across GPUs w/ and w/o SSMB.

Figure 14: TFLOPs of enabling SSMB vs. activation checkpointing.

#### 5.5 Cross-platform Performance

<span id="page-9-3"></span>

| TFLOPs                  | DeepSpeed-MoE | Tutel | X-MoE |
|-------------------------|---------------|-------|-------|
| Small (s=2048, l=28)    | OOM           | OOM   | 46.87 |
| Small-SR (s=1024, l=28) | 27.08         | 28.26 | 27.33 |
| Small-LR (s=2048, l=14) | 52.15         | 64.00 | 62.51 |

Table 5: TFLOPs comparison of DeepSpeed-MoE, Tutel and X-MoE on 8×NVIDIA A100 40GB GPUs. The Small model is the 10.1 B model listed in Table 3. "Small-SR" and "Small-LR" models refer to sequence length reduced (SR) or number of layers reduced (LR) while maintaining the other configurations unchanged.

To show X-MoE's portability and performance beyond the AMD GPU platform, we evaluate X-MoE on eight NVIDIA A100 40 GB GPUs and compare against DeepSpeed-MoE and Tutel (Table 5). We train the 10.1 B (Small in Table 3) model in this experiment. Under the full 2k sequence length and 28 layers, both competing frameworks encounter out-of-memory (OOM) failures, whereas X-MoE sustains training at 46.87 TFLOPS. To further show the throughput comparison, we reduce either the sequence length to 1k (Small-SR) or the depth to 14 layers (Small-LR). In these two settings, all three systems succeed, with X-MoE delivering 27.33 TFLOPS (versus 27.08 and 28.26 for DeepSpeed-MoE and Tutel at Small-SR, and 62.51 TFLOPS versus 52.15 and 64.00 at Small-LR). These results confirm that X-MoE's memory-efficient designs, especially our PFTbased expert routing, enable larger configurations under tight GPU memory constraints, with only a modest throughput trade-off due to the extra padding-free GEMM transforms required for maximal memory reuse on NVIDIA hardware.

#### 5.6 Implementation Validation

To verify the correctness of X-MoE, we compare its training loss curve against DeepSpeed-MoE on the 10.1B MoE model. The experiment is conducted on 16 GPUs with EP=8 and ZeRO DP enabled. In

this setting, we confirm that X-MoE closely tracks the convergence behavior of DeepSpeed-MoE, a production grade-implementation, as shown in Fig. [15.](#page-10-8) This confirms that X-MoE provides numerical convergence while enabling new system optimizations for scaling MoEs. We also investigate why the two curves do not match exactly and find that it is caused by a subtle difference in token-dropping logic. In DeepSpeed-MoE, a token is dropped from an expert if its routing score is negative, regardless of whether the expert's capacity has been exceeded. In contrast, X-MoE only drops tokens when they exceed expert capacity. As a result, X-MoE retains more tokens per batch, which might lead to its slightly lower loss under the same token consumption budget.

<span id="page-10-8"></span>![](_page_10_Figure_2.jpeg)

Figure 15: Loss validation with DeepSpeed-MoE and X-MoE.

## 6 Related Work

MoE Inference Frameworks. SGLang, vLLM, and TensorRT-LLM [\[12,](#page-11-23) [22,](#page-11-24) [27,](#page-11-25) [40\]](#page-11-26) are general inference frameworks that can also serve MoEs. SGLang provides optimized gather and scatter kernels in triton that are hardware agnostic; however, these kernels leverage block-sparse primitives and incur padding. vLLM provides optimized hardware-agnostic triton kernels via its FlashInfer [\[38\]](#page-11-27) backend. However, currently only MoEs which activate up to 7B parameters are supported. TensorRT-LLM is NVIDIA's LLM inference engine, but it is tightly coupled to the NVIDIA ecosystem. Moreover, none of these frameworks solve the activation memory explosion of large ℎ and tensors during MoE training.

Efficient Communication Primitives. DeepEP [\[39\]](#page-11-28) is an opensource efficient EP implementation by DeepSeek, relying on intrinsics available only on NVIDIA Hopper GPUs. TCCL [\[20\]](#page-11-29) modifies NVIDIA's NCCL to specifically optimize ring-based collectives on systems where the predominant interconnect is PCIe. Both techniques are tightly coupled to the NVIDIA ecosystem. Centauri [\[7\]](#page-10-9) introduces an automated way to uncover good schedules where computation is overlapped with communication in heterogeneous environments by decomposing a training task hierarchically into multiple tiers. Unlike these works, X-MoE focuses on system-level optimizations that enable scalable training of expert-specialized MoEs on non-NVIDIA platforms.

## 7 Conclusion

In this paper, we have taken a leap forward in designing an MoE training system X-MoE to scale expert-specialized MoEs, an increasingly popular model class. With techniques like padding-free MoE training pipeline with cross-platform kernels, redundancybypassing dispatching, and hybrid parallelism with sequence sharded MoE blocks, X-MoE enables training of massive MoEs on AMDbased HPC platforms while achieving high throughput, offering a system blueprint to train emerging expert-specialized MoEs on today's HPC platforms.

## Acknowledgements

We sincerely appreciate the insightful feedback from the anonymous reviewers. We also thank Emily Herron, Junqi Yin, and Hao Lu from ORNL for their useful discussion of this research. This research was supported by the National Science Foundation (NSF) under Grant No. 2441601. This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a nonexclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan). This research used resources at the Oak Ridge Leadership Computing Facility which is a DOE Office of Science User Facility. The work also utilized the Delta and DeltaAI system at the National Center for Supercomputing Applications (NCSA) through allocation CIS240055 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. The Delta advanced computing resource is a collaborative effort between the University of Illinois Urbana-Champaign and NCSA, supported by the NSF (award OAC 2005572) and the State of Illinois. UIUC SSAIL Lab is supported by research funding and gift from Google, IBM, and AMD.

## References

- <span id="page-10-1"></span>[1] Meta AI. 2024. Introducing Meta LLaMA-3. [https://ai.meta.com/blog/meta-llama-](https://ai.meta.com/blog/meta-llama-3/)[3/.](https://ai.meta.com/blog/meta-llama-3/)
- <span id="page-10-3"></span>[2] Meta AI. 2025. Llama 4: Multimodal Intelligence. [https://ai.meta.com/blog/llama-](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)[4-multimodal-intelligence/.](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- <span id="page-10-2"></span>[3] Anthropic. 2024. Claude 3 haiku: our fastest model yet. [https://www.anthropic.](https://www.anthropic.com/news/claude-3-haiku)
- <span id="page-10-5"></span>[com/news/claude-3-haiku.](https://www.anthropic.com/news/claude-3-haiku) [4] Argonne National Laboratory. 2024. Aurora Supercomputer. [https://www.alcf.](https://www.alcf.anl.gov/aurora) [anl.gov/aurora.](https://www.alcf.anl.gov/aurora)
- <span id="page-10-4"></span>[5] Scott Atchley, Christopher Zimmer, John Lange, and et al. 2023. Frontier: Exploring Exascale. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2023.
- <span id="page-10-0"></span>[6] Tom Brown, Benjamin Mann, Nick Ryder, and et al. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems (NeurIPS '20).
- <span id="page-10-9"></span>[7] Chang Chen, Xiuhong Li, Qianchao Zhu, Jiangfei Duan, Peng Sun, Xingcheng Zhang, and Chao Yang. 2024. Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '24).
- <span id="page-10-7"></span>[8] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. Training Deep Nets with Sublinear Memory Cost. arXiv[:1604.06174](https://arxiv.org/abs/1604.06174) [cs.LG]
- <span id="page-10-6"></span>[9] Weihao Cui, Zhenhua Han, Lingji Ouyang, Yichuan Wang, Ningxin Zheng, Lingxiao Ma, Yuqing Yang, Fan Yang, Jilong Xue, Lili Qiu, Lidong Zhou, Quan Chen, Haisheng Tan, and Minyi Guo. 2023. Optimizing Dynamic Neural Networks with Brainstorm. In USENIX Symposium on Operating Systems Design and

- Implementation (OSDI '23). 797–815.
- <span id="page-11-8"></span>[10] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. In the 62nd Annual Meeting of the Association for Computational Linguistics (ACL '24).
- <span id="page-11-20"></span>[11] DeepSeek-AI. 2025. DeepSeek-V3 Technical Report. arXiv[:2412.19437](https://arxiv.org/abs/2412.19437) [cs.CL]
- <span id="page-11-23"></span>[12] Zhixu Du, Shiyu Li, Yuhao Wu, Xiangyu Jiang, Jingwei Sun, Qilin Zheng, Yongkai Wu, Ang Li, Hai Helen Li, and Yiran Chen. 2024. SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models. In Proceedings of Machine Learning and Systems (MLSys '24), Vol. 6. 224–238.
- <span id="page-11-4"></span>[13] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research (JMLR '22) 23, 1 (2022), 5232–5270.
- <span id="page-11-14"></span>[14] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. 2023. Megablocks: Efficient sparse training with mixture-of-experts. Proceedings of Machine Learning and Systems (MLSys '23) 5 (2023).
- <span id="page-11-17"></span>[15] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '22). 15 pages.
- <span id="page-11-12"></span>[16] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. 2023. Tutel: Adaptive mixtureof-experts at scale. Proceedings of Machine Learning and Systems (MLSys '23) (2023).
- <span id="page-11-5"></span>[17] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, et al. 2024. Mixtral of experts. arXiv:2401.04088 (2024).
- <span id="page-11-18"></span>[18] Chenyu Jiang, Ye Tian, Zhen Jia, Shuai Zheng, Chuan Wu, and Yida Wang. 2024. Lancet: Accelerating mixture-of-experts training via whole graph computationcommunication overlapping. (2024).
- <span id="page-11-30"></span>[19] Rashika Kheria, Wenkai Du, Yongseok Koh, Raghu Raja, gilbertlee amd, James Dinan, Denis Maryin, dmitrygx, Ryan Hankins, Stanley Tsang, David Addison, AWSNB, Eric Badger, Gina Sitaraman, Nicholas Sly, Shi Jin, Sylvain Jeaugey, Theofilos Manitaras, and YoonGi Kim. 2024. ROCm/aws-ofi-rccl. [https://github.](https://github.com/ROCm/aws-ofi-rccl) [com/ROCm/aws-ofi-rccl](https://github.com/ROCm/aws-ofi-rccl)
- <span id="page-11-29"></span>[20] Heehoon Kim, Junyeol Ryu, and Jaejin Lee. 2024. TCCL: Discovering Better Communication Paths for PCIe GPU Clusters (ASPLOS '24).
- <span id="page-11-13"></span>[21] John Kim, William J. Dally, Steve Scott, and Dennis Abts. 2008. Technology-Driven, Highly-Scalable Dragonfly Topology. In 35th International Symposium on Computer Architecture (ISCA 2008). IEEE Computer Society, 77–88.
- <span id="page-11-24"></span>[22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023. 611–626.
- <span id="page-11-6"></span>[23] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 (2020).
- <span id="page-11-9"></span>[24] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. CoRR abs/2006.16668 (2020). arXiv[:2006.16668](https://arxiv.org/abs/2006.16668)
- <span id="page-11-22"></span>[25] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. 2024. Deepseekv2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 (2024).
- <span id="page-11-21"></span>[26] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. 2021. Efficient large-scale language model training on GPU clusters using megatron-LM. In International Conference for High Performance Computing, Networking, Storage and Analysis, (SC '21). 58.
- <span id="page-11-25"></span>[27] NVIDIA Corporation. 2023. NVIDIA TensorRT: Programmable Inference Accelerator.<https://developer.nvidia.com/tensorrt>
- <span id="page-11-0"></span>[28] OpenAI. 2023. GPT-4 Technical Report. CoRR abs/2303.08774 (2023).
- <span id="page-11-1"></span>[29] OpenAI. 2024. GPT-4o System Card. arXiv[:2410.21276](https://arxiv.org/abs/2410.21276) [cs.CL] [https://arxiv.org/](https://arxiv.org/abs/2410.21276) [abs/2410.21276](https://arxiv.org/abs/2410.21276)
- <span id="page-11-19"></span>[30] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of NAACL-HLT 2019: Demonstrations.
- <span id="page-11-10"></span>[31] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. In International Conference on Machine Learning (ICML'22).

- <span id="page-11-16"></span>[32] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deep-Speed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. In The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining(KDD 20). 3505–3506.
- <span id="page-11-7"></span>[33] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).
- <span id="page-11-11"></span>[34] Siddharth Singh, Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He, and Abhinav Bhatele. 2023. A hybrid tensor-expert-data parallelism approach to optimize mixture-of-experts training. In Proceedings of the 37th International Conference on Supercomputing (SC '23). 203–214.
- <span id="page-11-2"></span>[35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
- <span id="page-11-15"></span>[36] Philippe Tillet, Hsiang-Tsung Kung, and David D. Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (SIGPLAN 2019).
- <span id="page-11-3"></span>[37] XAI. 2025. Grok.<https://x.ai/grok>
- <span id="page-11-27"></span>[38] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. arXiv preprint arXiv:2501.01005 (2025).
- <span id="page-11-28"></span>[39] Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. 2025. DeepEP: an efficient expert-parallel communication library. [https://github.com/deepseek-ai/DeepEP.](https://github.com/deepseek-ai/DeepEP)
- <span id="page-11-26"></span>[40] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. 2023. Efficiently Programming Large Language Models using SGLang. CoRR abs/2312.07104 (2023).

#### **A** Evaluation Setup Details

Our evaluation is based on DeepSpeed version 0.15.5 and DeepSpeed-Megatron. We use PyTorch version 2.2.0 and AMD ROCm version 5.7.1. The peak device throughput of two MI250X GCDs is 383 TFLOPS, and the per-effective-GPU peak throughput is 191.5 TFLOPs. For cross-node communication on Frontier, we use the AWS-OFI-RCCL plugin [19] to enhance inter-node connectivity, which maps RCCLs connection-oriented transport APIs to libfabric's interface. We use libfrabric version 1.20.1. We set environmental variables CUDA\_DEVICE\_MAX\_CONNECTIONS=1 and NCCL\_NET\_GDR\_LEVEL=3 for better RCCL efficiency, as recommended by [5]. For DeepSpeed-MoE and DeepSpeed-TED, we use the DeepSpeed library version 0.15.5. Since Tutel is not provided as the end-to-end training pipeline, we integrate its MoE layer implementation from Tutel library version 0.3 into the DeepSpeed library. In end-to-end experiments, we refer to Tutel for this integration.

#### **B** Implementation of PFT Training Pipeline

