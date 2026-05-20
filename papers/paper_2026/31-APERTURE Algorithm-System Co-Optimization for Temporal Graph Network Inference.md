**==> picture [176 x 57] intentionally omitted <==**

# **APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference** 

Hailong Yang 

## Enze Yu 

Yiqing Wang 

Beihang University 

Beihang University State Key Laboratory of Complex & Critical Software Environment Beijing, China hailong.yang@buaa.edu.cn 

Beihang University Beijing, China 22371494@buaa.edu.cn 

State Key Laboratory of Complex & Critical Software Environment Beijing, China yiqingwang@buaa.edu.cn 

Kejie Ma Kaige Zhang Beihang University Beihang University Beijing, China Beijing, China kejiema@buaa.edu.cn kaige.zhang@buaa.edu.cn 

Qingxiao Sun[∗] Beihang University Beijing, China qingxiaosun@buaa.edu.cn 

Chenhao Xie Beihang University Beijing, China fenahuhu@gmail.com 

Depei Qian Beihang University Beijing, China depeiq@buaa.edu.cn 

## **Abstract** 

## _**CCS Concepts:**_ • **Computer systems organization** → **Neural networks** . 

Temporal Graph Networks (TGNs) are widely used to model evolving relationships in dynamic graphs. However, existing inference systems enforce a step-wise paradigm: processing each temporal graph sequentially with a memory update followed by aggregation. We break this dependency by decoupling memory updates from aggregation while preserving prediction accuracy, thereby enabling a global view for fine-grained parallelism control. This design unlocks new optimization opportunities but introduces three system-level challenges: managing intermediate multi-state representations, curbing memory-bound update overheads, and selecting a safe yet efficient aggregation granularity. We present _APERTURE_ , a TGN inference framework that bridges algorithmic semantics and system design. To address the above challenges, _APERTURE_ (1) jointly aggregates temporal states via computation graph transformation, (2) minimizes redundant memory traffic through dependency-aware update reconstruction; (3) selects the optimal granularity by analytically modeling. The experimental results show that _APERTURE_ achieves up to 59.3× speedup over state-of-the-art baselines without compromising accuracy. 

_**Keywords:**_ Temporal Graph Networks, Inference System, Parallel Execution, Memory Efficiency 

## **ACM Reference Format:** 

Yiqing Wang, Hailong Yang, Enze Yu, Qingxiao Sun, Kejie Ma, Kaige Zhang, Chenhao Xie, and Depei Qian. 2026. APERTURE: AlgorithmSystem Co-optimization for Temporal Graph Network Inference. In _Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP ’26), January 31 – February 4, 2026, Sydney, NSW, Australia._ ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3774934.3786450 

## **1 Introduction** 

Graph Neural Networks (GNNs) have emerged as fundamental building blocks across a wide spectrum of domains [12, 31, 37]. By combining message passing over graph structures with neural operations, GNNs effectively capture complex dependencies among entities (i.e., graph nodes), thereby enabling accurate predictions in tasks such as node classification and community detection [19, 32, 37]. However, most real-world graphs are dynamic, with interactions continuously evolving over time. Static GNNs lack the temporal modeling capability to handle such evolving data. 

∗Corresponding author. 

To support dynamic graphs, Temporal Graph Networks (TGNs) extend static GNNs by transforming the input stream of timestamped interactions into a sequence of _temporal graphs_ [18, 22, 26, 36]. To capture dependencies across temporal graphs, TGNs maintain a persistent _node memory_ , analogous to the hidden state in recurrent neural networks: after each temporal graph is processed, the node memory is 

This work is licensed under a Creative Commons Attribution 4.0 International License. _PPoPP ’26, Sydney, NSW, Australia_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 https://doi.org/10.1145/3774934.3786450 

564 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Wang, Yang, and Yu et al. 

**==> picture [238 x 124] intentionally omitted <==**

**----- Start of picture text -----**<br>
s Indexed states f Indexed features<br>Memory update GNN aggregation<br>×<br>s s s s ...<br>f f<br>(a)<br>s s s<br>... f<br>(b)<br>**----- End of picture text -----**<br>


**Figure 1.** Comparison between (a) baseline TGN inference paradigm and (b) decoupled TGN inference paradigm. Removing unnecessary sequential dependencies enables memory update restructure and GNN aggregation parallelism. 

updated, and the resulting _temporal state_ is propagated stepwise to subsequent graphs, thereby accumulating long-range context [1, 10, 18]. Owing to this capability, TGNs have been widely adopted in both academic research [6, 20, 34, 39, 41, 42] and industrial applications, ranging from recommendation platforms [21, 24] to financial fraud detection [4, 35] and intelligent transportation systems [8, 13]. 

Recent efforts have sought to optimize specific components of TGN inference. TGL [41] introduces a temporal graph format to accelerate sampling, while ETC [5] reduces input transfers by eliminating redundant accesses and reconstructing layouts on GPUs. TGOpt [29] employs deduplication, memoization, and precomputation to cut redundancy in sampling, computation, and loading, and TGLite [30] extends TGOpt with additional system-level abstractions. While these systems deliver notable improvements, they all remain bound to the same _step-wise processing paradigm_ , leaving broader opportunities unaddressed. 

In this paradigm (Figure 1 (a)), inference must traverse temporal graphs strictly in chronological order, enforcing two coupled stages: (1) the _memory update stage_ , which refreshes per-node memory with new interactions and produces updated temporal states; and (2) the _GNN aggregation stage_ , which applies message passing over updated states and node features to generate predictions. However, we observe that only memory updates require chronological ordering across temporal graphs. The dependency in Figure 1 (a), which forces each memory update to wait for the preceding GNN aggregation, is unnecessary for semantic correctness and can be safely removed. This insight motivates us to decouple memory updates from GNN aggregation, as illustrated in Figure 1 (b), thereby enabling opportunities for restructuring memory updates and parallelizing aggregations that are unattainable under the conventional paradigm. 

To exploit these opportunities in practice, the rigid stepwise execution must be broken by transforming the Computation Graph (CG) of TGN so that multiple memory updates 

can be deferred and aggregated jointly. However, a naive transformation fails to deliver real performance gains. On one hand, handling multiple temporal states in node memory triggers recurrent GPU memory management overhead. On the other hand, aggregating these temporal states with separate kernels leads to redundant computation and underutilized GPU resources. _Challenge 1: how to defer the execution of multiple temporal states with lightweight memory management and efficient aggregation?_ 

After optimizing GNN aggregation, the performance bottleneck shifts to the memory update stage. Prior systems process one temporal graph at a time, where dependencies are implicitly carried through node memory handoffs. When multiple temporal states are retained, these handoffs expose cross-graph dependencies, resulting in redundant global memory reads and writes. In prior step-wise processing paradigm, this overhead remained hidden, as each temporal graph was processed independently. However, when memory updates are organized back-to-back, the limitation becomes evident: updated states are written to global memory, and in the next step, they must be read again for further computation, creating excessive traffic across temporal graphs. _Challenge 2: how to support cross-graph state updates without incurring prohibitive global memory traffic from localized node-memory handoffs?_ 

A further difficulty lies in determining the appropriate aggregation granularity, i.e., the number of memory updates accumulated before performing a joint GNN aggregation. From a performance perspective, larger granularity is favorable: it lowers the number of aggregations, reducing computation cost; with fewer aggregations, it also reduces synchronization points between memory updates, reducing coordination overhead. However, larger granularity also prolongs the lifetime of temporal states, raising peak memory consumption and risking oversubscription. Because inference precludes costly autotuning, granularity must be determined ahead of execution with negligible overhead. _Challenge 3: how to select an aggregation granularity that maximizes efficiency while guaranteeing memory safety?_ 

To address these challenges, we present _APERTURE_ , an end-to-end TGN inference framework that maximizes performance while ensuring memory safety. _APERTURE_ achieves this by separating memory updates from GNN aggregation and exploiting a global view to restructure updates and parallelize read-only aggregation. For _Challenge 1_ , _APERTURE_ remaps sampled results into a global feature map, and leverages this layout to pre-allocate memory space and eliminate duplicate operations during aggregation. For _Challenge 2_ , _APERTURE_ constructs a state-based DAG in parallel and leverages the global dependency information to optimize memory update execution and reduce memory traffic. For _Challenge 3_ , _APERTURE_ models temporal state lifetimes, using an analytical model to select granularity and a topological traversal to track peak memory usage. 

565 

APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

To the best of our knowledge, this is the first work to overcome the inherent limitations of the step-wise TGN paradigm, bridging the gap between algorithmic and systemic optimization. The contributions are as follows[1] : 

- We comprehensively analyze the temporal sensitivity of the two stages involved in TGN inference and illustrate the opportunities for parallel execution brought about by breaking inherent dependencies. 

- We propose three novel modules including _CG transformation engine_ , _memory update manager_ , and _aggregation granularity calculator_ , which reduce redundant operations and global memory accesses while maximizing efficiency under memory constraints. 

- We develop a TGN inference framework _APERTURE_ that coordinates algorithmic and systemic optimizations to accelerate the entire pipeline without accuracy loss. The experimental results show that _APERTURE_ achieves a maximum speedup of 59.3× in end-to-end inference compared to the state-of-the-art work. 

## **2 Background** 

## **2.1 Temporal Graph Networks** 

Temporal Graph Networks (TGNs) are designed for link prediction on dynamic graphs, i.e., predicting whether an edge will appear between two nodes at a future timestamp based on their historical interactions. The input is a stream of timestamped edges representing node interactions. To balance temporal ordering with execution parallelism, TGNs process edges in fixed-size groups in chronological order. For each group, _temporal neighbor sampling_ constructs a _temporal graph_ on the fly by selecting the _𝑘_ latest edges per node before the current interaction, where _𝑘_ limits the number of temporal neighbors. The resulting sequence of temporal graphs preserves temporal structure and serves as input to TGN inference, which consists of two tightly coupled stages: memory update and GNN aggregation. 

_(1) Memory update_ : TGN maintains a persistent _node memory_ , represented as a tensor in which each row corresponds to the state vector of a unique node identifier. At each step, node states involved in the current temporal graph are first updated using messages generated in the previous step: 

**==> picture [203 x 12] intentionally omitted <==**

where _𝑠𝑢_[−][denotes the state of node] _[𝑢]_[before the update,] _[𝑚] 𝑘𝑢_[−][is] the message previously generated from neighbor _𝑘_ to _𝑢_ , and _𝑁_ ( _𝑢_ ) is the neighbor set of _𝑢_ . AGGR(·) aggregates the stored messages (by mean or most-recent selection [14, 18, 25]), and UPDT(·) updates the node state, typically implemented with recurrent units such as RNNs [7, 14] or GRUs [3, 18]. 

After states are updated, the current interactions generate new messages that will be consumed in the next step. For each interaction in the temporal graph, edge _𝑒𝑢𝑣_ from node 

1The artifact for this paper is publicly available on Zenodo under DOI [27]. 

**Table 1.** Components of a temporal graph. 

|**Component**|**Description**|
|---|---|
|_Node IDs_|Identifers for all nodes in the temporal graph.|
|_Dstnode IDs_|Destination nodes involved in interactions.|
|_Latest Nbrs_|Latest neighbors of destination nodes.|
|_Latest Times_|Timestamps of the latest interactions.|



_𝑢_ to _𝑣_ at time _𝑡_ involves the destination nodes _𝑢_ and _𝑣_ ; thus, two messages are produced: 

**==> picture [187 x 26] intentionally omitted <==**

where MSG(·) is a learnable module (e.g., MLP), _𝑠𝑢_ and _𝑠𝑣_ are the latest states of _𝑢_ and _𝑣_ , _𝐸𝑢𝑣_ ( _𝑡_ ) is the edge feature, and ( _𝑡_ − _𝑡_ ·) encodes the elapsed time since the node’s last update. 

_(2) GNN aggregation_ : After memory updates, TGNs apply temporal GNN layers to generate embeddings for prediction. Each layer performs message passing with temporal attention, combining the updated node states with node features, edge features, and time encodings on the temporal graph: 

**==> picture [204 x 56] intentionally omitted <==**

where Φ(·) encodes time intervals, ATTN denotes attentionbased aggregation [18], and FFN is a feed-forward predictor. 

## **2.2 Existing TGN Inference Systems** 

Existing TGN systems typically implement the inference pipeline in three stages as follows. 

(i) _Sampling_ : Construct a temporal graph using temporal neighbor sampling. This stage generates the temporal graph, which serves as the input for the subsequent memory update and aggregation stages. Each temporal graph is composed of the components listed in Table 1, which together provide a structured abstraction for subsequent computations. 

(ii) _Memory update_ : Update node states through NM (Node Memory) and RMS (Raw Message Store), as shown in Figure 2. NM stores node states and timestamps, while RMS stores messages and their corresponding timestamps. During execution, node states are loaded from NM via node IDs. Time features are then computed by encoding times in NM and RMS, both accessed via node IDs. These features are concatenated with messages from RMS (also accessed via node IDs) to form raw messages. Finally, the node states and raw messages are processed together to update the node states. 

The updated node states are written back to both NM (via node IDs) and RMS (via dstnode IDs and latest Nbrs). The times from RMS are written back to NM (via node IDs), and the latest Times are written back to RMS (via dstnode IDs). 

566 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Wang, Yang, and Yu et al. 

(iii) _GNN aggregation_ : Apply temporal attention layers to aggregate the updated node states with features and time encodings, generating link-prediction outputs. 

Prior studies mainly target optimizations for the three stages. ETC [5] proposes a three-step data access policy with inter-batch pipeline for reducing redundant data accesses and preload time. TGOpt [29] targets storage and computation redundancies in TGAT [33] inference. These redundancies include intra-batch repeated edge interactions, interbatch duplicate embedding calculations, and repeated timeencoding operations. TGOpt addresses them through three techniques: deduplication, memoization, and precomputation. TGLite [30] extends TGOpt by providing lightweight abstractions and composable operators for programming TGNs. It introduces a set of TBlock based operators such as temporal neighborhood sampling, scatter or segmented computations that can be flexibly composed to build customized inference pipelines. Cascade [2] supports training with large batch sizes by exploiting spatial independence in scattered events. Overall, these approaches still operate within the step-wise paradigm, missing broader opportunities for enhancing parallelism and efficiency. 

Numerous works have studied system optimizations for static or snapshot-based GNNs [9, 11, 15–17, 21, 23, 28, 38, 40], but their techniques do not naturally extend to temporal scenarios, where sequential dependencies dominate. 

## **2.3 Potential Decoupling Opportunities** 

Our primary analysis focuses on the semantic correctness of the decoupling between memory updates and GNN aggregation. Specifically, we model TGN inference as discrete steps _𝑡_ = 1 _, . . . ,𝑇_ . Let _𝑆𝑢_[(] _[𝑡]_[)] denote node _𝑢_ ’s state _after_ processing step _𝑡_ (i.e., after memory updates), and let _𝑌𝑢_[(] _[𝑡]_[)] be the output embedding used for prediction at step _𝑡_ . A generic TGN step can be written as: 

**==> picture [210 x 41] intentionally omitted <==**

**==> picture [212 x 41] intentionally omitted <==**

The decoupling is semantics-preserving under the following sufficient conditions: (C1) _Read-only read path_ - Agg and MsgAgg[read] do not modify _𝑆_[(] _[𝑡]_[)] and do not feed back into Upd; (C2) _Deterministic temporal encoding_ - _𝜓_[read] (Δ _𝑡_ ) depends only on timestamps (independent of execution order/parallelism); (C3) _Permutation invariance_ - MsgAgg[read] is permutation-invariant over the neighbor multiset. Under C1–C3, the evolution of states { _𝑆_[(] _[𝑡]_[)] } is fully determined by the update recurrence alone, and the read path at step _𝑡_ is a pure function of the snapshot _𝑆_[(] _[𝑡]_[)] (and timestamps). Therefore, we may execute all memory updates in temporal 

**==> picture [234 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
loading  computation update<br>updated<br>states<br>Memory update<br>node raw<br>states states states tfeats<br>states msg<br>Node IDs Node IDs Node IDs<br>Time encoding<br>timesNM timesRMS<br>Node IDs Node IDs<br>NM.s NM.t RMS.s RMS.s RMS.t<br>Node IDs Node IDs Dstnode IDs Latest Nbrs Latest Times<br>**----- End of picture text -----**<br>


**Figure 2.** Illustration of _memory update_ stage. 

order to obtain { _𝑆_[(] _[𝑡]_[)] }, and postpone/reorder/parallelize the read-only aggregation as long as it reads the same snapshots, yielding identical { _𝑌_[(] _[𝑡]_[)] } to step-wise execution. 

## **3 Motivation** 

Decoupling memory updates from GNN aggregation can break the rigid step-wise paradigm in TGN inference systems, opening up system-level optimization opportunities. We make three observations based on profiling the TGN model over the _lastfm_ dataset on A100 GPU. 

## **3.1 Observation 1: Multi-states aggregation is accuracy-neutral but performance-sensitive** 

The rigid sequential processing of traditional TGNs limits parallelism and underutilizes hardware. A natural way to improve efficiency is to process multiple temporal graphs together, which we term the _inference step size_ . However, as shown in Figure 3, larger step sizes cause significant accuracy degradation due to causal violations: memory updates must follow strict temporal order to ensure correct state evolution. 

However, we observe that the dependency forcing each memory update to wait for the preceding GNN aggregation is not semantically necessary. Although memory updates require temporal consistency, GNN aggregation only reads node states without modifying them and is therefore not constrained by temporal order. This insight enables the decoupling of memory updates from aggregation: multiple memory updates can be executed sequentially while preserving temporal semantics, followed by a single GNN aggregation over the resulting states. We refer to this as _multi-state aggregation_ , governed by a granularity parameter Γ. The experimental results confirm that multi-state aggregation preserves model quality (See Section 5.3). 

We implement a naive computation graph transformation ( _Naive-CG_ ) that executes Γ memory updates in sequence, materializes the intermediate node states, and then performs 

567 

APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [234 x 316] intentionally omitted <==**

**----- Start of picture text -----**<br>
AP<br>0.85<br>AUC<br>0.80<br>0.75<br>0.70<br>0 2 4 6 8 10<br>Step Size<br>Figure 3.  Inference accuracy when processing multiple tem-<br>poral graphs simultaneously. AP and AUC denote Average<br>Precision and Area Under the ROC Curve, respectively.<br>GNN Aggregation Memory Management<br>Memory Update<br>8.38 8.70<br>7.93<br>3.64 5.66 6.25<br>5 5.70 6.60<br>1.64 2.80 3.97<br>3.10 1.98 0.37 1.91 0.29 1.94 0.23 1.87<br>0.88<br>0<br>1 100 200 300 400<br>Granularity<br>Accuracy Metrics<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 3.** Inference accuracy when processing multiple temporal graphs simultaneously. AP and AUC denote Average Precision and Area Under the ROC Curve, respectively. 

**Figure 4.** Time breakdown of naive computation graph transformation as the aggregation granularity increases. 

a single GNN aggregation over their concatenation. Figure 4 reports the breakdown as Γ increases: _GNN aggregation time_ decreases due to reuse and fewer kernel launches, _memory update time_ remains nearly unchanged, yet the _end-to-end time_ first drops and then grows, eventually exceeding the no-transformation baseline. The inflection is driven by repeated (de)allocations and large-state concatenations, whose overhead scales with Γ and offsets the aggregation gains. These findings suggest that while multi-state aggregation is accuracy-neutral, fully realizing its performance potential requires a careful CG transformation engine. 

## **3.2 Observation 2: Localized node-memory updates incur excessive memory access overhead** 

Figure 5 breaks down the execution time of memory updates, showing that the majority of cost is dominated by _global load/store operations_ , while the actual update logic and time encoding account for only a small fraction. This indicates that memory updates are highly _memory-bound_ , and reducing redundant global accesses is critical for efficiency. 

We further analyze consecutive temporal graphs and measure the overlap in nodes (Node IDs). As shown in Figure 6, a large portion of graphs exhibit high redundancy: over 70% of 

**==> picture [234 x 236] intentionally omitted <==**

**----- Start of picture text -----**<br>
LD+ST mem_cell time<br>67.9% 24.8% 7.4%<br>0% 25% 50% 75% 100%<br>Figure 5.  Time breakdown of memory update stage.<br>40<br>30<br>20<br>10<br>0<br>>80%78%-80%76%-78%74%-76%72%-74%70%-72%68%-70%65%-68%63%-65%61%-63%59%-61%57%-59%55%-57%<55%<br>34.92%<br>20.56%<br>17.87%<br>4.13% 6.51% 5.06% 5.58% 3.41%<br>1.76%<br>0.21% 0.00% 0.00% 0.00% 0.00%<br>Percentage (%)<br>**----- End of picture text -----**<br>


**Figure 6.** Percentage distribution of repeated nodes across consecutive memory update stages. 

graphs have more than 70% repeated nodes between successive updates. This reveals substantial opportunity for _fusing repeated accesses_ , motivating techniques that minimize crossgraph traffic and reuse updated states. 

## **3.3 Observation 3: Aggregation granularity affects both performance and memory usage** 

Aggregation granularity Γ is pivotal for inference performance and memory usage. As shown in Figure 7, runtime declines in both _memory update_ and _GNN aggregation_ stages as Γ increases. In memory update stage, larger Γ reduces inter-aggregation synchronization points, so each node state is updated fewer times across temporal boundaries. In GNN aggregation, larger Γ improves reuse via deduplication, eliminates redundant computation and feature loads, and lowers kernel-launch overhead. The auxiliary overhead remains nearly flat. However, peak GPU memory usage increases with Γ, as shown in Figure 8. For example, _wiki-talk_ reaches OOM near Γ/Γmax ≈ 0 _._ 3, whereas _lastfm_ sustains larger Γ. This motivates selecting the largest Γ per workload and device that remains within memory capacity. 

## **4 Methodology** 

We present _APERTURE_ , a decoupled TGN inference framework that decouples memory update from GNN aggregation and exposes a global state view for fine-grained scheduling. Figure 9 overviews three cooperating modules. 

The _CG Transformation Engine_ introduces a global feature map to eliminate allocator/concatenation overhead. It 

568 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Wang, Yang, and Yu et al. 

**==> picture [234 x 117] intentionally omitted <==**

**----- Start of picture text -----**<br>
250<br>GNN Aggregation<br>200 Memory Update<br>150 Auxiliary Overhead<br>100<br>50<br>0<br>100 300 500 700 900<br>Granularity<br>Time (ms)<br>**----- End of picture text -----**<br>


**Figure 7.** Time breakdown vs. granularity Γ on _lastfm_ . 

**==> picture [234 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
lastfm wiki-talk A100 (40GB) Memory Limit<br>40<br>30<br>20<br>10<br>0<br>0 20 40 60 80 100<br>Granularity Ratio (%)<br>Memory Usage (GB)<br>**----- End of picture text -----**<br>


**Figure 8.** Peak GPU memory usage vs. Γ/Γmax on _wiki-talk_ and _lastfm_ . The Γmax is the per-sequence upper bound. 

first performs remap-dedup-unmap preprocessing for sizing and balanced work, then preassigns per-state slots and builds a block-diagonal adjacency, executing a single zeroredundancy aggregation kernel to remove repeated loads, computation, and kernel launches. 

The _Memory Update Manager_ re-expresses updates at state granularity and builds a state-based DAG to make producerconsumer relations explicit. It materializes per-state temporal inputs and _recomposes_ consecutive levels when dependencies permit, forwarding updated values on-chip to avoid intermediate global memory accesses. 

The _Aggregation Granularity Calculator_ analytically models runtime and memory, estimates peak usage via topological traversal of state lifetimes, and selects the granularity that minimizes inference time under the given memory budget. 

## **4.1 CG Transformation Engine** 

**4.1.1 Data Remapping Preprocessing.** _APERTURE_ preallocates space for intermediate states, requiring the sizes of all sampling outputs. This prevents overlapping CPU-side sampling with GPU execution; we therefore migrate the entire temporal graph sampling procedure to the GPU. To simplify analysis and scheduling of memory updates, we deduplicate sampled results within current temporal graphs. Sampling is inherently balanced because each temporal graph selects a fixed number of targets with bounded neighbors; deduplication, by contrast, yields variable per-graph sizes, 

and per-graph processing reintroduces imbalance. We adopt a _remap-dedup-unmap_ pipeline: _remap_ assigns offsets via a prefix-sum over preceding graphs to enable inter-temporalgraph deduplication; after deduplication, _unmap_ restores the original per-graph partition. This design balances work across thread blocks, eliminates redundant kernel launches, and preserves the boundaries of temporal graphs with negligible overhead. 

**4.1.2 Global Feature Map Management.** Building on the deduplicated results, _APERTURE_ manages all intermediate data at the granularity of _states_ . A state corresponds to the update of a specific node at a particular temporal step, and serves as the basic unit of both memory updates and GNN aggregation. Managing data at this fine granularity avoids redundant storage of unreferenced entries on the GPU. 

A state is described by three attributes: (1) node_id: the original node identifier that the state corresponds to; (2) level_id: the temporal step index, where 1 ≤ level_id ≤ _𝑇_ and _𝑇_ is the total number of temporal graphs in the input sequence; (3) uniq_id: the position of this state among deduplicated entries at the same level. A state can be referenced in two complementary ways: the pair ⟨level_id _,_ node_id⟩ associates the state with its original graph entity, while the pair ⟨level_id _,_ uniq_id⟩ uniquely determines its storage location in the global structure. 

To store states compactly, we organize them into a CSRstyle global feature map. Let _𝑛ℓ_ denote the number of distinct states at level _ℓ_ . The prefix sums _𝑆ℓ_ are calculated as: 

**==> picture [168 x 30] intentionally omitted <==**

so that the storage address of state ⟨level_id _,_ uniq_id⟩ (addr) is calculated as: 

**==> picture [177 x 10] intentionally omitted <==**

This design provides two key advantages. First, recurrent GPU memory allocations are eliminated by assigning each state a pre-determined slot in the feature map, producing a compact representation that significantly reduces memory usage compared to baseline _𝑂_ ( _𝑇_ × _𝑁_ ) storage, where _𝑇_ is the number of temporal graphs and _𝑁_ is the total number of node identifiers in the dynamic graph. Second, the ⟨level_id _,_ uniq_id⟩ indexing scheme directly maps each state to its storage location, avoiding repeated scatter-gather operations and facilitating downstream aggregation. 

**4.1.3 Zero Redundancy Aggregation.** Instead of launching separate sparse aggregations for temporal graphs, _APERTURE_ constructs a block-diagonal sparse matrix by concatenating their adjacency structures along the diagonal, and multiplies it with the global feature map in one operation. This design offers multiple benefits. First, redundant memory accesses are avoided due to one-time feature loading. Second, repeated kernel launches are eliminated, where _𝑇_ 

569 

APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [505 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
temporal graphs node states storage CG Transformation Engine  §Section 4.1 ...<br>t0t0tt11t1 Node Memory Raw Message Store Data RemappingPreprocessing Zero RedundancyAggregation f f<br>§Section 4.1.1 §Section 4.1.3<br>t0<br>load store load store t1 ...<br>× Global Feature Map Management t2 f<br>s s [+] s s [+] ... §Section 4.1.2<br>t0 t1t t 12 f t1t2 f l0 l1 Memory Update Manager  §Section 4.2<br>Problem Formulation§Section 4.3.1 Aggregation GranularityCalculator FeatureGlobal s0 → s0 [+] s1 ConstructionState-basedDAG MaterializationTimedeltaPer-state<br>§Section 4.3 Map §Section 4.2.1 §Section 4.2.2<br>Peak Memory Estimation Adaptive Granularity Selection<br>§Section 4.3.2 §Section 4.3.3 s0 s0 [+] s1 s1 [+] Cross-level State Recomposition<br>§Section 4.2.3<br>**----- End of picture text -----**<br>


**Figure 9.** Overview of _APERTURE_ design. 

small-scale sparse aggregations are replaced with one kernel. Third, both intra- and inter-step redundancies are removed by accessing deduplicated features in the global layout. 

## **4.2 Memory Update Management** 

**4.2.1 State-based DAG Construction.** To feed a _global feature map_ , _APERTURE_ must recover all state dependencies, otherwise obscured by NM/RMS. We attach a lightweight metadata field parent_states to each state descriptor to record the _minimal_ dependency set for subsequent computations, without changing the canonical references ⟨level_id _,_ node_id⟩ and ⟨level_id _,_ uniq_id⟩. Let _𝑠_ be a state with node_id = _𝑣_ and level_id = _ℓ_ , the minimal parent set satisfies: 

**==> picture [201 x 26] intentionally omitted <==**

where _𝑢_ is the latest Nbrs (LNid) paired with _𝑣_ when a raw message ( _𝑣,𝑢_ ) is involved at level _ℓ𝑝_ ( _ℓ𝑝 < 𝑙_ ). Intuitively, NM updates always include the state’s own node memory _𝑣_ ; RMS-involved updates additionally depend on the sibling indexed by LNid. 

We materialize only LNid in parent_states (sentinel if absent); _𝑣_ is implicit from node_id. We also record a single _parent level ℓ𝑝_ in parent_states: the parent(s) are co-level (i.e., share the same level_id), so one level_id suffices. This design keeps the dependency DAG lightweight: each state carries _𝑂_ (1) metadata, with at most one explicit parent (its node_id plus a single parent level_id), thereby reducing DAG-construction complexity. 

**Parallel construction.** As shown in Figure 10, we propose a two-step kernel that builds state dependencies in parallel. 

_Step 1: Initialize two 𝑇_ × _𝑁 tables._ (i) The _Nid-level_ table _𝑀_ NM [ _ℓ, 𝑣_ ] records, for node _𝑣_ and level _ℓ_ , the nearest prior level _ℓ𝑝_ ≤ _ℓ_ at which _𝑣_ ’s NM was updated; entries are initialized to −1 when absent. (ii) The _DNid-LNid_ table _𝑀_ RMS [ _ℓ, 𝑣_ ] stores the latest neighbor _𝑢_ (LNid) paired with destination node _𝑣_ (DNid) at level _ℓ_ for raw-message updates ( _𝑣,𝑢_ ). 

_Step 2: Per-state parent lookup (one thread per state)._ Given a state ( _ℓ, 𝑣_ ), a thread first queries _𝑀_ NM along the time axis to obtain the parent level _ℓ𝑝_ as the nearest non-−1 entry for node _𝑣_ (i.e., the most recent NM update of _𝑣_ at or before _ℓ_ ). This yields the first parent ( _ℓ𝑝, 𝑣_ ). It then reads _𝑢_ from _𝑀_ RMS [ _ℓ𝑝, 𝑣_ ]: if _𝑢_ = −1, the state has only the NM parent; otherwise the second parent is ( _ℓ𝑝,𝑢_ ). 

_Example._ For the state ( _ℓ_ =2 _, 𝑣_ =0), scanning _𝑀_ NM [: _,_ 0] backward finds 0 (the value of _ℓ𝑝_ ), looking up _𝑀_ RMS [0 _,_ 0] returns 2 (the value of _𝑢_ ), so the two parents are (0 _,_ 0) and (0 _,_ 2). 

_Remark._ For small graphs with few nodes, we materialize a per-node prefix fill along levels so that _𝑀_ NM [ _ℓ, 𝑣_ ] already stores _ℓ𝑝_ for every ( _ℓ, 𝑣_ ). This optimization removes the backward scan, reducing the complexity of Step 2 to _𝑂_ (1). For large graphs, we keep the on-the-fly scan design to avoid the extra _𝑂_ ( _𝑇𝑁_ ) preprocessing and storage cost. 

**4.2.2 Per-state Timedelta Materialization.** We further materialize per-state temporal inputs to avoid redundant global NM/RMS accesses in memory update stages. 

**Computation pattern.** For each state _𝑠_ = ( _ℓ, 𝑣_ ) with parent key ( _ℓ𝑝, 𝑣,𝑢_ ) in the state DAG, we use the lookup function _𝜅_ (·) to read the required timestamps: 

**==> picture [216 x 13] intentionally omitted <==**

**==> picture [167 x 10] intentionally omitted <==**

Therefore, _𝜏_ ( _𝑠_ ) is a pure function of read-only NM/RMS entries selected by _𝜅_ ( _𝑠_ ), with no cross-state dependence, and its cost is dominated by global-memory access rather than arithmetic. 

**Same addressing pattern as DAG construction.** Both time prefill and DAG construction resolve the same parent key _𝜅_ ( _𝑠_ ) via the Nid-level and DNid-LNid tables: (i) _DAG construction_ uses _𝜅_ ( _𝑠_ ) to emit parent edges ( _ℓ𝑝, 𝑣_ ) and optionally ( _ℓ𝑝,𝑢_ ); (ii) _Time prefill_ uses _𝜅_ ( _𝑠_ ) to fetch both parent timestamps and produces the encoded temporal feature _𝜏_ ( _𝑠_ ). 

**Parallel materialization.** As shown in Figure 11, we materialize _𝜏_ ( _𝑠_ ) per state using the same _𝜅_ ( _𝑠_ ) as in Section 4.2.1. _Step 1: Initialize two 𝑇_ × _𝑁 time tables._ (i) _RMS-time 𝑇_ RMS. Initialize all entries to −1, then set _𝑇_ RMS [0 _, 𝑣_ ] to the initial 

570 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Wang, Yang, and Yu et al. 

**==> picture [241 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 init Nid:  Node IDs DNid:  Dstnode IDs LNid:  Latest Nbrs<br>Nid-level table DNid-LNid table Nid DNid LNid<br>0 0 0 -1 2 -1 0 -1 T0 0,1,2 0,2 2,0<br>T -1 1 1 1 T -1 -1 3 2 T1 1,2,3 2,3 3,2<br>2 2 2 2 1 0 -1 -1 T2 0,1,2,3 1,0 0,1<br>N N<br>2 DAG i: level id, j: node id<br>Nid-level table DNid-LNid table 0-0 0-1 0-2 i-j<br>0 0 0 -1 2 -1 0 -1<br>1-1 1-2 1-3<br>-1 1 1 1 -1 -1 3 2<br>2 2 2 2 1 0 -1 -1 2-0 2-1 2-2 2-3<br>**----- End of picture text -----**<br>


**Figure 10.** Example of state-based DAG construction. 1 Initialize the Nid-level and DNid-LNid tables from local indices. 2 Lookup both tables to resolve parent states, producing the global DAG (right). The colors of nodes correspond to their index sources (Nid, LNid) and states in DAG. 

timestamps. For _ℓ_ ≥ 1, write latest timestamps _only_ to entries ( _ℓ, 𝑣_ ) indicated by the DNid-LNid table (i.e., where a pair ( _𝑣,_ LNid) exists at level _ℓ_ ); keep other entries at −1 to be resolved by predecessor lookup at query time. (ii) _NM-time 𝑇_ NM. Initialize all entries to −1, then set _𝑇_ NM [0 _, 𝑣_ ] to the initial timestamps. For any ( _ℓ, 𝑣_ ), obtain _𝑇_ NM [ _ℓ, 𝑣_ ] by a predecessor lookup in _𝑇_ RMS, i.e. scan column _𝑣_ backward along the level axis from _ℓ_ to the nearest non-empty entry (level _ℓ_ ) and set _𝑇_ RMS [ _ℓ_[′] _, 𝑣_ ] to _𝑇_ NM [ _ℓ, 𝑣_ ] (Row 0 provides the initializer, so this lookup always succeeds). 

_Step 2: Per-state parent lookup (one thread per state)._ Launch one thread per state as enumerated by the Nid-level table; each thread writes to the orange slots in the figure. For a state ( _ℓ, 𝑣_ ): (1) obtain _𝑡_ RMS by a predecessor lookup in column _𝑣_ of _𝑇_ RMS, i.e., take the nearest prior non-empty entry at or before _ℓ_ ; (2) obtain _𝑡_ NM by the same predecessor rule in _𝑇_ NM (nearest prior non-empty in column _𝑣_ at or before _ℓ_ ); (3) compute Δ _𝑡_ ( _𝑠_ ) = _𝑡_ RMS − _𝑡_ NM and record it. After all Δ _𝑡_ ( _𝑠_ ) values are produced, run a single encoding pass to obtain _𝜏_ ( _𝑠_ ) = Φ(Δ _𝑡_ ( _𝑠_ )). Both predecessor lookups are guaranteed to succeed by construction (row 0 initialization and the NM parent definition), so no case handling is required. 

_Example._ For the dark-orange state ( _ℓ_ =2 _, 𝑣_ =1), step (1) finds _𝑇_ RMS [2 _,_ 1] empty and backtracks to obtain _𝑎_ 1; step (2) reads _𝑇_ NM [2 _,_ 1] = _𝑎_ 1; hence Δ _𝑡_ ( _𝑠_ ) = _𝑎_ 1 − _𝑎_ 1 = 0. 

_Remark._ Obtained time features are stored in a contiguous buffer _𝑇_ ; each state keeps _𝑂_ (1) metadata, an integer time_idx( _𝑠_ ) pointing to the row _𝑇_ [time_idx( _𝑠_ )]. 

**4.2.3 Cross-level State Recomposition.** Leveraging (1) an explicit state-based DAG ( _SDAG_ ) (Section 4.2.1), (2) perstate materialization of temporal inputs (Section 4.2.2), and (3) management of _states_ in global feature map across the entire pipeline (Section 4.1.2), we recompose the state operators of adjacent levels into a single kernel execution. 

|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>TRMStable<br>-1<br>t3<br>0<br>0<br>0<br>-1<br>Nid-level table<br>e0<br>e1<br>e2<br>e3<br>TNMtable<br>2<br>-1<br>1<br>-1<br>-1<br>0<br>0<br>3<br>-1<br>-1<br>2<br>-1<br>T<br>DNid-LNid table<br>N<br>a1-e1<br>a0-e0<br>a2-e2<br>a0<br>t0<br>-1<br>a1<br>-1<br>-1<br>a2<br>t1<br>t2<br>a3<br>-1<br>t3<br>2<br>TRMStable<br>**_lookup_**<br>1 **_init_**<br>e0<br>a0<br>-1<br>e1<br>a1<br>a1<br>e2<br>a2<br>t1<br>e3<br>-1<br>a3<br>TNMtable<br>TimeDelta|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||0|0|0|-1||e0|e1|e2|e3||a0-e0|a1-e1|a2-e2||
|T|-1|1|1|1||a0|a1|a2|-1|||a1-a1|t1-a2|a3-e3|
||2|2|2|2||-1|a1|t1|a3||t0-a0|a1-a1|t2-t1|t3-a3|
|N|||||||||||||||



**Figure 11.** Example of timedelta computation. 1 Initialize the _𝑇𝑅𝑀𝑆,𝑇𝑁𝑀_ based on DNid-LNid and Nid-level tables. 2 Lookup both tables to resolve parent times, producing the timedelta (right). The colors of nodes correspond to their index sources (Nid, LNid) and parent times. 

We define the consumers in level _ℓ_ +1 that depend on a unique producer in level _ℓ_ as the _composable subset_ . To identify this subset, we scan the maintained _SDAG_ in parallel: for each state in _ℓ_ +1, we test whether its parent resides in _ℓ_ ; these checks are independent across states and thus incur negligible overhead. At execution, the kernel (i) directly reuses the materialized temporal inputs for both levels, avoiding extra fetches and remapping; (ii) computes the update at _ℓ_ and forwards the result to _ℓ_ +1 via registers or shared memory, avoiding intermediate materialization; and (iii) writes back only the _final_ state of _ℓ_ +1 to global memory (the intermediate state at _ℓ_ is consumed on-chip and never spilled). This design reduces global-memory round trips and kernel-launch overhead, improving on-chip reuse while meeting temporalordering requirements. For portions that are not directly fusible, we reuse threads that finish early within the fused kernel to perform global-to-global ( _G2G_ ) data movement, thereby still avoiding an additional kernel launch. 

## **4.3 Aggregation Granularity Calculator** 

**4.3.1 Problem Formulation.** Let _𝑇_ denote the number of temporal steps. We model the total cost as: 

**==> picture [225 x 14] intentionally omitted <==**

where _𝑇_ MU (Γ) and _𝑇_ AGG (Γ) are the costs of memory update and aggregation, _𝑀_ peak (Γ) is the peak memory usage. We find that the total workload during inference is roughly constant, but increasing Γ improves the locality and parallelism of memory updates and GNN aggregation. Therefore, the inference cost theoretically decreases with increasing Γ, as supported by the experimental results in Section 3.3. Consequently, we set Γ as large as memory budget permits, which motivates the peak memory estimation below. 

**4.3.2 Peak Memory Estimation.** Although inference operates in an online fashion, our preprocessing stage provides complete visibility into each temporal graph, including the 

571 

APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

deduplicated-node upper bound. This enables a conservative yet accurate estimation of peak GPU memory usage under any given granularity Γ, avoiding costly fallback while still capturing the substantial gains enabled by larger Γ. Users may specify whether to move edge/node features, NM, and RMS to GPU memory. We define: 

- _𝑁_[ˆ] : deduplicated-node upper bound per temporal graph; 

- _𝑁_[˜] := (1 + _𝜖_ ) _𝑁_[ˆ] : safety-margined node count, _𝜖 >_ 0; 

- _𝑘_ : number of sampled neighbors per node; 

**Table 2.** Graph datasets ( _𝑑_ denotes feature dimension). 

|Dataset|# Nodes|# Edges|max(t)|_𝑑𝑣_|_𝑑𝑒_|
|---|---|---|---|---|---|
|lastfm|1,980|1,293,103|1.4e8|128|128|
|reddit|2,601,977|672,447|2.7e6|128|128|
|wiki|9,928|157,474|2.7e6|128|128|
|mooc|7,047|411,749|2.6e6|128|128|
|wiki-talk|1,140,149|7,833,140|1.2e9|100|172|
|gdelt|16,682|191,290,882|1.8e5|128|128|



   - _𝑏_ : bytes per feature element; 

   - dimout _,_ dimtime: dimensions of states, time features. 

- The worst-case number of nodes and edges under Γ are: 

**==> picture [173 x 12] intentionally omitted <==**

where _𝑘_ is the sampled fanout per node. We estimate the total peak memory usage as: 

**==> picture [209 x 65] intentionally omitted <==**

where _𝑏_ is the number of bytes per feature element, and _𝑓_ agg is a model-specific aggregation cost function obtained via static operator graph analysis. The fixed overhead accounts for storage components, node features, edge features, NM, and RMS, that may reside on either CPU or GPU. We determine their contribution to GPU memory by parsing the experiment configuration, which specifies whether each component is moved to device memory. 

memory and raw message store) are stored in host memory and fetched on demand. In TGLite-GPU, the above tensors are stored in device memory to avoid H2D transfers. However, on large graphs (e.g., _gdelt_ ), keeping multiple tensors resident in GPU memory can lead to out-of-memory (OOM) failures. All methods are executed under the same configurations for fair comparison. 

**Hardware Configuration -** All experiments run on a server with Intel Xeon Gold 6336Y CPU and NVIDIA A100 GPU (40 GB). To demonstrate hardware portability and memory safety guarantees of _APERTURE_ , we also report end-toend results on a second server with Intel Xeon Gold 6230R CPU and NVIDIA RTX 4090 GPU (24 GB). 

Following prior works [2, 5, 18, 29, 30], we split each dataset into train/val/test with a 70/15/15 ratio. We train on the training split and report results on the held-out test split. For small datasets, we process temporal graphs in fixedsize groups of 200; for large datasets, we use 2000. To reduce measurement noise, we repeat each experiment five times and report the mean. 

## **5.2 End-to-End Performance** 

**4.3.3 Adaptive Granularity Selection.** Once preprocessing concludes and prior to graph construction, we determine the aggregation granularity Γ for the subsequent process. By evaluating the _𝑀_ peak (Γ) function with respect to hardware constraints, we select the maximal feasible Γ with negligible runtime overhead. In this way, _APERTURE_ supports adaptive granularity selection while remaining robust to allocator variance and workload irregularities. 

## **5 Evaluation** 

## **5.1 Experimental Setup** 

To validate the effectiveness of _APERTURE_ , we conduct comprehensive experiments across diverse workloads. 

**Model and Datasets -** We adopt three representative models including TGN [18], TGAT [33] and JODIE [14]. As listed in Table 2, we use standard datasets from [29, 33, 41], where _wiki-talk_ and _gdelt_ are large-scale. 

**Baselines -** We compare _APERTURE_ with state-of-theart baseline TGLite [30]. We further compare with two implementations of TGLite: TGLite-CPU and TGLite-GPU. In TGLite-CPU, the node/edge features and state tensors (node 

As shown in Figure 12, _APERTURE_ consistently outperforms TGLite across all datasets and models on both A100 and RTX 4090. On A100, it yields an average speedup of 29 _._ 19× over TGLite-CPU and 21 _._ 15× over TGLite-GPU. On RTX 4090, the average speedup further increases to 34 _._ 53× over TGLiteCPU and 27 _._ 85× over TGLite-GPU. 

**Hardware-wise -** we observe larger relative speedups on RTX 4090: memory updates are highly memory-bound, and our global feature map plus a single block-diagonal _zeroredundancy_ aggregation reduce redundant global accesses and kernel launches, which disproportionately benefits devices under tighter memory bandwidth pressure. 

**Dataset-wise -** the gains align with redundancy profiles. For TGN on _lastfm_ and _gdelt_ , speedups are especially high: these graphs have many more edges than nodes, so eliminating repeated feature loads and consolidating aggregation removes substantial computation and traffic. For TGAT on _wiki-talk_ , we see notably higher gains than those for TGN: aggregation-side redundancy is significantly reduced, whereas TGN still bears memory-update traffic; _wiki-talk_ also exhibits tighter memory headroom at larger Γ. For _wiki_ , 

572 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Wang, Yang, and Yu et al. 

**==> picture [490 x 479] intentionally omitted <==**

**----- Start of picture text -----**<br>
TGLite-CPU TGLite-GPU APERTURE<br>60<br>45.9x 45.0x<br>41.7x<br>40 27.9x 30.5x 24.8x 32.7x 29.0x 36.7x 25.3x 38.0x 26.1x 25.2x<br>20 21.4x 21.4x 19.0x 14.2x 20.5x<br>0<br>lastfm wiki-talk reddit wiki mooc gdelt<br>(a) A100 GPU<br>59.3x<br>50.1x<br>50 42.7x 42.2x 43.3x<br>37.5x 37.3x 36.7x 34.8x<br>25 27.5x 28.8x 24.1x 17.8x 30.5x 26.7x 31.5x 28.0x 22.8x<br>0<br>lastfm wiki-talk reddit wiki mooc gdelt<br>(b) RTX 4090 GPU<br>Figure 12.  Performance speedup of  APERTURE  and TGLite-GPU over that of TGLite-CPU.<br>GA (SOTA) TE (SOTA) MU (SOTA) SD (SOTA) LD (SOTA)<br>GA (APERTURE) TE (APERTURE) MU (APERTURE) SD (APERTURE) LD (APERTURE)<br>100<br>10.4% 10.3% 10.5% 9.9% 9.6%<br>0.6% 0.3% 0.5% 0.6% 0.6%<br>35.2%<br>80<br>25.6% 30.0% 30.3% 30.5% 32.3%<br>1.6% 0.4%<br>60 5.2% 4.7% 2.0% 4.9% 2.5% 4.2% 1.7% 4.3% 1.7% 8.4% 0.4%<br>0.1%<br>1.6% 0.1% 0.1% 0.1%<br>28.0%<br>40<br>58.8% 55.0% 54.3% 55.4% 53.8% 1.8% 0.6%<br>20 0.0%<br>26.6%<br>2.2% 1.6% 2.5% 2.2% 2.4% 1.9%<br>0<br>lastfm wiki-talk reddit wiki mooc gdelt<br>TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE<br>TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE TGN TGAT JODIE<br>Speedup vs TGLite-CPU<br>Speedup vs TGLite-CPU<br>Normalized Time (%)<br>**----- End of picture text -----**<br>


**Figure 13.** Runtime breakdown of the components involved in _APERTURE_ . 

_mooc_ , and _reddit_ , which are relatively sparser, improvements are more uniform across models and mainly come from kernel consolidation rather than heavy redundancy removal. 

AUC/AP as TGLite (up to rounding), indicating that decoupling memory updates from GNN aggregation is semanticspreserving and does not alter node-state evolution. Thus, the performance gains incur no loss in predictive quality. 

## **5.3 Prediction Accuracy** 

We verify that _APERTURE_ preserves model quality by comparing Area under Curve (AUC, threshold-independent classification accuracy) and Average Precision (AP, ranking precision) against baseline systems across all datasets and three models. As shown in Table 3, _APERTURE_ attains the same 

## **5.4 Breakdown Analysis** 

Figure 13 reports the normalized runtime breakdown across datasets for TGN on A100 GPU. For clarity, we map each runtime stage to the corresponding _APERTURE_ components as follows. 

573 

APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**Table 3.** Prediction accuracy comparison (AP / AUC). 

|DT|System|||AP / AUC|||
|---|---|---|---|---|---|---|
|||lastfm|wiki-talk|reddit<br>wiki|mooc|gdelt|
|TGN|TGLite <br>**Our**|0.85/0.87 <br>0.85/0.87|0.96/0.95 <br> 0.96/0.95|0.99/0.99 0.98/0.99 <br> 0.99/0.99 0.98/0.99|0.99/0.99 <br> 0.99/0.99|0.98/0.98<br> 0.98/0.98|
|TGAT|TGLite <br>**Our**|0.73/0.77 <br>0.73/0.77|0.89/0.87 <br> 0.89/0.87|0.99/0.99 0.95/0.95 <br> 0.99/0.99 0.95/0.95|0.98/0.98 <br> 0.98/0.98|0.98/0.98<br> 0.98/0.98|
|JODIE|TGLite <br>**Our**|0.73/0.79 <br>0.73/0.79|0.96/0.95 <br> 0.96/0.95|0.98/0.99 0.89/0.93 <br> 0.98/0.99 0.89/0.93|0.99/0.99 <br> 0.99/0.99|0.97/0.97<br> 0.97/0.97|



- **GA** (GNN Aggregation): _Zero-Redundancy Aggregation_ and _Aggregation Granularity Calculator_ , which consolidate aggregation into a single kernel to reduce redundant loads, computations, and launches. 

- **TE** (Time Encoding): _Per-state Timedelta Materialization_ , which precomputes time deltas to avoid repeated timestamp access and improve parallelism. 

- **MU** (Memory Update): _State-based DAG Construction_ , _Crosslevel State Recomposition_ , and _Global Feature Map Management_ , which jointly reduce memory traffic and management overhead across temporal levels. 

- **SD** (Sample and Dedup): _Data Remapping Preprocessing_ , which remaps and deduplicates sampled graphs on GPU before aggregation. 

- **LD** (Loading): Also reduced by _Zero-Redundancy Aggregation_ , which minimizes redundant edge-indexed loads. 

The most significant effect of _APERTURE_ is that the dominant costs of **GA** and **MU** nearly vanish, dropping to only singledigit percentages across all datasets. 

## **5.5 Aggregation Granularity Analysis** 

**5.5.1 Peak Memory Estimation.** We evaluate the _𝑀_ peak (Γ) function by comparing estimated peak memory usage with measured peaks across datasets on TGN. As shown in Figure 14, the estimator is consistently conservative (upwardbiased), with an average relative error of 14.14%, which avoids OOM re-executions in practice while still enabling large feasible Γ for improved performance. 

**5.5.2 Performance Difference.** We vary Γ and measure end-to-end inference performance. As shown in Figure 15, increasing Γ consistently reduces runtime, as memory update stage involves fewer inter-aggregation synchronizations, while GNN aggregation stage benefits from higher reuse (deduped features), fewer loads, and fewer kernel launches. The auxiliary overhead remains nearly flat. 

## **5.6 Overhead Analysis** 

We profile three sources of overhead: (i) one-time allocation of the global feature map, (ii) state-based DAG construction (including preparation for temporal-feature computation), and (iii) aggregation-granularity selection. The first and third 

**==> picture [234 x 291] intentionally omitted <==**

**----- Start of picture text -----**<br>
wiki-talk lastfm Perfect Estimation Line<br>gdelt reddit<br>20.0 50<br>17.5<br>40<br>15.0<br>12.5 30<br>10.0<br>20<br>7.5<br>5.0 10<br>2.5<br>2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 20 30 40 50<br>Measured Peak Memory (GB)<br>Figure 14.  Estimated and measured peak memory usage.<br>lastfm gdelt<br>100<br>2.0<br>75<br>1.5<br>1.0 50<br>0.5 25<br>2 6 10 14 18 22 26 30 34 38 2 6 10 14 18 22 26 30 34<br>Aggregation Granularity Aggregation Granularity<br>Estimated Peak Memory (GB)<br>End to End Time (s)<br>**----- End of picture text -----**<br>


**Figure 15.** Performance varies with aggregation granularity. 

**==> picture [234 x 89] intentionally omitted <==**

**----- Start of picture text -----**<br>
State-based DAG Construction<br>9% 7.8%<br>6%<br>2.8%<br>3% 0.4% 0.8% 1.2% 1.0%<br>0%<br>lastfm wiki-talk reddit wiki mooc gdelt<br>Percentage of Total Time<br>**----- End of picture text -----**<br>


**Figure 16.** Overhead normalized to end-to-end inference. 

are negligible, the former is a single contiguous tensor initialization, the latter a lightweight function evaluation. As shown in Figure 16, DAG construction dominates the overhead but remains under 10% of end-to-end time across all workloads. The DAG construction of _wiki-talk_ and _gdelt_ accounts for a larger fraction due to their node-heavy and edge-heavy characteristics, respectively. The DAG construction cost scales with the number of nodes/edges, whereas the inference performance depends mainly on the chosen group size and dataset redundancy. 

## **6 Conclusion and Future Work** 

_APERTURE_ breaks the step-wise TGN inference paradigm by decoupling memory updates from GNN aggregation. It combines global feature map management, dependency-aware update restructuring, and analytical granularity selection to 

574 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Wang, Yang, and Yu et al. 

improve inference efficiency while guaranteeing memory safety. The evaluation results across configurations show end-to-end gains without accuracy loss, indicating that the decoupling methodology paves a promising system optimization road for TGNs. For future work, we plan to support strict QoS online serving by segmented preprocessing and consistent granularity selection with peak memory bound. We also plan to extend _APERTURE_ to multi-GPU execution via partitioned feature maps and DAG-guided updates with boundary exchange. 

## **Acknowledgments** 

This work is supported by National Natural Science Foundation of China (No. 62322201, U23B2020, 62402525, U22A2028 and 92373110), the Fundamental Research Funds for the Central Universities (JKF-2025012343648 and JKF-20240598), and State Key Laboratory of Complex & Critical Software Environment (SKLCCSE-2025ZX-04). 

## **References** 

- [1] Ke Cheng, Peng Linzhi, Junchen Ye, Leilei Sun, and Bowen Du. 2024. Co-neighbor encoding schema: A light-cost structure encoding method for dynamic link prediction. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 421–432. 

- [2] Yue Dai, Xulong Tang, and Youtao Zhang. 2025. Cascade: A Dependency-aware Efficient Training Framework for Temporal Graph Neural Network. In _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ . 95–110. 

- [3] Rahul Dey and Fathi M Salem. 2017. Gate-variants of gated recurrent unit (GRU) neural networks. In _2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS)_ . IEEE, 1597–1600. 

- [4] Mingjiang Duan, Da He, Tongya Zheng, Lingxiang Jia, Mingli Song, Xinyu Wang, and Zunlei Feng. 2025. Global Attribute-Association Pattern Aggregation for Graph Fraud Detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 39. 11616–11624. 

- [5] Shihong Gao, Yiming Li, Yanyan Shen, Yingxia Shao, and Lei Chen. 2024. Etc: Efficient training of temporal graph neural networks over large-scale dynamic graphs. _Proceedings of the VLDB Endowment_ 17, 5 (2024), 1060–1072. 

- [6] Shihong Gao, Yiming Li, Xin Zhang, Yanyan Shen, Yingxia Shao, and Lei Chen. 2024. Simple: Efficient temporal graph neural network training at scale with dynamic data placement. _Proceedings of the ACM on Management of Data_ 2, 3 (2024), 1–25. 

- [7] Stephen Grossberg. 2013. Recurrent neural networks. _Scholarpedia_ 8, 2 (2013), 1888. 

- [8] Jindong Han, Weijia Zhang, Hao Liu, Tao Tao, Naiqiang Tan, and Hui Xiong. 2024. Bigst: Linear complexity spatio-temporal graph neural network for traffic forecasting on large-scale road networks. _Proceedings of the VLDB Endowment_ 17, 5 (2024), 1081–1090. 

- [9] Kezhao Huang, Jidong Zhai, Zhen Zheng, Youngmin Yi, and Xipeng Shen. 2021. Understanding and bridging the gaps in current GNN performance optimizations. In _Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming_ . 119– 132. 

- [10] Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, and Reihaneh Rabbany. 2023. Temporal graph benchmark for machine learning on temporal graphs. _Advances in Neural Information Processing Systems_ 36 (2023), 2056–2073. 

- [11] Zhihao Jia, Sina Lin, Rex Ying, Jiaxuan You, Jure Leskovec, and Alex Aiken. 2020. Redundancy-free computation for graph neural networks. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ . 997–1005. 

- [12] Weiwei Jiang, Jiayun Luo, Miao He, and Weixi Gu. 2023. Graph neural network for traffic forecasting: The research progress. _ISPRS International Journal of Geo-Information_ 12, 3 (2023), 100. 

- [13] Duc Kieu, Tung Kieu, Peng Han, Bin Yang, Christian S Jensen, and Bac Le. 2024. TEAM: Topological evolution-aware framework for traffic forecasting. _Proceedings of the VLDB Endowment_ 18, 2 (2024), 265–278. 

- [14] Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting dynamic embedding trajectory in temporal interaction networks. In _Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining_ . 1269–1278. 

- [15] Fangxin Liu, Shiyuan Huang, Ning Yang, Zongwu Wang, Haomin Li, and Li Jiang. 2025. CROSS: Compiler-Driven Optimization of Sparse DNNs Using Sparse/Dense Computation Kernels. In _2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, 963–976. 

- [16] Xiaobo Lu, Jianbin Fang, Lin Peng, Chun Huang, Zixiao Yu, and Tiejun Li. 2025. Gator: Accelerating Graph Attention Networks by Jointly Optimizing Attention and Graph Processing. _ACM Transactions on Architecture and Code Optimization_ 22, 2 (2025), 1–24. 

- [17] Sudipta Mondal, Susmita Dey Manasi, Kishor Kunal, Ramprasath S, and Sachin S Sapatnekar. 2022. GNNIE: GNN inference engine with load-balancing and graph-specific caching. In _Proceedings of the 59th ACM/IEEE Design Automation Conference_ . 565–570. 

- [18] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. 2020. Temporal graph networks for deep learning on dynamic graphs. _arXiv preprint arXiv:2006.10637_ (2020). 

- [19] Oleksandr Shchur and Stephan Günnemann. 2019. Overlapping community detection with graph neural networks. _arXiv preprint arXiv:1909.12201_ (2019). 

- [20] Guangming Sheng, Junwei Su, Chao Huang, and Chuan Wu. 2024. Mspipe: Efficient temporal gnn training via staleness-aware pipeline. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 2651–2662. 

- [21] Jie Sun, Zuocheng Shi, Li Su, Wenting Shen, Zeke Wang, Yong Li, Wenyuan Yu, Wei Lin, Fei Wu, Bingsheng He, and Jingren Zhou. 2025. Helios: Efficient distributed dynamic graph sampling for online gnn inference. In _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ . 2–15. 

- [22] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. 2019. Dyrep: Learning representations over dynamic graphs. In _International conference on learning representations_ . 

- [23] Chunyang Wang, Desen Sun, and Yuebin Bai. 2023. PiPAD: pipelined and parallel dynamic GNN training on GPUs. In _Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ . 405–418. 

- [24] Ruijie Wang, Jingyuan Huang, Yutong Zhang, Jinyang Li, Yufeng Wang, Wanyu Zhao, Shengzhong Liu, Charith Mendis, and Tarek Abdelzaher. 2024. Tgonline: Enhancing temporal graph learning with adaptive online meta-learning. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 1659–1669. 

- [25] Xiaoyun Wang, Minhao Cheng, Joe Eaton, Cho-Jui Hsieh, and Felix Wu. 2018. Attack graph convolutional networks by adding fake nodes. _arXiv preprint arXiv:1810.10751_ (2018). 

- [26] Xuhong Wang, Ding Lyu, Mengjian Li, Yang Xia, Qi Yang, Xinwen Wang, Xinguang Wang, Ping Cui, Yupu Yang, Bowen Sun, and Zhenyu Guo. 2021. Apan: Asynchronous propagation attention network for real-time temporal graph embedding. In _Proceedings of the 2021 international conference on management of data_ . 2628–2638. 

575 

APERTURE: Algorithm-System Co-optimization for Temporal Graph Network Inference PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

- [27] Yiqing Wang. 2025. _PPoPP26_AE_APERTURE_CODE_ . https://doi.org/ 10.5281/zenodo.17710612 

- [28] Yuke Wang, Boyuan Feng, Gushu Li, Shuangchen Li, Lei Deng, Yuan Xie, and Yufei Ding. 2021. {GNNAdvisor}: An adaptive and efficient runtime system for {GNN} acceleration on {GPUs}. In _15th USENIX symposium on operating systems design and implementation (OSDI 21)_ . 515–531. 

- [29] Yufeng Wang and Charith Mendis. 2023. Tgopt: Redundancy-aware optimizations for temporal graph attention networks. In _Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ . 354–368. 

- [30] Yufeng Wang and Charith Mendis. 2024. Tglite: A lightweight programming framework for continuous-time temporal graph neural networks. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ . 1183–1199. 

- [31] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022. Graph neural networks in recommender systems: a survey. _Comput. Surveys_ 55, 5 (2022), 1–37. 

- [32] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. 2020. A comprehensive survey on graph neural networks. _IEEE transactions on neural networks and learning systems_ 32, 1 (2020), 4–24. 

- [33] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2020. Inductive representation learning on temporal graphs. _arXiv preprint arXiv:2002.07962_ (2020). 

- [34] Yuanyuan Xu, Wenjie Zhang, Ying Zhang, Maria Orlowska, and Xuemin Lin. 2024. TimeSGN: Scalable and effective temporal graph neural network. In _2024 IEEE 40th International Conference on Data Engineering (ICDE)_ . IEEE, 3297–3310. 

- [35] Jie Yang, Rui Zhang, Ziyang Cheng, Dawei Cheng, Guang Yang, and Bo Wang. 2025. Grad: Guided Relation Diffusion Generation for Graph 

Augmentation in Graph Fraud Detection. In _Proceedings of the ACM on Web Conference 2025_ . 5308–5319. 

- [36] Jiaxuan You, Tianyu Du, and Jure Leskovec. 2022. ROLAND: graph learning framework for dynamic graphs. In _Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining_ . 2358–2366. 

- [37] Muhan Zhang and Yixin Chen. 2018. Link prediction based on graph neural networks. _Advances in neural information processing systems_ 31 (2018). 

- [38] Kai Zhong, Shulin Zeng, Wentao Hou, Guohao Dai, Zhenhua Zhu, Xuecang Zhang, Shihai Xiao, Huazhong Yang, and Yu Wang. 2023. CoGNN: An algorithm-hardware co-design approach to accelerate GNN inference with minibatch sampling. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ 42, 12 (2023), 4883–4896. 

- [39] Yuchen Zhong, Guangming Sheng, Tianzuo Qin, Minjie Wang, Quan Gan, and Chuan Wu. 2023. Gnnflow: A distributed framework for continuous temporal gnn learning on dynamic graphs. _arXiv preprint arXiv:2311.17410_ (2023). 

- [40] Hongkuan Zhou, Ajitesh Srivastava, Hanqing Zeng, Rajgopal Kannan, and Viktor Prasanna. 2021. Accelerating large scale real-time GNN inference using channel pruning. _arXiv preprint arXiv:2105.04528_ (2021). 

- [41] Hongkuan Zhou, Da Zheng, Israt Nisa, Vasileios Ioannidis, Xiang Song, and George Karypis. 2022. TGL: a general framework for temporal GNN training on billion-scale graphs. _Proceedings of the VLDB Endowment_ 15, 8 (2022), 1572–1580. 

- [42] Hongkuan Zhou, Da Zheng, Xiang Song, George Karypis, and Viktor Prasanna. 2023. Disttgl: Distributed memory-based temporal graph neural network training. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ . 1–12. 

Received 2025-09-01; accepted 2025-11-10 

576 

