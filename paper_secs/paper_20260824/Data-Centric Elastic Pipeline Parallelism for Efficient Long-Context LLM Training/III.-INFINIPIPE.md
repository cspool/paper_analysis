# III. INFINIPIPE

Fig. 4 illustrates the overview of InfiniPipe, which adopts an  $\underline{disaggregated}$  architecture, where the solver solves the schedule plan on CPUs, and the executor carries out training on GPUs. The current batch's training process **overlaps** with the next iteration's schedule solving, eliminating the solver's overhead. Sequences sampled from the variable-length dataset are first processed by the solver's sequence processor (§ III-B) into three types of chunks (micro-batch of EPP) in a workload-balanced manner. Afterward, the chunk scheduler (§ III-C) schedules the chunks into pipelines and solves the optimal gradient checkpointing plan for each chunk and each pipeline stage. The tuned optimal execution plan is finally carried out by the executor to conduct LLM training. InfiniPipe employs a parallelism strategy of SP (equipped with ZeRO-3, degree  $d_s$ ) and PP (degree  $d_p$ ), with cluster size N equals  $d_s \times d_p$ .

#### <span id="page-2-1"></span>A. Cost Model

In this section, we present the fundamental cost model employed by InfiniPipe to depict the behavior of EPP, which is built at a theoretical standpoint, and is verified and refined via offline profiling and regression fitting.

*1) Definition of Chunk:* We firstly introduce *Chunk*, a key concept in InfiniPipe.

As illustrated in Fig. [4,](#page-2-0) *Chunk* can be categorized into three types: 1) Batched Chunk. Short sequences are batched together, resulting in a chunk formulated as S, a set containing multiple slices. 2) Split Chunk. A long sequence is split into multiple slices with the last one referred to as *tail slice*. Due to the causal mask of self-attention, a split chunk's computation relies on keys and values of preceding slices. We formulate a split chunk with its length s and context length C. 3) Hybrid Chunk. Short sequences can be packed with a *tail slice* s<sup>0</sup> for the sake of workload balance, formulated as a context C for s<sup>0</sup> and a set of slices S (s<sup>0</sup> included). Notably, *packing of two tail slices is avoided* as it forces co-scheduling of two long sequences[1](#page-3-0) , increasing memory overhead. In summary, all three types of chunks can be expressed using a uniform representation {C, S}, where C denotes the context length (0 for batched chunk) and S is the set containing the lengths of slices in the chunk.

Estimations on computation and communication overhead as well as memory footprint, are then introduced based on the concept of *Chunk*.

*2) Computation and Communication Analysis:* As for computation, we assume a quadratic time complexity with respect to sequence length due to the self-attention operation. Therefore, the computation time for processing chunk {Ck, Sk} during both forward and backward passes is modeled as:

$$T_{comp}(C_k, S_k) = \frac{1}{N} (\alpha_1 ((C_k + s_0)^2 - C_k^2) + \alpha_2 s_0 + \sum_{s \in (S_k - \{s_0\})} (\alpha_1 s^2 + \alpha_2 s)) + \frac{\beta_1}{d_p}$$
(1)

As for communication, V (communication volume), Bcomm (bandwidth), βcomm (latency) and f (frequency) are utilized to model the overhead:

$$T_{comm}(V, f) = \left(\frac{V}{B_{comm}} + \beta_{comm}\right) \cdot f, \tag{2}$$

Specifically, Ulysses-style SP requires four All-to-All communications in each layer, resulting in the following overhead:

$$T_{all2all}(S_k) = \left(\frac{e(D + D_{kv}) \sum_{s \in S_k} s}{d_s B_{all2all}(d_s)} + 2 \cdot \beta_{all2all}(d_s)\right) \frac{2L}{d_p}$$
(3)

where D, Dkv, and L indicate the model's hidden dimension, kv dimension, and number of layers, respectively. The total execution time for a chunk comprises both computation and communication components:

<span id="page-3-1"></span>
$$T_{tot}(C_k, S_k) = T_{comp}(C_k, S_k) + T_{all2all}(S_k), \tag{4}$$

which is applicable in both forward and backward passes.

<span id="page-3-0"></span><sup>1</sup>For instance, tail slice A<sup>3</sup> of sequence A is packed with B<sup>2</sup> of sequence B to AB. As a result, A1, A<sup>2</sup> and B<sup>1</sup> must be scheduled before AB, introducing activation overhead of both A and B.

*3) Stage-Aware Memory Footprint Analysis:* We begin by analyzing the activation memory footprint of a chunk. The activation memory footprint is proportional to the number of tokens after employing flash-attn [\[9\]](#page-11-21), [\[10\]](#page-11-22), which eliminates the need to materialize attention scores with O(S 2 ) space complexity. Notably, Mdkv representing the overhead of keys and values' gradients for *split chunk* is included in our estimation based on two observations of TPP: 1) chunks of a sequence execute backward *reversely* with the last chunk executing backward first (see Fig. [2\)](#page-1-2); 2) gradients for keys and values of all chunks are materialized *simultaneously* during the backward pass of the last chunk, while freed *asynchronously until* the completion of its own backward pass. Moreover, as current LLMs typically adopt a tokenizer with a large vocab size (≥ 128K), the memory overhead for logits is non-negligible. Consequently, the overall activation memory footprint is modeled as:

$$M_{act}(p, S_k) = Act(p, S_k) + M_{dkv}(S_k),$$

$$Act(p, S_k) = \left(\frac{M_{token}}{N} + \frac{M_{logits}}{d_s}[p = d_p]\right) \sum_{s \in S_k} s,$$

$$M_{dkv}(S_k) = (1 - I_k) \frac{2eLD_{kv}}{N} \sum_{s \in S_k} s,$$
(5)

where Mtoken, Mlogits are model-specific constants representing the activation and logits memory overhead per token, respectively. I<sup>k</sup> denotes whether the k th chunk is a *split chunk*.

The total memory footprint for the p th pipeline stage comprises the memory allocated for fixed model states Mms(p) and the ever-changing activation memory Mact(p, t):

$$M_{tot}(p,t) = M_{ms}(p) + M_{act}(p,t)$$
(6)

The peak memory of the 1F1B pipeline occurs during the steady phase, where each pipeline stage p maintains a *constant* number of chunks that have not finished their backward passes, as shown in Fig. [2.](#page-1-2) Let Wp(t), called the *chunks window*, denote the set of these chunks at time t, satisfying:

<span id="page-3-2"></span>
$$|W_p(t)| = d_p - p + N_{split} \tag{7}$$

which is equivalent to the number of micro-batches in the warmup phase. As activations of all chunks within Wp(t) must be accommodated, the total memory footprint is modeled as:

<span id="page-3-3"></span>
$$M_{tot}(p,t) = M_{ms}(p) + M_{act}(p, W_p(t))$$

$$= M_{ms}(p) + \sum_{k \in W_p(t)} M_{act}(p, S_k)$$
(8)

*4) Combining Gradient Checkpointing Together:* Gradient checkpointing affects estimation for both memory footprint and time cost. Same as common practice in Megatron-LM [\[28\]](#page-11-23), InfiniPipe applies checkpointing at layer granularity and let lckpt denote the checkpointed layers.

We first analyze how checkpointing affects memory footprint estimation. The *split chunk* exhibits a different behavior than other chunks: although gradient checkpointing is applied, its keys and values can not be released, as they are needed by the subsequent slices to perform the self-attention operation. However, the other activations of a checkpointed layer can be ignored. The phenomenon drives us to deal with keys and values individually in cost estimation. Specifically, we include not only the checkpointed layer's input but also keys and values in checkpointing memory overhead  $M_{ckpt}$ :

<span id="page-4-3"></span>
$$M_{ckpt}(S_k) = \frac{e(D + 2(1 - I_k)D_{kv}) \cdot l_{ckpt}}{d_s} \sum_{s \in S_k} s$$
 (9)

Moreover, checkpointing has no impact on  $M_{dkv}$ , and a chunk's activation footprint is further reformulated as:

<span id="page-4-4"></span>
$$M_{act}(p, S_k) = M_{dkv}(S_k) + M_{ckpt}(S_k) + \left(\frac{L - l_{ckpt} \cdot d_p}{L} \cdot \frac{M_{token}}{N} + \frac{M_{logits}}{d_s}[p = d_p]\right) \sum_{s \in S_k} s, \tag{10}$$

As for time cost estimation, checkpointing only affects the backward pass with a recomputation cost:

$$T_{ckpt}(C_k, S_k) = \frac{l_{ckpt}}{L \cdot d_s} \cdot T_{tot}(C_k, S_k)._{fwd}$$
 (11)

