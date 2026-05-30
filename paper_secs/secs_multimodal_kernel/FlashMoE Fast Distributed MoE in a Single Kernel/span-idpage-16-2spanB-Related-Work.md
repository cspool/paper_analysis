# <span id="page-16-2"></span>B Related Work

Computation-Communication Overlap and Kernel Fusion. To reduce the communication overheads of synchronization in distributed DNN training, many research efforts have been focused on increasing the overlap of computation and communication. For generic Transformer-based models without MoE layers, many works [\[39–](#page-12-12)[47\]](#page-13-0) have provided insights and techniques to partition and schedule computation and communication operations, aimed at finer-grained overlapping. To address the challenges posed by *AlltoAll* communication and expert parallelism in MoE training, Tutel [\[48\]](#page-13-1)

<span id="page-17-1"></span>![](_page_17_Figure_0.jpeg)

Figure 14: Straggler effect of synchronous AlltoAll.  $M \times N$  A100 or V100 denotes N GPUs within a node across M nodes. Every GPU communicates with every other GPU per AlltoAll step. We capture the distribution of delay induced by stragglers across many steps. Actual Time  $t_a$  denotes the fastest kernel execution time across all GPUs, conversely Total Time t is the maximum recorded step time, while Delay is the maximum difference between t and  $t_a$ . Note Delay is idle time.

and FasterMoE [14] overlap *AlltoAll* with expert computation. Lancet [49] additionally enables both non-MoE computation in forward pass and weight gradient computation in backward pass to be overlapped with *AlltoAll*. Despite overlapping, the performance of these approaches is limited in practice due to blocking synchronous collective communication with barriers. In contrast, Flash-MoE fundamentally eliminates these inefficiencies with asynchronous, device-initiated data transfers overlapped with tiled computation all *within a single kernel*. FlashMoE further differentiates itself from SOTA works like COMET [12] and DeepEP [1], which also use this form of kernel-initiated communication but at a coarse-grained granularity and without complete kernel fusion.

#### <span id="page-17-0"></span>C Proof of Theorem 3.1

We begin with two necessary definitions vital to the proof.

**Definition C.1.** Define a write as  $w(p_s, p_t, i)$ , where  $p_s$  is the source process and i is an ordered tuple indicating the index coordinates for L residing on the target process  $p_t$ . A write-write conflict occurs when there exist at least two distinct, un-synchronized, concurrent writes  $w_1(p_{s_1}, p_{t_1}, i_1)$  and  $w_2(p_{s_2}, p_{t_2}, i_2)$ , such that  $p_{t_1} = p_{t_2}$  and index coordinates  $i_1 = i_2$  but  $p_{s_1} \neq p_{s_2}$ 

**Definition C.2.** For any source process  $p_s$ , a valid index coordinate i = (p\*, r, b, e, c) satisfies the following:

1. For inter-device writes, it must hold that  $p*=p_s$  and b=1. Note this also applies to self-looping writes  $w(p_t, p_t, i)$ .

2. For any write  $w(p_s, p_t, i)$ , if b = 0, then  $p_s = p_t$ . This rule describes intra-device staging writes.

We restate Theorem 3.1 and outline its proof below.

**Theorem C.1.** The symmetric tensor layout L is write-write conflict-free.

*Proof.* As is the case for typical physical implementations, assume that each index coordinate i maps to a distinct memory segment in L. Next, we show by contradiction that no write-write conflicts can exist when accessing L using  $valid\ i$ . For simplicity, we only include the index coordinates when describing a write. Assume that there exist at least two writes  $w_1(p_{s_1}, p_{t_1}, i_1)$ ,  $w_2(p_{s_2}, p_{t_2}, i_2)$  with  $p_{t_1} = p_{t_2}$  and valid destination coordinates  $i_1, i_2$ , where  $i_1 = i_2$  lexicographically and both are unpacked below.

$$i_1 = (p_1, r_1, b_1, e_1, c_1), i_2 = (p_2, r_2, b_2, e_2, c_2)$$

Note that intra-process writes always have distinct  $c_j$  coordinates, where  $j \in \{0, C-1\}$ . For inter-process transfers, we have two cases.

Case 1: 
$$p_{s_1} = p_{s_2}$$

Here,  $w_1$  and  $w_2$  are identical operations. This contradicts the definition of a conflict, which requires that  $p_{s_1} \neq p_{s_2}$ . In practice, such repeat writes never even occur.

Case 2: 
$$p_{s_1} \neq p_{s_2}$$

To ensure validity for  $i_1$  and  $i_2$ , it is the case that  $p_1 = p_{s_1}$  and  $p_2 = p_{s_2}$ . However, this implies that  $i_1 \neq i_2$  yielding a contradiction as desired.

### <span id="page-18-0"></span>**D** Memory Overhead

We measure the GPU memory required for the symmetric tensor L and runtime bookkeeping state of FlashMoE. The memory overhead primarily depends on the tile size, expert capacity (EC), and the number of experts (E). Table 3 summarizes the memory overhead across recent MoE models [50–55] during inference, showing that FlashMoE maintains a modest and predictable memory footprint. In particular, the symmetric tensor (ST) accounts for at most 2.15% additional memory relative to the total inference memory requirements.

<span id="page-18-1"></span>Table 3: Memory overhead of FlashMoE (tile size bM = 128,  $Size(T) = Tokens \times 4KB$ ).

| Model             | Params | S    | E   | H  | I     | ST (GB) | Model (GB) | Overhead (%) |
|-------------------|--------|------|-----|----|-------|---------|------------|--------------|
| Moonlight-16B-A3B | 16B    | 8K   | 64  | 2K | 1.38K | 0.25    | 59         | 0.49         |
| Ğrok-1            | 314B   | 8K   | 8   | 6K | 32K   | 0.75    | 1169       | 0.15         |
| Snowflake-Arctic  | 479B   | 4K   | 128 | 7K | 4.75K | 1.75    | 1784       | 0.12         |
| Qwen3-235B-A22B   | 235B   | 40K  | 128 | 4K | 1.5K  | 3.00    | 875        | 0.38         |
| Mixtral 8x7B      | 47B    | 32K  | 8   | 4K | 14K   | 2.00    | 175        | 2.15         |
| DeepSeek-V3       | 685B   | 160K | 256 | 7K | 2K    | 1.50    | 2551       | 0.11         |

