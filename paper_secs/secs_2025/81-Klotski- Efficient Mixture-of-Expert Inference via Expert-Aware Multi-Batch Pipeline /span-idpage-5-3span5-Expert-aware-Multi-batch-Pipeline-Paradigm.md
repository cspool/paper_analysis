# <span id="page-5-3"></span>5 Expert-aware Multi-batch Pipeline Paradigm

We aim to develop a pipeline that minimizes all bubbles to maximize GPU utilization. To achieve this, we propose an expert-aware multi-batch pipeline paradigm, which is designed based on zig-zag block schedule [34]. By considering the computations of multiple batches simultaneously, this paradigm enables weight sharing and orchestrates the multi-batch computational graph around the experts to reduce bubbles. A partial computational graph is illustrated in Figure 7, where each row corresponds to the computations of one batch, and the multiple batches are considered together

as a batch group. Ultimately, this results in a nearly bubble-free pipeline, as shown in Figure 9. In the following, we will detail this paradigm from two perspectives: minimizing inter-layer bubbles and minimizing intra-layer bubbles.

First, minimizing inter-layer bubbles. Inter-layer bubbles primarily occur between the attention layer and the MoE layer. During the computations of multiple batches in the attention layer, Klotski prefetches only the weights of the gate and the hot experts, rather than the entire MoE layer. Because overlapping the I/O for the entire MoE layer is challenging, and Equation 1 must be satisfied.

<span id="page-5-1"></span>
$$n * t_{c\_A} \ge t_{I/O\ MoE} \tag{1}$$

Here, n represents the number of batches in a batch group,  $t_{c\_A}$  denotes the computation time of an attention layer for a batch, and  $t_{I/O\_MoE}$  is the time required to transfer the entire MoE layer. Equation 1 clearly necessitates a large n to hold true, which would introduce a significant amount of KV cache. What's more, due to the nature of sparse activation, some experts may not be activated, even when multiple batches are being processed at the same time. Loading them all into VRAM not only wastes resources but also increases latency. In contrast, only overlapping the I/O for the gate and hot experts is easier and more effective, which just needs to satisfy Equation 2.

<span id="page-5-2"></span>
$$n * t_{c A} \ge t_{I/O G} + K * t_{I/O E}$$
 (2)

Here,  $t_{I/O\_G}$  and  $t_{I/O\_E}$  represent the transfer times for the gate and a single expert, respectively. K equals k, the number of experts selected by the top-k gate, usually 1 or 2. Hot experts are chosen because they are likely engaged in most of the computations (see Figure 5), which provides an opportunity to minimize intra-layer bubbles subsequently. Additionally, during the computations of the gate, no prefetching is done. Instead, it is determined whether each gate-selected expert is a hot expert or one that has already been transferred. If not, the transfer of that expert is initiated immediately.

Second, minimizing intra-layer bubbles. As illustrated in the left panel of Figure 7(a), the sequence of experts shows that hot experts 2 and 4 have already been prefetched, while experts 5 and 3 are still undergoing transfer. Thus, the sequence of computations [2523424...] would result in the GPU stalling at positions 5 and 3, due to the incomplete transfer of data at these locations. However, computations involving experts 2 and 4 could proceed immediately. To reduce such unnecessary delays, we further adjust the order of expert computations across multiple batches, allowing computations involving the same experts to run continuously and prioritizing computations of hot experts. Since hot experts are transferred to GPU memory first and engaged in more computations, this adjustment allows more time for the transfer of experts still being loaded. After the computations for hot experts, the remaining experts compute in the order they are transferred. Additionally, experts that

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

(a) Expert-aware multi-batch computational graph orchestration

(b) Final computational graph example

**Figure 7.** Expert-aware multi-batch computational graph.

```
Algorithm 1: Schedule Algorithm of the Paradigm.
```

<span id="page-6-1"></span>Init: Generate length *l*, number of layers *n\_layer*, number of batches *n\_batch*, hidden state *h*, KV cache *c*. The indices *i*, *j*, *k* indicate that the it is processing the *i*-th token, performing computations at the *j*-th layer for the *k*-th batch.

```
1 for i < l do
      for j < n_layer do
2
         if layers[j] is not Gate then
3
             load(layers[j+1])
4
         if layers[j] is Expert_Layer then
5
             load(c[i][j+1][0])
6
             Experts process all tokens across batches.
             compute(layers[j])
             store(h[i][j])
             load(h[i][j+1][0])
         else
10
             ▶ Non-expert process each batch vertically.
11
             for k < n batch do
12
                 sync(load_cache_stream)
13
                 load(h[i][j][k+1], c[i][j][k+1])
14
                 compute(layers[j][k])
15
                 sync(store_cache_stream)
16
                 store(h[i][j][k], c[i][j][k])
17
         sync(load_weight_stream)
18
```

have completed all computations are offloaded immediately, rather than waiting for the entire layer's computations to finish, to reduce peak GPU memory usage.

Finally, Klotski executes computations according to the computational graph shown in Figure 7(b), sharing the loaded weights across multiple batches. This approach not only reduces the number of I/O operations to approximately 1/n of the original but also overlaps the time for each I/O, resulting in an almost bubble-free pipeline as illustrated in Figure 9

and significantly improving throughput. The algorithm details of this paradigm are formulated in algorithm 1. First, since hot experts are already prefetched during the attention layer, we do not perform prefetching in the gate layer (line 3), instead, the real-time transfer of experts is based on its results. Second, experts process all tokens across batches (line 5), since the computations of the expert layer are divided by experts rather than by batches. Third, the non-expert layer processes each batch sequentially (line 11), prefetching the necessary activations, key-value caches, etc., for the corresponding batch. Additionally, we synchronize the transfers of various streams using the *sync(*) function.

