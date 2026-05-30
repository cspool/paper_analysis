# <span id="page-14-0"></span>B Computational Overhead of Routing and Merging

Here we investigate the computational overhead of our Moe models compared to the dense counterpart. We consider an MoE layer and an input tensor x consisting of L tokens and d dimensions:  $x : L \times d$ . We assume that the model uses SwiGLU as the activation function in FFNs and it up-projects the input x to d'-dimensional activations in FFNs. In this case, processing the input on an FFN requires roughly  $6 \times L \times d \times d'$  FLOPs (there are two up projections and one down projections in SwiGLU-based FFNs). The overhead of soft-routing MoE comes mainly from the merging operation. Suppose that there are E experts and that the model makes a routing decision for every segment of *T* tokens (equivalently, there are L/T routing decisions). Each merging operation on E experts takes  $6 \times E \times d \times d'$  FLOPs (we compute three merged matrices). Therefore, the total overhead will be  $\frac{L}{T} \times 6 \times E \times d \times d'$ FLOPs. This indicates that compared to a dense FFN layer, an MoE layer with *E* experts requires  $\frac{E}{T}$  more FLOPs, compared to the dense counterpart. In our experiments, we set T = 256; this suggests that using E = 8 experts introduces 3.1% more computations and using E = 32 experts introduces 12.5% more computations at the FFN/MoE layers. It is worth noting that the computations from FFN layers are only a subset of the full model computations, so 3.1% is an overhead upperbound when measuring on full models. In our experiments, our most straightforward implementation leads to a 15% or 28% slowdown of training efficiency when using 8 or 32 experts (Table 3).

| Model    | Throughput (tokens/s/gpu) |
|----------|---------------------------|
| 0.3B     | 29,000                    |
| 0.3B/8E  | 24,500                    |
| 0.3B/16E | 22,900                    |
| 0.3B/32E | 20,800                    |

<span id="page-15-4"></span>**Table 3:** Training throughput (tokens/s/gpu) of our MoE models and the dense counterpart. Our implementation is based on data parallelism with the ZeRO optimization (Rajbhandari et al., 2020).

