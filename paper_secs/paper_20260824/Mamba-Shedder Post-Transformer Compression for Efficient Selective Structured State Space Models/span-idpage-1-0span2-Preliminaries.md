# <span id="page-1-0"></span>2 Preliminaries

#### 2.1 State Space Models

State space models (SSMs) have a long history of modeling sequences and dynamic systems. Recently, *structured* SSMs, e.g., S4 [\(Gu et al.,](#page-10-2) [2022\)](#page-10-2), have been proposed as an alternative to Transformers because of their efficient capabilities for mapping input to output signals. When dealing with discrete sequences as in Natural Language Processing (NLP), the parameters A, B and C of these models are discretized to transform an input sequence, x<sup>t</sup> , and hidden state, h<sup>t</sup> , to obtain the output sequence, yt . It can be formalized as:

<span id="page-1-1"></span>
$$h_t = \mathbf{A}h_{t-1} + \mathbf{B}x_t, y_t = \mathbf{C}^{\top}h_t.$$
 (1)

Mamba: Selective State Space Models S4 and other structured SSMs are linear time-invariant (LTI), i.e., their parameters are fixed, limiting their effectiveness for sequence modeling. For instance, structured state space models fail in many contentand context-based reasoning tasks. These limitations have motivated the development of timevarying alternatives, e.g., Mamba [\(Gu and Dao,](#page-10-3) [2023\)](#page-10-3), which incorporate selection mechanisms and are suitable for solving tasks previously SSM generations failed. Specifically, Mamba's SSM module, S6, allows its parameters to depend on the input, thereby modifying the formulation from time-invariant to time-varying. A second improvement proposed in Mamba compared to previous SSMs is a hardware-aware algorithm that speeds up execution while reducing memory IOs.

Furthermore, Mamba-2 [\(Dao and Gu,](#page-10-4) [2024\)](#page-10-4) improves the original Mamba architecture by proposing *state space duality (SSD)*, which improves its efficiency on hardware accelerators compared to S6. This improvement is achieved by changing the *state matrix*, A, which directly controls the latent state, h. A is modified from being structured as a diagonal matrix to a formulation that utilizes a scalar-times-identity structure.

Additionally, Mamba-2 introduces the concept of heads in SSMs inspired by how multi-head attention (MHA) works and implementing a groupedvalue attention (GVA) head structure. Overall, the Mamba-2 architecture, with its SSD core component, allows for improved parallelism of the block's projections.

Mamba block Mamba models comprise several blocks stacked after each other. Figure [1](#page-2-1) on the left illustrates a single Mamba block. Each block has the selective SSM mechanism (S6 for Mamba-1 and SSD for Mamba-2) at its core, placed within a larger structure that combines a gated multilayer perceptron (MLP), a convolution, and SILU activation functions [\(Elfwing et al.,](#page-10-5) [2018\)](#page-10-5).

For more details about selective structured state space models, we refer the reader to [Gu and Dao](#page-10-3) [\(2023\)](#page-10-3) and [Dao and Gu](#page-10-4) [\(2024\)](#page-10-4).

#### 2.2 Hybrid Models

Lately, new models have been proposed that achieve the best of both worlds (Transformers and Selective SSMs) by proposing architectures with both classes of blocks. Zamba [\(Glorioso et al.,](#page-10-6) [2024\)](#page-10-6) is one example of such a hybrid model. It combines the strengths of Mamba's backbone and the efficiency of selective SSMs with a shared Transformer block that incorporates Transformers' powerful in-context learning capabilities. The *shared attention* mechanism, in which two attention blocks are reused and interleaved in an ABAB pattern throughout the network, is a characteristic innovation of Zamba. This model also applies LoRA adapters [\(Hu et al.,](#page-10-7) [2022\)](#page-10-7) to the shared MLP blocks, achieving specialization when interacting with the affected layers, memory efficiency, and faster inference with reduced computational overhead.

Another example of a hybrid model is Hymba [\(Dong et al.,](#page-10-8) [2024\)](#page-10-8). This model takes a different approach than Zamba, proposing an entirely new hybrid-head module, illustrated in Figure [1](#page-2-1) on the right, in which the SSM and Attention mechanisms contribute in parallel to the sequence modeling. Additionally, Hymba benefits from group query attention, cross-layer KV cache sharing, and learnable meta-tokens, resulting in higher throughput, reduced memory requirements, and competitive performance compared to models of similar size.

