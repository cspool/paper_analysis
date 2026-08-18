# I. INTRODUCTION

Fully Homomorphic Encryption (FHE) enables computation on encrypted data [1], emerging as a foundational technology for privacy-preserving applications. Among FHE cryptosystems, FHE over the Torus (TFHE) [2] is uniquely promising due to its programmable bootstrapping feature, which allows TFHE to efficiently support a wider variety of operations (e.g.,

B Corresponding authors are Ying Wang, Yinhe Han, and Mengdi Wang

logical and relational operators) [3] and makes it suitable for numerous privacy-preserving applications [4]–[9].

Despite its promise, TFHE suffers from significant performance overhead. A key source of this performance overhead lies in the n sequential iterations of homomorphic MUXs (HMUXs), also known as Blind Rotation, in the bootstrapping procedure, where n is an encryption parameter. To preserve cryptographic security, a large n is often required [2], [3]. Prior accelerators process n HMUX iterations sequentially [10]–[13], which makes their performance fundamentally constrained by this serial execution dependency.

In this paper, we exploit the previously overlooked pipeline parallelism among the n HMUXs, overcoming the sequential execution bottleneck to achieve extraordinary throughput. *However,* ➊ *unleashing this pipeline parallelism introduces extreme memory-bandwidth pressure from massive concurrent BSK access*. Each of the n HMUXs requires access to a unique bootstrapping key (BSK), which is represented as high-degree polynomials. This concurrent BSK access could create a dramatic memory-bandwidth requirement for the hardware. Prior works [10]–[12] attempt to mitigate massive BSK access by using High Bandwidth Memory (HBM) [14]. However, even with an optimized multi-level memory hierarchy, they cannot resolve the memory-bandwidth bottleneck caused by concurrent BSK access in cross-HMUX pipeline parallelism. The brute-force approach of provisioning more HBM stacks to meet the enormous bandwidth demand is fundamentally impractical. For instance, in Morphling [12], one HBM stack would consume nearly 30W of power, which is nearly 56% of the accelerator die's power. Therefore, scaling up the number of HBMs to sustain pipeline parallelism incurs unacceptable overheads in cost and power consumption.

To address this extreme bandwidth challenge, we propose a *multi-chiplet pipelined architecture* that features distributed on-chip SRAMs to store all BSKs (up to 126 MB), eliminating BSK movement over slow external memory. As BSKs can be reused by batching multiple bootstrappings, keeping them resident in on-chip SRAMs is a promising approach. To fully unleash the potential of this pipelined architecture, we design a fine-grained pipeline model with two levels: intrachiplet and inter-chiplet polynomial coefficient-grained (PCG) pipeline, which decentralizes BSK accesses across chiplets and confines each access within its local chiplet, removing the centralized bandwidth bottleneck.

Another challenge arises from the 2 intermediate ciphertext transfers (ICTs) between HMUXs. Each HMUX depends on the output ciphertext from the previous one, which introduces frequent ICTs and consequently causes severe D2D communication traffic. Existing locality-prioritized policies attempt to minimize D2D communication, such as widely used bootstrapping-level HMUX mapping [11], [12] and segmented mapping that evenly segments n HMUXs across c chiplets. However, these approaches either require expensive BSK duplication across all chiplets or incur severe pipeline bubbles due to coarse-grained mapping. To address this, we propose an Interleaved-Fusion policy that fuses contiguous HMUXs into multiple groups and interleaves these groups across the chiplets. By fusing HMUXs into a group, the ICTs within the group are confined to one chiplet, reducing D2D communications. Formally, this mapping is expressed by a two-dimensional temporal-spatial function, f(t, c), which represents the fused group mapped to the temporal layer t on chiplet c. Crucially, these groups may have variable fusion sizes. Our study shows that this flexible two-dimensional mapping unlocks several advantages, including avoiding the ICT bottleneck, reducing pipeline bubbles, and improving mapping utilization.

Finding the optimal mapping is non-trivial, as the ideal configuration varies under different encryption parameters (e.g., different iteration counts n). Therefore, we propose an Offline Interleaved-Fusion Scheduler (OIFS), which models this as an integer-partitioning problem and uses a dynamic programming algorithm to determine the optimal f(t,c).

This paper proposes CASCADE, a multi-chiplet pipelined TFHE accelerator that fully unleashes the potential of pipeline parallelism across HMUXs by addressing its two primary challenges: the extreme bandwidth pressure of concurrent BSK access and the intermediate ciphertext transfer bottleneck. CASCADE achieves significant performance-per-area improvement for practical applications, setting a new paradigm for high-efficiency TFHE acceleration. The key contributions of this paper are as follows:

- We identify the performance bottleneck resulting from the n HMUX iterations in TFHE bootstrapping, and propose \nexploiting the previously overlooked pipeline parallelism to overcome this sequential execution bottleneck and achieve high throughput.
- To address the extreme memory bandwidth demand from concurrent BSK access, we propose a multichiplet pipelined architecture, which features a distributed SRAM hierarchy with a BSK-distributed method to keep BSKs resident in distributed SRAMs, and intra- and interchiplet PCG pipelines to achieve high pipeline efficiency.
- To address frequent intermediate ciphertext transfers, we propose an Interleaved-Fusion mapping policy that combines the temporal and spatial mapping spaces, and an Offline Interleaved-Fusion-based scheduler that finds the optimal Interleaved-Fusion mapping under different encryption parameters.

#### **Algorithm 1** Programmable Bootstrapping (BSP)

```
Require: LWE ciphertext: c_{in} = (a_1, a_2, ..., a_n, b).
Require: RLWE ciphertext: c_{T}; GGSW ciphertexts: Bootstrapping Keys BSK; KeySwitching Keys: KSK.

Ensure: LWE ciphertext: c_{out}.

1: c \leftarrow \frac{2N}{q} \cdot c_{in}
2: (a_1, ..., a_n, b) \leftarrow c
3: ACC \leftarrow X^{-b} \cdot c_T
4: for (i \leftarrow 1; i \leq n; i \leftarrow i + 1) do Blind Rotation (BR)
5: BSK_i \leftarrow (X^{-\overline{a_i}} - 1) \cdot BSK_i
6: ACC_i \leftarrow BSK_i \boxdot ACC_{i-1}
7: end for
8: (a'_{i,1}, ..., a'_{i,l_{ks}}, b') = SampleExtract(ACC)
9: c_{out} \leftarrow (0, ..., 0, b') - \sum_{i=1}^{N} \sum_{j=1}^{l_{ks}} a'_{i,j} \cdot KSK_{i,j}
```

• We conducted experiments on TFHE applications and performed performance/area comparisons with TFHE accelerators. CASCADE achieves  $3.1 \times -30.5 \times$  speedup/area over prior TFHE accelerators.

#### II. BACKGROUND AND MOTIVATION

