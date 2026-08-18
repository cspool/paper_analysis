# C. Robustness to Low-Cost Randomness

A practical concern for FPGA deployment is whether decoder gains rely on high-quality, fully independent random streams, which are often expensive to implement in hardware. To evaluate this risk in a conservative setting, we run all experiments with a *fixed-seed* random policy during decoding. In our implementation, each decoding shot uses a single stateful PRNG stream initialized from a fixed base seed.

We evaluate our decoder under different random policies. Fig. 18 shows the LER comparison and quantifies the relative LER difference between different PRNGs. The shaded region indicates a 95% non-significant zone using a binomial approximation. Our work remains competitive against both

![](_page_12_Figure_0.jpeg)

<span id="page-12-0"></span>Fig. 19. Decoding latency with and without the proposed optimizations.

MWPM and UF when using cheap but low-quality PRNGs. The relative differences are small and mostly lie within the 95% non-significant band, indicating no meaningful instability from the low-cost fixed-seed setting.

#### D. Optimization Ablation

We perform an ablation study to quantify the impact of different optimizations. We use a hardware-oriented software simulator that mirrors the dataflow of our final microarchitecture and allows individual hardware features to be selectively enabled or disabled.

We take the coset ensemble decoder architecture without any additional optimization as our baseline. Through ablation experiments, we aim to demonstrate that these optimizations can work synergistically to achieve an overall reduction in decoding latency. The trend in Fig. 19 shows that the benefits of our optimizations increase with larger code distance d. At d=11 and p=0.0015, Hierarchical ID Mapping delivers  $1.03\times$ , Multi-bank Hashing delivers  $2.30\times$ , and Graph Compression delivers  $1.18\times$  speedup over the baseline; enabling all optimizations achieves a  $3.24\times$  overall speedup.

#### IX. RELATED WORK

#### A. QEC Algorithm

While there exist families of quantum error correction codes and decoding algorithms [28]-[33], our work focuses on the surface code [34], [35]. Two mainstream decoding methods are Minimum-Weight Perfect Matching (MWPM) [13] and Union-Find (UF) [7]. MWPM solves the physical ML error problem while UF is a faster, sub-optimal version. Our decoder, by accounting for degeneracy and logical cosets, achieves higher accuracy than UF-based decoders in the LER comparisons (Fig. 8 and Fig. 9) while remaining in a similar low-latency regime (Fig. 10); compared with MWPMbased Micro-Blossom, it provides comparable accuracy at significantly lower latency. In contrast, the Tensor-Network (TN) decoder [35], which directly solves the logical coset ML problem, suffers from high contraction complexity. Although our decoder solves a sub-optimal coset ML problem, it maintains very low latency and high scalability for real-time implementation.

# C. Robustness to Low-Cost Randomness

A practical concern for FPGA deployment is whether decoder gains rely on high-quality, fully independent random streams, which are often expensive to implement in hardware. To evaluate this risk in a conservative setting, we run all experiments with a *fixed-seed* random policy during decoding. In our implementation, each decoding shot uses a single stateful PRNG stream initialized from a fixed base seed.

We evaluate our decoder under different random policies. Fig. 18 shows the LER comparison and quantifies the relative LER difference between different PRNGs. The shaded region indicates a 95% non-significant zone using a binomial approximation. Our work remains competitive against both

![](_page_12_Figure_0.jpeg)

<span id="page-12-0"></span>Fig. 19. Decoding latency with and without the proposed optimizations.

MWPM and UF when using cheap but low-quality PRNGs. The relative differences are small and mostly lie within the 95% non-significant band, indicating no meaningful instability from the low-cost fixed-seed setting.

#### D. Optimization Ablation

We perform an ablation study to quantify the impact of different optimizations. We use a hardware-oriented software simulator that mirrors the dataflow of our final microarchitecture and allows individual hardware features to be selectively enabled or disabled.

We take the coset ensemble decoder architecture without any additional optimization as our baseline. Through ablation experiments, we aim to demonstrate that these optimizations can work synergistically to achieve an overall reduction in decoding latency. The trend in Fig. 19 shows that the benefits of our optimizations increase with larger code distance d. At d=11 and p=0.0015, Hierarchical ID Mapping delivers  $1.03\times$ , Multi-bank Hashing delivers  $2.30\times$ , and Graph Compression delivers  $1.18\times$  speedup over the baseline; enabling all optimizations achieves a  $3.24\times$  overall speedup.

#### IX. RELATED WORK

#### A. QEC Algorithm

While there exist families of quantum error correction codes and decoding algorithms [28]-[33], our work focuses on the surface code [34], [35]. Two mainstream decoding methods are Minimum-Weight Perfect Matching (MWPM) [13] and Union-Find (UF) [7]. MWPM solves the physical ML error problem while UF is a faster, sub-optimal version. Our decoder, by accounting for degeneracy and logical cosets, achieves higher accuracy than UF-based decoders in the LER comparisons (Fig. 8 and Fig. 9) while remaining in a similar low-latency regime (Fig. 10); compared with MWPMbased Micro-Blossom, it provides comparable accuracy at significantly lower latency. In contrast, the Tensor-Network (TN) decoder [35], which directly solves the logical coset ML problem, suffers from high contraction complexity. Although our decoder solves a sub-optimal coset ML problem, it maintains very low latency and high scalability for real-time implementation.

