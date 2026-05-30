# <span id="page-17-7"></span>A Detailed Dataflow Description

#### A.1 Grouped-Query Attention

Figure 10.(II) illustrates the computation dataflow of the 64 query heads projection. The token activation X in the chip array has shape (1, 2880). The activation X in all chips is split into four equal slices of (1, 720) and each chip takes one of the slices to calculate the partial sum of the query tensor. Each chip contains a private, hard-wired slice of  $W_q$  with

shape (720, 1024) inside the HN array. Therefore the product is generated locally and without any weight fetch. The four partial products are summed by a column-internal reduce operation, yielding a 16-head query vector (1, 1024). The 16 query heads are then reshaped into (2, 8, 64), reflecting the Grouped-Query Attention structure where every 8 query heads correspond to a single KV head.

Figure 10.(III) follows a similar spatial pattern for the key path. Each chip multiplies its (1,720) input slice with its hard-wired  $W_k$  slice (720,128) and emits a partial key vector (1,128). After the column-internal reduction and reshape, the new key head of (2,64) is held in chip# (# =  $\ell$  mod 4, where  $\ell$  is the token's position in the sequence). At this point the data layout is intentionally asymmetric: the query vector is fully replicated across columns, whereas each key vector is unique per chip in the *sequence* dimension, a choice that minimizes the traffic of the subsequent dot-product. The data flow of  $X \cdot W_n$  mirrors that of  $W_k$ .

Figure 10.(IV) shows the computation of attention weight Z in column-i chips. Every chip already has the complete duplicate of the column-i Q heads (2, 8, 64). The key tensor is partitioned horizontally, each chip keeps cached K with shape  $(2, 64, \ell/4)$  in its local KV-cache ( $\ell$  is the current context length). The VEX unit multiplies Q head (2, 8, 64) with K  $(2, 64, \ell/4)$  and produces the local attention weight Z  $(2, 8, \ell/4)$ . Because each chip sees only  $\ell/4$  tokens, a column-wise all-reduce needs to be performed, after which every chip completes the normalization of its local fragment.

Figure 10.(V) completes the attention score. V is tiled exactly like K; each chip reads a V slice  $(2, \ell/4, 64)$  from its KV-cache. The VEX unit multiplies the V  $(2, 8, \ell/4)$  with the local attention weight Z  $(2, 8, \ell/4)$ , and emits the partial-O tensors (2, 8, 64). A column-wise all-reduce needs to perform to add the four partial-O tensors. After that, all chips in column-i contain the 16 heads of the attention score O with the shape (2, 8, 64). Then, the matrix is flattened to the shape (1, 1024) for the multiplication with  $W_o$ .

Figure 10.(VI) depicts the output projection and first residual path. After the attention score computation, each of the four chips in the same column now holds the same, flattened attention score for 16 heads. The  $W_0$  matrix is partitioned row-wise across the column group chips, with each column assigned a weight shape of (1024, 2880) and each individual chip's HN array receiving a (1024, 720). Each chip generates a partial-O of shape (1, 720). These partial-Os are combined via one row-wise all-reduce and one column-wise all-gather to yield the final  $X_0$  of shape (1, 2880) in all chips.

### A.2 Feed-Forward Network with MoE

Figure 10.(VII) shows the experts router stage. After the Group Query Attention computation, all chips in HNLPU contain the complete  $(1,2880)~X_o$  vector. These values are passed through RMSNorm before entering the routing layer.

As  $W_{\rm rout}$  only accounts for about 0.01% of the total weights, we replicated a copy of the router weights on all 16 chips to avoid inter-chip data exchange. After the computation of  $X_{\rm norm} \cdot W_{\rm rout}$ , each chip obtains complete  $X_{\rm rout}$ . Next, top-k and softmax operations are performed. The top-k result is used to generate a masked input tensor, X, with a shape of (128, 1, 2880). In this tensor, the values for the top-k experts are set to those of  $X_{\rm norm}$ , while all others are set to zero. Additionally, the top-k results are normalized with a softmax to obtain the expert weights.

Figure 10.(VIII) shows the up- and gate-projection stage. Following the top-k masking, each chip processes its masked  $X_{\text{mask}}$  tensor, which consists of 8 vectors, each of shape (1, 2880). Among the 128  $X_{\text{mask}}$  (1, 2880) vectors in all chips, only k are non-zero. The weight of  $W_{\rm up}$  is sliced into 16 tiles, with the shape of (8, 2880, 2880) for each chip. Then, the local 8 vectors of  $X_{\text{mask}}$  are multiplied with the sliced weight  $W_{\rm up}$  of eight experts. The 16 chips produce a total  $X_{\rm up}$  tensor of shape (128, 1, 2880). Because each chip stores the complete weight matrices for all experts, this step requires no inter-chip data communication. The gate projection follows the same partitioning:  $X_{\text{mask}}$  is multiplied by the corresponding  $W_{\text{gate}}$  slice, yielding an  $X_G$ , (128, 1, 2880) in total and (8, 1, 2880) for each chip. Applying the SwiGLU activation to  $X_G$  and  $X_{up}$ , and taking the element-wise product, we get the output  $X_t$  for the subsequent down-projection.

Figure 10.IX shows the down-projection and second residual path. Still,  $X_t$  in each chip with the shape of (8, 1, 2880) multiplies its down weight slice  $W_{\rm down}$ , (8, 2880, 2880), to produce a partial  $X_{\rm down}$ , (8, 1, 2880). Next, the expert weights, which were obtained from the previous stage (VII), are multiplied with the corresponding expert outputs to get the weighted output for each expert. Subsequently, an all-chip all-reduce operation is performed to sum the partial results. The shape of  $X_{\rm down}$  in all chips is from (128, 1, 2880) to (1, 2880). The summed  $X_{\rm down}$  is then added to  $X_o$  to yield the final layer output Y (1, 2880).

#### <span id="page-18-0"></span>B Notes to Table 3.

<sup>1</sup>Deployment scale. We define the "Low Volume" scenario as a single HNLPU system. The "High Volume" scenario targets OpenAI-scale throughput (~100 M tokens/s [63, 64]), corresponding to a 50-system HNLPU cluster. To ensure a fair TCO comparison, we normalize hardware counts based on equivalent inference throughput. Under a high-concurrency workload (1K prefill/1K decode, concurrency 50), the average throughput per H100 GPU in a distributed setting is 1.08 K tokens/s [15]. Consequently, given the HNLPU's ~2 M tokens/s throughput under the same workload configuration, we equate one HNLPU system to approximately 2,000 H100 GPUs.

<sup>2</sup>Facility power modeling / PUE. Facility-level PUE is assumed to be 1.4, consistent with modern hyperscale AI

<span id="page-19-1"></span><span id="page-19-0"></span>datacenters [\[19\]](#page-15-23).

Table 5. HNLPU Cost Analysis.

| -                                          | Cost (\$)         |
|--------------------------------------------|-------------------|
| Recurring Cost (\$ / chip)                 |                   |
| Wafer                                      | 629               |
| Package & Test                             | 111 ∼ 185         |
| HBM                                        | 1,920 ∼ 3,840     |
| System Integration                         | 1,900 ∼ 3,800     |
| Non-recurring Cost (\$)                    |                   |
| Photomask                                  |                   |
| Homogeneous Mask                           | 13.85 M ∼ 27.69 M |
| Metal-Embedding Mask                       | 18.46 M ∼ 36.92 M |
| Design & Development                       |                   |
| Architecture                               | 1.87 M ∼ 3.74 M   |
| Verification                               | 9.97 M ∼ 19.93 M  |
| Physical                                   | 4.80 M ∼ 14.41 M  |
| IP                                         | 10.23 M ∼ 20.46 M |
| Total Cost Scenarios (\$)                  |                   |
| Initial Build (Full NRE + Recurring)       |                   |
| 1-HNLPU                                    | 59.25 M ∼ 123.3 M |
| 50-HNLPU                                   | 62.83 M ∼ 129.9 M |
| Re-spin (Metal-Embedding Mask + Recurring) |                   |
| 1-HNLPU                                    | 18.53 M ∼ 37.06 M |
| 50-HNLPU                                   | 22.11 M ∼ 43.68 M |

<sup>3</sup>Node price of H100 and HNLPU.For H100 price, each NVIDIA HGX H100 platform (8 GPUs/node) costs about \$ 320,000, including server, intra-node networking and 3-year hardware warranty [\[73,](#page-16-23) [80\]](#page-17-21). For HNLPU, we break down the node cost in Table [3](#page-13-0) into recurring cost and non-recurring engineering (NRE) cost in Table [5.](#page-19-1)

Regarding recurring cost, we first estimate the silicon cost. Assuming a cost of \$ 16,988 for a 300 mm 5 nm wafer [\[7,](#page-15-24) [43\]](#page-16-24), Murphy's model (<sup>0</sup> = 0.11 def/cm<sup>2</sup> ) predicts a 43% yield (∼27 of 62 dies), resulting in ≈ \$629 per good die. Second, packaging and testing are estimated at \$3,000–\$5,000 per wafer to account for 2.5D integration complexity [\[68\]](#page-16-25), resulting in an amortized cost of \$111–\$185 per chip. Third, given an HBM cost of \$10–\$20/GB [\[21,](#page-15-25) [24\]](#page-15-26), the 8-stack configuration (24 GB per stack) amounts to \$1,920–\$3,840 per HNLPU module. Finally, we include system integration costs, covering the chassis, motherboard, cooling, power, and CXL interconnects; these figures align with per-chip costs from established commercial platforms [\[70\]](#page-16-26).

Regarding one-time costs, we distinguish between photomask NRE and design & development expenses. First, we model the photomask NRE using a normalized cost model based on lithography complexity for the 5 nm technology node. To account for the disparity in manufacturing costs, we assign a cost weighting factor of 6× to EUV reticles relative to standard 193i DUV reticles [\[75\]](#page-17-22). Given a typical 5 nm layer stack comprising 12 EUV and 58 DUV layers [\[4,](#page-14-5) [40,](#page-15-27) [60,](#page-16-27) [82\]](#page-17-23), the total mask set value corresponds to 58 + (12 × 6) = 130

normalized DUV units. In HNLPU architecture, the metalembedding configuration requires 10 DUV reticles (VIA7, M8 Mandrel, M8 Cut, VIA8, M9 Mandrel, M9 Cut, VIA9, M10, VIA10, M11); consequently, this variable portion accounts for 7.7% (10/130) of the full mask set, while the remaining 92.3% represents the homogeneous mask cost shared across all variants. Anchoring the absolute 5 nm mask set cost to a range of \$ 15 M (optimistic) to \$ 30 M (pessimistic) [\[30,](#page-15-28) [42,](#page-16-28) [69\]](#page-16-29), we derive a shared homogeneous mask cost of \$ 13.85–\$ 27.69 M. The metal-embedding cost is estimated at \$ 1.15–\$ 2.31 M per variant, amounting to \$ 18.46–\$ 36.92 M in total for 16 chips. Second, for HNLPU design and development costs, we derive our estimates from internal engineering data and design experience.

<sup>4</sup>Data center infrastructure. We consider two primary capital expenditures: inter-node networking and facility construction. For the H100 cluster baseline, we assume a standard three-tier non-blocking Fat-Tree topology. In terms of hardware composition, the network fabric is built using NVIDIA ConnectX-7 [\[16\]](#page-15-29) network interface cards (NICs) and Quantum-2 (QM9700) InfiniBand switches [\[27\]](#page-15-30), interconnected with corresponding optical transceivers [\[26\]](#page-15-31). Accounting for the NICs, switches, and cabling costs, the estimated network equipment capital expenditure is approximately \$45 K per node. The facility construction cost is modeled as \$12 M per MW of critical IT load [\[17\]](#page-15-32). For HNLPU, we scale the networking cost based on the number of chips, while the construction cost is scaled based on total power consumption.

<sup>5</sup>Re-spin cost. For H100, changing the model does not require a re-spin; therefore, this cost is zero. For HNLPU, the re-spin cost comprises the Non-Recurring cost for the metalembedding mask, plus the recurring costs for fabricating, packaging, and testing, as listed in Table [5.](#page-19-1)

<sup>6</sup>Electricity. The electricity cost is calculated based on the industrial electricity price of \$0.095/kWh, representative of major U.S. data center hubs [\[81\]](#page-17-24).

<sup>7</sup>Maintenance & Support. For H100 clusters, Maintenance & Support includes software licensing and hardware maintenance. Software license fees are calculated based on NVIDIA AI Enterprise pricing guidelines [\[61\]](#page-16-30). Hardware maintenance is conservatively estimated as 5% of hardware CapEx per year [\[45,](#page-16-31) [72\]](#page-16-32). For HNLPU, we model this cost by provisioning spare nodes: one for the low-volume scenario and five for the high-volume scenario.

<sup>8</sup>CO<sup>2</sup> emission. Our carbon emission analysis incorporates both embodied emissions from hardware manufacturing and operational emissions from energy consumption. We estimate the manufacturing emission for a single H100 card or an HNLPU module at 124.9 kgCO2e [\[50,](#page-16-33) [99\]](#page-17-25). The grid carbon intensity is assumed to be 0.38 kgCO2e per kWh [\[34\]](#page-15-33).