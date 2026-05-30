# D. Energy Consumption

Duplex reduces energy consumption by up to 33.28%, 42.03%, and 34.59% for Mixtral, GLaM, and Grok1 compared to GPU. Fig. 15 shows normalized energy consumption for generating one token with Duplex and GPU. We can see that most of the energy is consumed in the MoE and attention layers. Duplex reduces off-chip memory access energy in MoE and attention layer by leveraging Logic-PIM. The energy consumed by the attention layer rises as the sequence length increases, Duplex shows better energy efficiency in the long sequence lengths.

In particular, energy efficiency deteriorates in Mixtral and Grok1 compared to GLaM as the batch size increases. While

Duplex reduces latency through co-processing, it consumes more energy by utilizing xPU, which uses more DRAM energy than Logic-PIM. As Mixtral and Grok1 employ fewer experts than GLaM, resulting in a higher Op/B for the MoE layers in Mixtral and Grok1 than in GLaM at the same batch size. Thus, Duplex relies on xPU to process more experts in Mixtral and Grok1 compared to GLaM, leading to relatively lower energy efficiency compared to GPU when the batch size increases.

#### <span id="page-11-0"></span>E. Area Overhead

The total overhead for processing units in Duplex for each Logic-PIM stack is 17.80 mm<sup>2</sup>, which accounts for 14.71% of a 121 mm<sup>2</sup> HBM3 logic die [41]. The pitch of TSV was measured at 22 um [49], and the number of TSVs per channel was conservatively scaled by increasing it to four times the number required per channel in conventional HBM3 [49]. The added TSVs account for an area overhead of 10.89 mm<sup>2</sup>. Each Logic-PIM stack includes 32 GEMM modules, with each module comprising 512 FP16 MACs operating at 650MHz and a 8KB buffer, accounting for 3.02 mm<sup>2</sup>. Further, Logic-PIM contains two 1MB buffers to store input vectors and temporal results, occupying 2.26 mm<sup>2</sup>. A softmax unit, which consists of a comparator tree, adders, exponential units, an adder tree, dividers, multipliers, and a total of 128 KB buffers, occupies 1.64 mm<sup>2</sup>. Considering that the area overhead ranges from 20% to 27% [28], [29], Duplex demonstrates significant performance improvements with a lower area overhead.

