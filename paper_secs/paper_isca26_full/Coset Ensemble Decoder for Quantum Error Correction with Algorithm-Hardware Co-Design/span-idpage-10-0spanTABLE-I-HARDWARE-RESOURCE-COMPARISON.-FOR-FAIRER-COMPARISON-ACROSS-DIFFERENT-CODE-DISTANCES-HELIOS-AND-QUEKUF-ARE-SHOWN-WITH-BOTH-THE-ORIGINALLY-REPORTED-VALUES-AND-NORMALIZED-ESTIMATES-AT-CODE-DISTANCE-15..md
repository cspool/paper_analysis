# <span id="page-10-0"></span>TABLE I HARDWARE RESOURCE COMPARISON. FOR FAIRER COMPARISON ACROSS DIFFERENT CODE DISTANCES, HELIOS AND QUEKUF ARE SHOWN WITH BOTH THE ORIGINALLY REPORTED VALUES AND NORMALIZED ESTIMATES AT CODE DISTANCE 15.

|               | Micro-B. | Helios |       | QUEKUF |       | Ours |
|---------------|----------|--------|-------|--------|-------|------|
|               |          | Orig.  | Norm. | Orig.  | Norm. |      |
| LUT (k)       | 867      | 889    | 614   | 309    | 463   | 108  |
| FF (k)        | NA       | 177    | 126   | 453    | 634   | 43   |
| BRAM tiles    | 3        | NA     | NA    | 548    | 828   | 252  |
| Freq (MHz)    | 43       | 75     | NA    | 238    | NA    | 163  |
| Code distance | 15       | 17     | 15    | 8      | 15    | 15   |

#### B. Hardware Scalability with Code Distance

Existing MWPM- and UF-based hardware decoders typically exhibit rapidly growing resource consumption and degraded clock frequency as the code distance increases. In contrast, our architecture is explicitly designed to scale more efficiently with distance, providing a more hardware-efficient solution toward larger-scale surface-code decoding.

Fig. 13 estimates FPGA resource costs as d scales from 3 to 25. Across this sweep we scale only the components addressed by lattice coordinates—primarily the multi-bank vertex/edge buffer of the clustering engine and the per-EFE

adjacency storage—while holding the rest of the design at its measured d=15 sizing. The scaled buffers grow analytically as  $O(2^{\lceil \log_2 d \rceil})$  due to power-of-two address quantization.

# <span id="page-10-0"></span>TABLE I HARDWARE RESOURCE COMPARISON. FOR FAIRER COMPARISON ACROSS DIFFERENT CODE DISTANCES, HELIOS AND QUEKUF ARE SHOWN WITH BOTH THE ORIGINALLY REPORTED VALUES AND NORMALIZED ESTIMATES AT CODE DISTANCE 15.

|               | Micro-B. | Helios |       | QUEKUF |       | Ours |
|---------------|----------|--------|-------|--------|-------|------|
|               |          | Orig.  | Norm. | Orig.  | Norm. |      |
| LUT (k)       | 867      | 889    | 614   | 309    | 463   | 108  |
| FF (k)        | NA       | 177    | 126   | 453    | 634   | 43   |
| BRAM tiles    | 3        | NA     | NA    | 548    | 828   | 252  |
| Freq (MHz)    | 43       | 75     | NA    | 238    | NA    | 163  |
| Code distance | 15       | 17     | 15    | 8      | 15    | 15   |

#### B. Hardware Scalability with Code Distance

Existing MWPM- and UF-based hardware decoders typically exhibit rapidly growing resource consumption and degraded clock frequency as the code distance increases. In contrast, our architecture is explicitly designed to scale more efficiently with distance, providing a more hardware-efficient solution toward larger-scale surface-code decoding.

Fig. 13 estimates FPGA resource costs as d scales from 3 to 25. Across this sweep we scale only the components addressed by lattice coordinates—primarily the multi-bank vertex/edge buffer of the clustering engine and the per-EFE

adjacency storage—while holding the rest of the design at its measured d=15 sizing. The scaled buffers grow analytically as  $O(2^{\lceil \log_2 d \rceil})$  due to power-of-two address quantization.

