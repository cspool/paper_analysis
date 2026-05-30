# A Skywork-MoE Architecture

As Skywork-MoE is upcycled from Skywork-13B, the MoE inherits most of the network configuration of the latter model, which is of Llama-like [\(Touvron et al.,](#page-9-14) [2023a,](#page-9-14)[b\)](#page-9-2) architecture featuring Rotary Positional Embedding (RoPE) [\(Su et al.,](#page-9-15) [2022\)](#page-9-15), RMSNorm [\(Zhang](#page-9-16) [and Sennrich,](#page-9-16) [2019\)](#page-9-16) and SwiGLU activation function [\(Shazeer,](#page-9-17) [2020\)](#page-9-17). Other details on Skywork-MoE is given in Table [2.](#page-10-0)

<span id="page-10-0"></span>

|                     | Skywork-MoE |
|---------------------|-------------|
| Vocab. Size         | 65,536      |
| Hidden Dim.         | 4,608       |
| FFN Dim.            | 12,288      |
| Head Dim.           | 128         |
| Num. Heads          | 36          |
| Num. Layers         | 52          |
| Num. Total Experts  | 16          |
| Num. Routed Experts | 2           |
| MoE Layer Frequency | 1           |
| Native Seq. Len.    | 8192        |

Table 2: Details on Skywork-MoE architecture.

