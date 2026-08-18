# A. Software / Hardware Implementation

Our design inherits practical experience from a real tapedout design. The proposed MLX architecture is a profiledriven, specialized subset of a general-purpose dataflow design implemented in Verilog RTL and synthesized in 12nm @1GHz using Synopsys DC. Profiling across FFT, BSMM, and dense LLM kernels revealed that many features of the general design were unnecessary for structured operators and impeded the multi-layer execution model. This enabled a streamlined architecture tailored explicitly for hybrid LLM workloads.

**Power & Area:** The floorplan of the full taped-out design (Fig. 14) serves as the reference for projecting area and power. Guided by the parameter analyses in Sec. IV-D, we reduce SIMD width from 32 to 8 and *remove unused units* such as vector shuffles, division, and high-precision floating-point pipelines. The resulting reduced design occupies only 10% of the area and 8% of the power of the original chip (Table II). The full-design power is measured post-silicon, while the reduced-design power is estimated from post-synthesis reports. **Performance:** We report performance using both (i) the cycle-accurate MLX simulator used during architectural exploration and (ii) measurements from the taped-out hardware. Both numbers will be reported to compare with counterpart accelerators with the same peak performance as shown in Table, IV.

**Software Deployment:** A RISC-V CPU serves as the host controller for our accelerator. To embed spatial accelerator bitstreams into C programs, developers write dataflow-style assembly specifying each PE's operations or use a LLVM-based C compiler [35] for programming. A lightweight "spatial assembler" then compiles this text format into binary and exports it as a header file for configuration on the MLX.

![](_page_9_Picture_0.jpeg)

|                    | Area-mm <sup>2</sup> | Power-mW    |
|--------------------|----------------------|-------------|
| Config Network     | 0.018                | 11.3        |
| Data Network       | 0.092                | 56.2        |
| Control Logic      | 0.011                | 7.5         |
| Tag Buffer         | 0.019                | 9.3         |
| Register File      | 0.044                | 28.7        |
| FU (SIMD32)        | 0.298                | 252.4 (70%) |
| PE (Skip-hop cost) | 0.482 (6.2%)         | 365.4       |
| PE Array           | 7.712                | 5846.4      |
| Ruduced (SIMD8)    | 0.772                | 433.8       |

Fig. 14: MLX floorplan. TABLE II: Area and Power.

## B. Benchmark Models and Hardware Baselines

- (1) From the algorithmic perspective, we evaluate our hybrid sparse method (FFT Compression and BSMM) on accuracy and computation reduction, using representative models of BERT, VIT as well as two LLMs Llama2-7B and InternLM2-7B, as detailed in Table III with their acronyms marked in parentheses. We also make a speedup comparison on the H100 for the two attention implementations of Llama2-7B.
- (2) From the architectural perspective, MLX provides a unified execution pattern for structured operators. To assess how this translates into hardware efficiency on LLM workloads, we evaluate MLX using a set of representative hardware baselines, as summarized in Table IV.

To ensure a fair and comprehensive comparison, we adopt a two-pronged evaluation strategy: the real taped-out design (1 TOp/s) is compared against an NVIDIA GPUs, while a reduced 256 GOp/s version is tuned in our simulator [36] to match several prior algorithm—accelerator co-designs with identical peak throughput [26, 29, 37, 38, 39, 40]. The performance numbers for these baselines are quoted directly from their original papers. To distinguish MLX's architectural benefit from algorithmic (ALGO) savings, we also list the FLOP reduction achieved by each prior work on T in the last row of Table IV. Jetson Xavier is chosen for its comparable peak performance (1.7 TFLOP/s vs. our 1 TFLOp/s) and identical 12 nm technology node. At last, We also compare against two more advanced GPUs (AGX Orin and RTX-3090) to demonstrate the generality of our efficiency gains.

