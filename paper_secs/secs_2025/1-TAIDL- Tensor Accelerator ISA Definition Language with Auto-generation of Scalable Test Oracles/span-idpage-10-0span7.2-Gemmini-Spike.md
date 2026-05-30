# <span id="page-10-0"></span>7.2 Gemmini Spike

Gemmini Spike [103] is a RISC-V ISA simulator [104] extension that models the Gemmini ISA [102], parameterized by the systolic array size DIM. We configured DIM as 16 (default), 64, 256, 1024. This only requires a single-line TAIDL change. Effectively, we evaluated four TAIDL-TOs against corresponding four Spike instances.

Benchmark Selection. The default Gemmini kernel library primarily supports tiled matrix multiplications and convolutions. Therefore, we chose the benchmark kernel as tiled matrix multiplication of the form  $C = A \times B + D$ , where A, C, D are of shape  $(I \cdot DIM) \times DIM$ , for some I, and B is of shape DIM $\times$ DIM. We varied I in powers of 2 from  $I_{min} = 1$  until the scratchpad ran out of space, i.e.,  $I_{max} \cdot DIM = 4096$ .

We observed similar simulation times for weight-stationary and output-stationary dataflow, with the latter used for reporting results. In §8.1, we discuss more complex kernels compiled using Exo [57].

Results & Observations. Figure 19 compares the performance of TAIDL-TO and Gemmini Spike simulator as kernel size increases for different configurations of DIM. We observed that the simulations using TAIDL-TO are orders of magnitude faster than Gemmini Spike for all configurations. The performance speedup increases as the kernel size increases, indicating the scalability of TAIDL-TO. For DIM = 1024, Gemmini Spike took over a minute for a simple matrix multiplication of size 1024×1024, whereas TAIDL-TO took only 9ms and 4ms on CPU and GPU (4 orders of magnitude faster). We discuss the reasons for these large speedups in §7.4.

#### <span id="page-10-1"></span>7.3 Intel SDE

Intel SDE is an emulator for Intel ISA extensions built on the Pin [10] dynamic binary instrumentation framework. Pin examines each static instruction in a program and asks Intel SDE if the instruction should be emulated or run natively. If an instruction is to be emulated, the Pin replaces it with a branch to an appropriate emulation function. We set Intel SDE to emulate only Intel AMX and AVX-512 instructions using sde64 -spr -force\_emulate skx. Rest of the x86 instructions, including AVX2 and non-SIMD, are executed natively.

Benchmark Selection. The Intel oneAPI Deep Neural Network Library (oneDNN) [50] is a deep learning library optimized for Intel processors, generating specialized implementations of certain kernels. In compiled oneDNN programs, we observed five recurring patterns of AMX and/or AVX-512 instructions – each pattern consists of repeated blocks differing only in memory addresses accessed. We used these five patterns as the benchmark kernels – two AMX-only, two AVX-512-only, and one mixed (AMX & AVX-512).

<span id="page-10-3"></span>

| Benchmark      | #BLOCKS = 4 |          | #BLOCKS = 256 |          |
|----------------|-------------|----------|---------------|----------|
| Dencimark      | #AMX        | #AVX-512 | #AMX          | #AVX-512 |
| cnn_inf_amx    | 40          | 0        | 2056          | 0        |
| rnn_inf_amx    | 24          | 0        | 1284          | 0        |
| sgemm_avx      | 0           | 238      | 0             | 9058     |
| mem_format_avx | 0           | 232      | 0             | 11320    |
| cnn_inf_mix    | 40          | 116      | 2056          | 7172     |

Table 2: Statistics of the selected oneDNN kernel benchmarks

10 KB

Kernel size

#### <span id="page-11-1"></span>(a) cnn inf amx (b) rnn inf amx (c) cnn inf mix Simulation time (in ms) 10 10 10 20 KB 20 KB 100 KB 600 KB 100 KB 600 KE 20 KB 100 KB 600 KE Kernel size Kernel size Kernel size (d) sgemm avx (e) mem format avx Simulation time (in ms) 10 10 Intel SDE TAIDL-TO (GPU) 10 TAIDL-TO (CPU)

#### TAIDL-TO for Intel AMX & AVX-512 vs. Intel SDE

Figure 20: The simulation time (lower is better) for the five selected oneDNN kernel benchmarks on Intel SDE and TAIDL-TO. The X-axis represents the kernel size, i.e., the total size of input and output tensors. Both axes are log-scaled. Similar to Figure 19, TAIDL-TO (GPU) is slower than TAIDL-TO (CPU) due to limited parallelization opportunities and kernel launch overhead.

10 KB

Table 2 provides the statistics of the instruction sequences in each benchmark. Each benchmark repeats its instruction sequence across multiple blocks, with the number of blocks (#BLOCKS) varying from 4 to 256 in powers of 2. The memory footprint scales linearly with #BLOCKS, as each block accesses a distinct memory region.

Results & Observations. Figure 20 compares the performance of TAIDL-TO and Intel SDE for selected benchmarks. We observed that the simulations using TAIDL-TO are significantly faster than Intel SDE for all benchmarks on CPU and three out of five benchmarks on GPU (except sgemm\_avx and mem\_format\_avx).

#### <span id="page-11-0"></span>7.4 Discussion: Performance Gains

We observed that the simulations using TAIDL-TO are orders of magnitude faster than the existing test oracles – Gemmini Spike and Intel SDE. The performance gains can be attributed to two main reasons – tensor optimizations and automatic parallelization of TAIDL-TO with the help of the XLA compiler.

Tensor optimizations. TAIDL defines the ISA semantics using XLA-HLO operators, which allows us to generate TAIDL-TO in a high-level XLA-HLO IR that can be optimized by domain-specific tensor compilers like XLA [31]. The generated TAIDL-TO leverages the optimizations present in the tensor compiler, such as operator fusion, algebraic simplification [35], and memory tiling & layout optimizations. Hand-crafted test oracles, like Gemmini Spike, are written in C++ with nested loops and conditionals. General-purpose compilers used to compile these simulators, like GCC or Clang, only offer limited optimizations across loops. Therefore, the generated TAIDL-TOs are more optimized than the existing counterparts.

Automatic parallelization. Hand-crafted test oracles are typically single-threaded and do not leverage the parallelism offered by modern multi-core CPUs or GPUs to speed up the simulations. While multi-threading can be added to these simulators using libraries like OpenMP [45], it requires manual effort to identify parallelizable

regions and add synchronization primitives. On the other hand, TAIDL-TO, due to its high-level IR representation, can be easily parallelized using tensor compilers like XLA that can automatically generate multi-threaded code and GPU kernels. These optimizations include loop parallelization, vectorization, and memory coalescing.

Breakdown analysis of simulation. We performed a breakdown analysis of the generated simulations for oneDNN kernels (Table 2). We observed that matrix multiply only constituted a small portion (~7%) of the simulation code. The simulation was dominated by memory read/write (33-60%) and layout transformations (16-60%). As a result, XLA optimizations are highly effective in making generated TAIDL-TOs orders of magnitude faster and more scalable.

Note that the performance gains are not specific to the simulators we evaluated but are a general trend observed across different ISAs and benchmarks. The key factor is the design of TAIDL, which enables ISA semantics to be defined in a high-level tensor IR like XLA-HLO, combined with the novel technique for generating tensor computation graphs to enable fast and scalable simulations.

