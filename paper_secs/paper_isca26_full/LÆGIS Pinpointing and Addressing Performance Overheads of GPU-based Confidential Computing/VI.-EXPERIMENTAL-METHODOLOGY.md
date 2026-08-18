# VI. EXPERIMENTAL METHODOLOGY

<span id="page-9-1"></span>To evaluate LÆGIS under realistic hardware constraints, we extend GPGPU-Sim v4.2 (based on commit-id 84c6cf4) [\[67\]](#page-14-11), [\[68\]](#page-14-12), a widely used cycle-level GPU simulator, with UVMSmart [\[20\]](#page-13-13) support for virtual memory, including a shared GMMU and batch handling. Table [II\(](#page-9-2)A) shows the GPU configuration, and Table [II\(](#page-9-2)B) lists the HBM2 stack parameters. To better reflect real-world CC behavior, we also profile a physical setup (Table [II\(](#page-9-2)C)) and use these measured values in our simulations.

TABLE II: Configuration Parameters

<span id="page-9-2"></span>

| (A) GPU [68], [123], [124]                                                      |                                                     |  |
|---------------------------------------------------------------------------------|-----------------------------------------------------|--|
| 80 SMs, 64 warps/SM; shared GMMU; private TLB; crossbar network                 |                                                     |  |
| L2: 1 MB (64 KB/bank), 128B line (32B sector), 120-cycle latency                |                                                     |  |
| 8GB HBM2 (1 stack); Page size: 4 KB base, 2 MB big; PCIe: 64 GB/s per direction |                                                     |  |
| Pt: 51% (default), 1% (aggressive); Bt: 128; driver thread switch: 2 µs         |                                                     |  |
| IV cache: 16 entries, fully assoc., 20-cycle hit latency, LRU                   |                                                     |  |
| Fault preparation and idle time: profiled per base page and Pt                  |                                                     |  |
| (B) Memory Node [123], [125]–[127]                                              |                                                     |  |
| 8GB 4-Hi stack, 8 channels, 16 pseudo channels, 1KB page                        |                                                     |  |
| FR-FCFS; 16 banks, 4 bank groups/channel; 128-bit interface (BL4/channel)       |                                                     |  |
| 256 GB/s via 8 channels @1GHz; CL:RCD:RAS:WR:RP = 14:14:33:16:14                |                                                     |  |
| Static mapping: RR.RRRRRRRR.RRRRRRRB.BBCCBDDD.CCCSSSSS                          |                                                     |  |
| (C) CC Hardware [5], [128]                                                      |                                                     |  |
| CPU                                                                             | 5th Gen Intel Xeon 6530 Gold @2.1GHz, 32 cores      |  |
| System                                                                          | Supermicro SYS-421GE-TNRT3                          |  |
| Software                                                                        | Linux 6.2.0-mvp10v1+8-generic, OpenSSL v3.0.2       |  |
| Hypervisor                                                                      | QEMU 7.2.0 (TDX patched), TDX 1.5 (tag 2023ww15)    |  |
| GPU                                                                             | NVIDIA H100, PCIe 5.0, CUDA 12.4, Driver 550.163.01 |  |

Implementation Details. We extend GPGPU-Sim with a UVM batch handling model that follows prior UVM studies [\[20\]](#page-13-13)– [\[26\]](#page-13-20), [\[38\]](#page-14-17), including fault batching, adjustable prefetch thresholds, and tree-based prefetching. In addition to prior models, we explicitly model fault preparation time, driver thread idle windows, and page-encryption performance using profiles collected from real hardware under different prefetching/batching thresholds (Section [III\)](#page-3-1). We model the CPU-side encryption using measured kernel-space UVM encryption throughput (other options are discussed in Section [VIII-B\)](#page-11-1), and the GPUside design using an pipelined AES engine on the CE/GMMU path. We model the GPU IV Bank as HBM-resident metadata with firmware-managed address mapping, plus a small onchip IV cache with hit/miss tracking and a simple write-back policy. We also implement a pre-encryption scheduler that uses idle windows to select candidate pages from the current UVM context.

Workloads. We evaluate LÆGIS using 16 applications from widely adopted benchmark suites [\[20\]](#page-13-13), [\[129\]](#page-16-15)–[\[131\]](#page-16-16). Additionally, we include a CNN workload from [\[129\]](#page-16-15) and a standalone FlashAttention kernel [\[132\]](#page-16-17). These workloads generate a significant number of page faults, demonstrate diverse computational and memory access patterns, and have been extensively used in prior studies on UVM.

Methods. We implement and evaluate the following methods within our simulation framework:

- Baseline: Models existing GPU-based CC hardware, where encryption is tightly synchronized and lies on the critical path. Pages are encrypted and transferred in 4 KB granularity. Prefetching follows the default TBNp threshold. Overheads are modeled using profiled data. IV access is synchronized.
- Ideal: An idealized CC baseline where crypto-level overheads are perfectly hidden, while system-level costs (e.g., batch handling overhead) remain. No IV access is required.
- F-LÆGIS: Leverages *false idle* periods for preencryption. Candidate pages are selected based on the next available fault buffer entries when the previous batch is dispatched.
- IR-LÆGIS: Utilizes *true idle* periods to speculatively pre-encrypt pages. Candidate pages are randomly sampled from CPU-resident pages.
- IN-LÆGIS: Same policy to choose candidate pages as in F-LÆGIS but operates only during *true idle* periods.
- IFN-LÆGIS: Represents the full design of LÆGIS. It combines both false and true idle periods to maximize encryption utilization. Initially, candidate pages are chosen from the fault buffer. If idle time remains after processing those pages, additional available pages managed by the UVM driver are sequentially pre-encrypted.

Each LÆGIS variant requires IV access. For each method above, they apply default prefetching (Pt:51%) , we also evaluate an aggressive prefetching version of them (e.g., pIFN-LÆGIS).

