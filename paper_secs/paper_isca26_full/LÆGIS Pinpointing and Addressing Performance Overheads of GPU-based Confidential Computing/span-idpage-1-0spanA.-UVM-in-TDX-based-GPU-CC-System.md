# <span id="page-1-0"></span>A. UVM in TDX-based GPU CC System

As shown in Figure 2, GPU-based Confidential Computing (CC) [1]-[8] relies on both a CPU trusted execution

<span id="page-1-2"></span><sup>&</sup>lt;sup>1</sup>Section IV identifies an additional opportunity for CPU-side preencryption once the fault batch has arrived, during the fault preparation stage.

<span id="page-1-3"></span><sup>&</sup>lt;sup>2</sup>LÆGIS is derived from the Old Norse word for opportunity and is also a fusion of Lance and Aegis

<span id="page-1-4"></span><sup>&</sup>lt;sup>3</sup>Unless otherwise specified, the term page or big page refers to a VABlock, which is 2 MB in size. The term base page refers to a 4 KB page.

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Fig. 2: TDX-based GPU confidential computing architecture. The dashed green parts are trusted components.

environment (TEE), such as Intel Trust Domain Extensions (TDX) [11], [69]–[72], and a CC-capable GPU, such as the NVIDIA H100 [12], [73], [74]. CPU TEE achieves VM-level isolation that enables a transition from the insecure world to the secure world without code modification.

The new VM-level isolation provided by Intel TDX and AMD SEV-SNP [75] can incorporate the entire GPU runtime and kernel driver into a confidential virtual machine (CVM) or trust domain (TD) in Intel terminology.<sup>4</sup> Once the GPU is in the CC mode, it can communicate with the TD through encrypted channels. The TD is managed by a secure software component called the TDX Module [72], [76], which can be viewed as a lightweight hypervisor, as shown in Figure 1 and Figure 2. Similar to other CPU TEEs, memory encryption [46] is integrated for TD memory access.

The GPU architecture consists of SMs, architectural engines, a GPU memory management unit (GMMU), and memory partitions, as shown in Figure 2(b). The architectural engines include several copy engines (CEs) capable of initiating DMA operations that orchestrate data movement between the CPU and GPU [77]. In a CC-capable GPU, the CEs are enhanced with hardware AES engines to ensure confidentiality and authentication. There are several security-related engines, such as the GPU System Processor (GSP) and the Secure Processor (SEC2). Both are RISC-V microcontrollers, as noted by Gu et al. [7], [78], [79]. Like the CEs, the GSP/SEC2 integrate hardware AES engines to accelerate cryptographic operations. NVIDIA patents [78], [79] also indicate the presence of onchip fuses used to store security keys.

Once GPU CC is ready and all keys are securely established (1), we describe how Unified Virtual Memory (UVM) operates. First, user-level CUDA runtime API calls interact with the driver module (e.g., /dev/nvidia-uvm). For example, cudaMallocManaged allocates UVM-managed memory, which can be accessed by both the CPU and GPU without explicit copy operations. GPU page faults trigger page migration, which is a source of significant performance overhead. The kernel module then handles GPU requests, such as fault batch handling. When GPU threads attempt to access a page, they first check its page table entry (PTE) by performing a page table walk (PTW). If the PTE is invalid, a replayable fault is triggered and handled by the GMMU. The fault information is stored in a fault buffer [18] (2).

The GPU then interrupts the CPU for batch handling [18], which first fetches the fault information (3). To improve efficiency, the driver groups faults into batches [18]. We define fault batching count  $(\mathcal{B}_f)$  as the maximum number of faults that form a batch. Under UVM, the default value of  $\mathcal{B}_f$  is 256. The interrupt service routine (ISR) then **pre-processes** the fault batches for future service and replay (4). We refer these steps (3) and 4) before actual fault service as fault preparation. When CPU services these faults, it performs page encryption in AES-GCM format and pushes corresponding commands (such as DMA migration and GPU decryption) to GPU via a pushbuffer (5). This communication traverses through the untrusted PCIe channel. Therefore, the commands and subsequent data transfers (6) must be submitted to a secure channel with per-direction keys already established. Note that currently there is no dedicated AES engine to be used by the driver, therefore, encryption of pages is performed in software. Once the GPU receives the required pages, the corresponding engines decrypts them and store them in HBM as plaintext (see Section II-C for more details).

UVM organizes memory into 2 MB virtual address blocks (VABlocks), each consisting of contiguous 64 KB basic blocks. To reduce repeated batch handling, UVM uses a tree-based neighborhood prefetcher (**TBNp**) [20], [21], [27], [80] that proactively migrates nearby data. TBNp represents each VABlock as a five-level full binary tree with 32 leaf nodes, where each leaf corresponds to one basic block. It migrates at leaf granularity, so a fault on a base page migrates the entire leaf containing that page. TBNp uses a tree-based prefetching threshold ( $\mathcal{P}_t$ ): once the migrated-leaf fraction exceeds  $\mathcal{P}_t$ , TBNp assumes high locality and migrates the remaining leaves.  $\mathcal{P}_t$  controls prefetching aggressiveness, with lower values triggering prefetching earlier.

#### <span id="page-2-0"></span>B. Memory Encryption

The memory encryption engine (MEE) [46] is a hardware component in the memory controller that performs AES-based encryption and decryption for memory accesses, protecting data against physical memory attacks. Commercial TEEs commonly use two AES modes: counter-mode encryption (CME) and counter-less encryption (CLE). NVIDIA CC uses a variant of CME, AES-GCM [14], to protect data over the CPU-GPU interconnect [5], [6]. The GPU also includes onchip AES-GCM engines, as discussed in Section II-A. In contrast, TDX protects private CPU memory with a Total Memory Encryption-Multi-Key (TME-MK) engine [81], [82] in the memory controller, which performs AES-XTS [13], [83] encryption and decryption for all accesses to private TDX memory. However, currently there is no dedicated CPU-side AES-GCM engine available for CC. Since AES operates on 128-bit inputs, or AES blocks, encrypting a large message requires multiple AES invocations. For AES-GCM, the nonce used to generate the OTP must never repeat under the same key. The nonce is typically constructed from a 96-bit IV and a 32-bit self-incrementing counter [14]. This layout can be cus-

<span id="page-2-2"></span><sup>&</sup>lt;sup>4</sup>We use Intel and NVIDIA terminology in this study.

<span id="page-3-3"></span>![](_page_3_Figure_0.jpeg)

Fig. 3: Breakdown of CPU batch handling time (stacked bars) and processed fault batches (points). The left y-axis shows the time fraction of each component. The x-axis shows the TBN<sup>p</sup> threshold (Pt): H, M, and L denote 1%, 51%, and 91%, respectively. The left and right stacked bars use B<sup>f</sup> values of 256 and 1024. The right y-axis shows the total processed batches, normalized to the batches observed under P<sup>t</sup> of 1%. Collected on real hardware.

tomized [\[46\]](#page-14-7), as long as uniqueness is preserved.[5](#page-3-2) Section [III](#page-3-1) provides a more detailed analysis of the CC encryption stack.

