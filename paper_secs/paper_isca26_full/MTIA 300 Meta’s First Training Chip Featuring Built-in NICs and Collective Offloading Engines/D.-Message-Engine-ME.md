# *D. Message Engine (ME)*

As DLRM training stresses the communication infrastructure, the ME is designed to address GPU limitations and achieve the following objectives.

Avoid host involvement in the data path: By integrating NICs into the MTIA 300 package, we avoid the PCIe data path. Moreover, managing 1.2 TB/s of IO via the host CPU would slow down work submission and completion queue handling and consume a significant number of host cores. Therefore, we offload these tasks to the ME.

Offload collective operations from PEs: In GPUs, compute cores handle processing collective reduction operations, which can be area-inefficient. The MEs deliver similar reduction bandwidth using only one-third the area of PE cores, maximizing area efficiency for collective operations.

Reduce NoC contention: High-bandwidth collective communication on compute cores also stresses the NoC due to heavy traffic. Placing the MEs at the edges of the PE grid minimizes cross-grid congestion. (Figure 4).

Figure 5 shows the ME architecture designed to address these limitations. It consists of three main functional blocks.

CPU-M and peripherals: The ME contains a single scalar RISC-V core (CPU-M), similar to the PE vector cores, and 256 KB of context SRAM. An important feature is the single large shared Completion Queue (CQ) per ME, which eliminates the need to poll multiple queues and prevents CQ overflow.

NIC interface: As MTIA 300 includes 12 separate RDMA NICs within the package, we want to avoid the ME managing a significant number of doorbell addresses in software. This is handled by the NIC interface, which receives work requests

![](_page_5_Picture_0.jpeg)

Fig. 4: Each MTIA 300 network chiplet consists of six RoCE NICs.

![](_page_5_Figure_2.jpeg)

Fig. 5: Message Engine (ME) architecture.

(WRs) in a single FIFO and distributes them to the correct doorbells on the appropriate NICs.

**Near Memory Compute (NMC):** The NMC is a reduction block capable of 128 B/cycle for reductions or DMA, dropping to 96 B/cycle if all are active concurrently. The MEs can provide up to 2.8 TB/s of reductions, over twice the I/O bandwidth (1.2 TB/s). This block is used in all reduction-based collectives, including Reduce, AllReduce, and ReduceScatter.

