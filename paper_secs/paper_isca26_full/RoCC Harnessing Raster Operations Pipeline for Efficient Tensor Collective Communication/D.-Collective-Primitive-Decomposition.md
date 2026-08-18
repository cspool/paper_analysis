# D. Collective Primitive Decomposition

Once the new collective instructions are issued, ROPs begin the CC phase. As ROPs cannot execute the CC operations, which consist of multiple steps of arithmetic and data exchange operations, as one instruction, we propose *collective primitive decomposition* to bridge the semantic gap between CC operations and ROP operations. Our first insight is that all collective routines can be broken down into a handful of basic primitives, similar to NCCL's primitives such as *send*, *recv*, *recvReduceSend*, *recvReduceCopySend*, etc. With these primitives, any collective routines can be implemented. Figure 17 shows a four-segment ring algorithm for AllReduce built with these primitives.

```
struct RoCCDescriptor {
    // Base Address of Src
    U64 SrcBase,
    // Base Address of Dst
    U64 DatBase,
    U64 DataSize,
    U32 LocalRank,
    dim3 TensorDim,
    dim3 TileShape,
    // Type of Collective
    U4 CollType,
    U4 Datatype
    }
}

struct Doorbell
    {
    // Offset of tile data
    U64 Offset,
    // Address in Doorbell region
    U64 PayloadAddr
    // Rank of origin GPU
    U32 SrcRank,
    // Current stage
    U32 Stage
}
```

Fig. 16: Data structure introduced by RoCC.

(b) Doorbell Descriptor

(a) RoCC Descriptor

![](_page_7_Figure_0.jpeg)

Fig. 17: An example decomposition of a collective operation (AllReduce) to primitives, and micro-ops

Then, we further decompose each primitive to ROP µOps. For instance, the recvReduceCopySend primitive is compiled to a ReadDoorbell that receives the previous rank's doorbell packet together with the reduction result, a DepBarrier that ensures local GEMM tile completion, a second ReadDoorbell to retrieve the local GEMM results, an Add of the local GEMM results to the previous rank's data, a Write of the reduced results to the local memory, and a RingDoorbell to send a doorbell packet with the result to the next GPU rank. ReadDoorbell and RingDoorbell use ROP's memory load function and our proposed doorbell manager (Section V-E). Write uses ROP's memory store function, Add uses the arithmetic unit, and DepBarrier tracks warp-tile completion by checking Offset and Stage fields in the Doorbell descriptor (Section V-E). With this decomposition, every CC operation can be implemented and executed on ROP. Table I shows the full conversions from CC to primitives, and to μOps in a 4-GPU setup under ring algorithm. The other algorithms can be similarly implemented.

To decode a given collective operation into primitives and μOps, we add a *collective decoder* and a *primitive decoder*, as shown in Figure 18. The collective decoder converts a given collective operation to a sequence of primitives<sup>1</sup>. A lookup table is used to maintain a mapping between the collective operations and primitives. In our baseline, we support the most representative three collective operations (All-ToAll, AllReduce, AllGather), where each can be converted to a combination of five primitives (*send*, *recvReduceSend*, *recvReduceCopySend*, *recvOpySend*, *recv*). When eight GPUs are involved, the CC runs up to 15 stages. Thus, the table maintains 3 (operations) × 15 (stages) × 3-bit (to represent five primitive types) primitives, which take up 135 bits only.

The primitive decoder takes each 3-bit primitive and generates a sequence of  $\mu$ Ops. All the primitives can be handled with a sequence of up to six  $\mu$ Ops each, as shown in each row of Table I. There are five  $\mu$ Ops (*ReadDoorbell*, *Write*, *DepBarrier*, *Add*, and *RingDoorbell*). Thus, the primitive decoder maintains a lookup table of 5 (primitives)  $\times$  6 ( $\mu$ Ops per primitive)  $\times$  3-bit (to represent five  $\mu$ Ops types)  $\mu$ Ops, which takes 90 bits only. While most of the representative collective operations of the common 4- and 8-GPU nodes can be supported with these small lookup tables (a total of 225 bits), depending on the number of collective operations and

![](_page_7_Picture_6.jpeg)

Fig. 18: RoCC-enabled ROP architecture: modifications for RoCC are highlighted with non-gray colors.

the number of GPUs involved in the CC, the lookup tables may need to include more primitives. Thus, in our baseline, we project a total of 1KB lookup table to support all decoders.

A collective command buffer is also added to maintain the decoded  $\mu Ops$  so that the ROP can process one  $\mu Op$  per cycle. There are four entries in the collective command buffer, one per each of the four execution units. Each entry requires 8 bytes for both Src and Dst addresses, 3 bits for the five types of  $\mu Ops$ , and 1 bit for a valid bit. In total, the commanded buffer requires 66B for all four entries.

#### E. Doorbell, The Collective Messaging Scheme

To assist CC operation execution on ROP, we introduce doorbell, which is a messaging interface between SM and ROP and between ROPs in different GPUs. Doorbell uses a doorbell descriptor that specifies warp-specific information for the CC operations (Figure 16b). Inside of each ROP, the doorbell scheme employs two modules (Figure 18); a doorbell manager, which recognizes doorbell messages from the incoming requests/messages, and a doorbell buffer, which is a queue of doorbell descriptors under processing. We will show how these components are used for the multi-stage collective operations below. In GPU memory, a doorbell region is allocated to store intermediate tensor computation results shared among GPUs during the CC phase. The doorbell region is reserved by the GPU driver before the tensor processing. We consider that there are 32 entries in the doorbell buffer, TABLE I: Collective decoding to ROP µOps for 4-GPU ring.

| Collective<br>Type | Primitive                                                                                    | ROP μOps sequence<br>(trigggered by doorbell)                                                                                                                                                                                                                                                                                                                                            |
|--------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AllReduce          | send recvReduceSend recvReduceSend recvReduceCopySend recvCopySend recvCopySend recvCopySend | $\begin{array}{l} Rd \rightarrow Rng \\ Rd \rightarrow DepB \rightarrow ALU \rightarrow Rng \\ Rd \rightarrow DepB \rightarrow ALU \rightarrow Rng \\ Rd \rightarrow DepB \rightarrow Rd \rightarrow ALU \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \end{array}$ |
| AllGather          | send<br>recvCopySend<br>recvCopySend<br>recv                                                 | $\begin{array}{c} Rd \rightarrow Rng \\ Rd \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \rightarrow Rng \\ Rd \rightarrow Wr \end{array}$                                                                                                                                                                                                    |
| AllToAll           | send<br>recv                                                                                 | $\begin{array}{c} Rd \rightarrow Rng \\ Rd \rightarrow Wr \end{array}$                                                                                                                                                                                                                                                                                                                   |

Abbrev.: Rd=ReadDoorbell, Wr=Write, Rng=RingDoorbell, DepB=DepBarrier, ALU=operations using ROP ALU (e.g., add for reduction).

<sup>&</sup>lt;sup>1</sup>We follow NCCL's CC algorithm design [40] for primitives.

where each doorbell carries up to one tile. While a typical tile computed by SM tensor engines is a  $16 \times 16 \times 8$  FP16 block (4 KB), the entire block is not mapped on each MPU. Modern GPUs interleave physical memory across all memory partitions to maximize bandwidth. In our baseline architecture, we assume the memory interleaving granularity as 128 Bytes. Therefore, to sustain 32 concurrent in-flight tiles, we reserve a 4 KB (32 entries  $\times$  128 Bytes) doorbell region per MPU in GPU memory.

- 1) Doorbell Decoding: When an SM issues a RoCC instruction, the instruction request packet is sent to ROP with a doorbell flag set in the header. Then, the doorbell manager in the MPU (1) in Figure 18) recognizes it as a doorbell, and copies the request information to the doorbell buffer. Each entry of the doorbell buffer contains a doorbell descriptor (Figure 16b). The *Offset* is the requester warp's address offset in the target tile, PayloadAddr is the tile pointer address sent with the instruction (i.e., C tile array pointer in Listing 1), SrcRank is the current GPU's rank ID, and Stage indicates the current primitive stage. The same process is followed when a doorbell is received from a remote GPU, except for the handling of tile data. In the inter-GPU doorbell, the tile data is sent in the doorbell packet as a payload. Thus, the doorbell manager allocates space in the doorbell region and copies the payload to the allocated region. The newly allocated region address is filled in to PayloadAddr in the doorbell buffer.
- 2) **Doorbell Execution**: Once the decoded doorbells are filled in the doorbell buffer, the ROP fetches up to four doorbells each cycle to issue on the four execution units (Section IV-B). The collective decoder checks the CC operation and the current primitive stage from the descriptors, and generates the corresponding primitive (e.g., stage 2 of AllReduce executes recvReduceSend as shown in Figure 17). The primitive is then passed to the primitive decoder **2**. The primitive decoder translates each primitive into a series of  $\mu$ Ops and enqueues them to the collective command buffer **3**. The command generator issues four  $\mu$ Ops of different primitives each cycle **4**. The rest of the executions are handled the same way as the ROP datapath (Section II).

Once a primitive is completed, the doorbell manager increments the stage in the corresponding doorbell buffer entry. If the current stage is not the final stage, based on the collective operation, the doorbell manager creates a new doorbell packet by encoding the doorbell descriptor contents into the packet header and copying the computed tile to the payload. The packet is sent to the next GPU rank according to the collective operation in the RoCC descriptor.

3) Doorbell Manager: The doorbell manager is implemented with a state machine that is triggered upon receiving a new memory request and when a primitive is completed. When a new memory request is received, it checks the packet header to distinguish doorbell messages from regular memory requests by comparing the target address against the locally reserved doorbell region (Section V-E). If the doorbell flag is set in the header, the doorbell manager copies payloads to the doorbell region or buffer. This requires two one-bit

![](_page_8_Figure_5.jpeg)

Fig. 19: Memory mapping with symmetric tensor allocation: Green and gray pages (Data C, D) are the tiles used for the CC operations. Due to an extra memory allocation (Local Data) on GPU 0, the virtual addresses (numbers in each page) are not symmetric across GPUs, yet the pages are located at identical physical addresses across GPUs (numbers between GPUs).

comparators to check flags and two 32-bit registers to maintain the available doorbell region and buffer addresses. Upon completion of each primitive, the doorbell manager creates a memory request packet to be sent to the next GPU rank and issues a remote GPU memory access request via the existing inter-GPU memory mechanisms. This requires a 4-bit counter to increment the Stage in the doorbell descriptor and a 4-bit comparator to check if the final stage is reached. The memory packet creation and issue are done with the existing memory issue logic in the MPU. In total, the state machine runs over seven states. Thus, in each of the 32 entries in the doorbell buffer, 3 bits are reserved for the state information, and a total of 96 bits are reserved per ROP.

#### F. Symmetric Tensor Allocation

To send a doorbell to a peer GPU's correct ROP that processes the same tile as the requester ROP, the requester ROP must know the physical address of the tile in the peer GPU. However, individual GPUs may map the tile in different virtual/physical addresses, as determined by the GPU driver. If the virtual address is shared, the physical address could be translated via MMU. However, as ROPs access data directly from L2 cache, which typically uses physical address [41], using virtual address may cause runtime overhead.

To address this challenge, we propose *symmetric tensor allocation*, which maps tiles used for the CC operations on the same local physical address across GPUs. When an application (e.g., DL framework) allocates memory for CC with our custom GPU memory allocator, SymMalloc, the allocator examines the pool of unused physical frames on the GPUs involved in the communication and selects those having the same local physical addresses across the GPUs. Figure 19 shows that tensors allocated with SymMalloc are mapped on the same physical frames on both GPUs (Src on green pages and Dst on gray pages), while the virtual addresses (in the VM bar) are not symmetric, physical addresses are matching. Such a symmetric memory mapping enables ROPs to locate

tiles from identical address across GPUs. This saves doorbell packetization time, at the cost of one-time overhead at the memory mapping time.

Note that, as DL workloads typically occupy the entire GPU, our allocator can in general locate symmetric physical frames without requiring contiguity. If it cannot find a symmetric address, ROP reverts to a virtual-address-based solution by recording the tensor virtual address in the descriptor.

#### G. Discussions

Cache coherence implication: In modern GPUs, L2 is the last-level unified cache. As directly interfaced with dedicated memory devices, each L2 slice maintains the data in the devices exclusively (i.e., no sharing across L2 slices) [1], [19], [39]. Thus, RoCC does not incur coherence issues.

Concurrent execution of RoCC and raster operations: The doorbell manager directs doorbell traffic to a separate doorbell buffer, isolating it from the atomic command stream. Therefore, the two streams do not block each other at dispatch. Also, as RoCC operations share the existing datapath in the ROP by time sharing it with ROP operations, RoCC and ROP operations can run concurrently.

Performance overhead of symmetric tensor allocation: The lookup for symmetric physical addresses occurs at initialization, off the critical path. Common DL sharding strategies (data and tensor parallelism) use uniform GPU memory allocation, making such addresses easy to locate.

#### VI. EVALUATION

#### A. Methodology

We model a multi-GPU system using MGPUSim [52]. GPU architectures are configured by following NVIDIA V100, H100, and B200. V100 is the baseline and H100 and B200 are used for a sensitivity study. In the baseline architecture, each GPU has 80 SMs, a 6 MB LLC, and 64 memory partitions connected via an on-chip crossbar [59], [63], [64], with one ROP per partition (1 KB cache, 28-cycle datapath, four 3cycle ALUs) supporting four concurrent doorbells. GPUs form a 300 GBps full mesh, and CPU-GPU links follow PCIe Gen 4 with ≈150-cycle latency. To capture modern GEMM-CC workloads, we extend MGPUSim with SM-initiated kernels and NCCL-style ring collectives, implementing AllReduce, AllGather, and AllToAll. GEMM parameters are drawn from LLM feed-forward layers (Table II), and we evaluate Column-Linear (CL), RowLinear (RL), and AllToAll (A2A) in Expert Parallelism. For AllToAll, we simulate a stress test where every expert exchanges tokens with all others. Figure 20 shows the simulated execution time breakdown between GEMM and CC phases. The CC phase accounts for 16%-58.3% of total runtime, matching the breakdown in Figure 6b.

