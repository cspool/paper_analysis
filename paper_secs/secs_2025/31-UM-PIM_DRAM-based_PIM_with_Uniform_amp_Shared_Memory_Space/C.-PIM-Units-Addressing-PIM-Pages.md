# C. PIM Units Addressing PIM Pages

It is also essential for a PIM task running on PIM units to access the PIM pages in their local memory (the remaining CPU pages are not accessible). The PIM task program, however, is unaware of the hardware address of PIM pages allocated by OS at run time.

Therefore, we introduce a virtual address space for the PIM unit and share the CPN information with the CPU. In our example, two chunks are allocated. Therefore, the PAddr and VAddr space available to the PIM unit is  $2 \cdot S_B$ =256 kB. Fig. 4 (a) depicts the correspondence between PIM's VAddr and HWAddr. Compared with the CPU's address mapping, a PIM unit does not need to contain its own PIM unit ID. Bits 47-17 of VAddr indicate the PIM-side Chunk Number (PCN).

Suppose program of PIM unit i=0 accesses the same byte (the  $k^{\rm th}$  byte,  $k=0\mathrm{x}20002$ ) as our example in section III-B, it generates k as VAddr.

$$VAddr = k \tag{4}$$

![](_page_5_Figure_0.jpeg)

Fig. 5. Page migration cost and chunk list area under different chunk sizes.

Then the PCN is N=1. PCN is translated to CPN by address translation circuit, i.e. PIM Chunk List (PCL), which is presented in section IV-B. CPN of the PIM side is the same as that of the CPU side (= 0 xB). With the same address mapping rule, they point to the same HWAddr. Bits 16-0 of both VAddr and PAddr represent the PIM offset (off=2), which equals that of the CPU side. Finally, PCN and PIM offset are combined into PAddr 0 x 160002, and mapped to the same HWAddr as CPU side.

#### D. Discussion on Chunk Size.

The chunk size is a critical system hyper-parameter. Generally, allocating a huge page larger than 4MB may require page migration to accommodate it. The larger the chunk size is, there will be more conflict pages that need to be migrated. On the other hand, if the chunk size is set too small, more chunks must be allocated to provide enough space for PIM units' use. This in turn requires a larger PCL to store the CPNs, leading to increased overhead. In Fig. 5, we display the migration overhead and PCL area under different chunk sizes. As multiple allocations and releases of PIM chunks can be avoided in programming, these migration overheads are typically one-time occurrences. Therefore, the choice of chunk size can be biased towards reducing the PCL area. We select 256 MB as the chunk size, striking a balance between migration and PCL overhead. Allowing for existing CPU systems not supporting arbitrary-sized huge pages, the chunk size can be chosen from the huge page sizes that are supported by the system in practice.

