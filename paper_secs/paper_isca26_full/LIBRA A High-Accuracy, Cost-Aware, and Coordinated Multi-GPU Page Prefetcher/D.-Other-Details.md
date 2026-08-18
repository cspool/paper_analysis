# D. Other Details

A remaining consideration is how LIBRA handles scenarios where a single SM concurrently executes CTAs from different processes. A straightforward solution would be to index PPC entries by PID so that each process maintains its own access-pattern state. However, this approach can inflate the table size when many processes are active. Instead, we leverage a property of modern NVIDIA GPUs: process-level context switches are coarse-grained and occur infrequently. In practice, an SM typically runs CTAs from one process for long intervals before switching to another [31]. When a context switch does

occur, LIBRA simply reloads the MMAT state for the new process, rather than maintaining multiple concurrent entries.

## E. Summary of Components and Interaction

The only newly introduced hardware component is the multi-way multi-stride prefetcher, which includes the Trigger Table and MMAT. All other components are implemented in software, including the page prefetching coordinator and modifications to the CPU UVM driver and runtime.

The high-level interaction among these components operates as follows. Neither the CPU-side UVM runtime nor the GPUside UVM support polls GPUs or continuously reads memorymapped registers; instead, both operate on demand and respond to interrupts or events. The GPU-side UVM support receives hardware access-counter update events and updates the corresponding MMAT counters. When a GPU page fault occurs and the page has not recently triggered prefetching, the prefetcher performs prediction and sends a prefetch request containing access information to the CPU-side UVM runtime via an interrupt. The runtime processes the request using previously collected access information to determine whether and where to perform page prefetching or migration. If prefetching or migration is selected, the operation is executed using the existing CPU-side UVM runtime mechanisms, which issue the corresponding events to the relevant GPUs.

#### F. Area Overhead

Our design incorporates the MMAT as on-chip components, The MMAT includes 100 SMs, each with 4 ways; each way comprises a 36-bit VPN, four sets each with a 6-bit stride and counter, a 36-bit monitor VPN, and a 10-bit total access counter, totaling 6,500 bytes per GPU.

We also evaluate MMAT using CACTI [57]. The estimated read energy per access is 0.0051 nJ (5.094 pJ), and the write energy per access is 0.0062 nJ (6.174 pJ). The data array area is 0.00964041  $mm^2$ , and the tag array area is 0.0031502  $mm^2$ , resulting in a total MMAT hardware area of 0.01279061  $mm^2$ . MMAT requires 6,500 bytes per GPU, which corresponds to 52,000 bits. Assuming a standard 6T SRAM cell [9] and approximating one NAND2-equivalent gate as four transistors, each bit corresponds to about 1.5 NAND2 gates. This results in an estimated storage cost of approximately  $7.8 \times 10^4$  NAND2-equivalent gates.

## G. Multi-Rack GPU Support

We discuss the potential extension of LIBRA to multi-rack GPU systems. Since the detailed design of UVM support for multi-rack GPUs has not been publicly disclosed, we consider two possible designs based on page-table organization. The first design adopts a **Centralized Page Table**, where one rack acts as the master rack and maintains the unified page table for the UVM memory across all racks. The second design adopts **Partitioned Page Tables**, where each rack's UVM runtime maintains the page table only for rack-local UVM memory.

We focus on the second design, as the first can be derived similarly. Under the partitioned design, the UVM runtime already supports remote access and page migration by routing requests to the appropriate per-rack GPU UVM runtime. Each rack maintains recent PTEs that point to pages located in other racks, enabling remote accesses, while page migrations coherently update the page tables across racks.

To extend LIBRA to this multi-rack UVM system, two modifications are required. First, the PPC needs to be extended into a per-rack PPC, which manages prefetch requests for pages stored in that rack's UVM memory. Second, LIBRA's cost–benefit model must be updated to account for multi-rack communication characteristics. In particular, remote-access latency should distinguish between intra-rack and inter-rack accesses, and page migration overhead should incorporate the costs of cross-rack page-table updates, cross-rack TLB invalidations, and inter-rack data transfer latency.

