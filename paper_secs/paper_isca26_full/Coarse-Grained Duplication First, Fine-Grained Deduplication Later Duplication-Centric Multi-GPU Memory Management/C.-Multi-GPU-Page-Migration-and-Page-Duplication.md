# *C. Multi-GPU Page Migration and Page Duplication*

When a GMMU miss occurs, a far fault is triggered and handled by the host UVM driver via an interrupt. The runtime resolves the fault by migrating or duplicating the page to the requesting GPU, following prior mechanisms [8], [42]. This process involves (1) flushing in-flight SM instructions, caches, and TLBs of the owning GPU, and (2) migrating or duplicating the page before notifying the requesting GPU.

Migration suffers from ping-pong effects when multiple GPUs share a page. Access-counter–based policies mitigate this by triggering migration only after the access count exceeds a threshold (e.g., 256 on Volta GPUs [35]). However, performance can still degrade due to numerous remote accesses and frequent page-table invalidations [4]. Moreover, migration enforces single-owner placement, causing other GPUs to incur remote accesses until ownership changes.

To reduce these costs for read-mostly data, page duplication replicates pages across GPUs. In traditional schemes [42], duplication occurs on the first remote read, while any write triggers deduplication. Recent systems such as GPS [30] and GRIT [42] improve page placement by selecting between migration and duplication or broadcasting fine-grained updates to subscribers. However, their fine-grained designs underutilize NVLink bandwidth because only small amounts of data (e.g., 4KB–64KB) are transferred per migration or duplication.

# *C. Multi-GPU Page Migration and Page Duplication*

When a GMMU miss occurs, a far fault is triggered and handled by the host UVM driver via an interrupt. The runtime resolves the fault by migrating or duplicating the page to the requesting GPU, following prior mechanisms [8], [42]. This process involves (1) flushing in-flight SM instructions, caches, and TLBs of the owning GPU, and (2) migrating or duplicating the page before notifying the requesting GPU.

Migration suffers from ping-pong effects when multiple GPUs share a page. Access-counter–based policies mitigate this by triggering migration only after the access count exceeds a threshold (e.g., 256 on Volta GPUs [35]). However, performance can still degrade due to numerous remote accesses and frequent page-table invalidations [4]. Moreover, migration enforces single-owner placement, causing other GPUs to incur remote accesses until ownership changes.

To reduce these costs for read-mostly data, page duplication replicates pages across GPUs. In traditional schemes [42], duplication occurs on the first remote read, while any write triggers deduplication. Recent systems such as GPS [30] and GRIT [42] improve page placement by selecting between migration and duplication or broadcasting fine-grained updates to subscribers. However, their fine-grained designs underutilize NVLink bandwidth because only small amounts of data (e.g., 4KB–64KB) are transferred per migration or duplication.

