# 9 Related Work

#### 9.1 GPU-to-CPU Migration

To address the disparity in parallelism between GPU programs and CPU architectures, researchers [\[42\]](#page-13-4) propose compiler transformations to group lightweight GPU workloads. This approach significantly reduces the number of threads required for the transformed CPU programs and is widely adopted in projects supporting SPMD programs on CPUs [\[7,](#page-12-5) [16,](#page-12-6) [35,](#page-13-10) [37,](#page-13-11) [40,](#page-13-12) [41,](#page-13-13) [47\]](#page-13-14). Han et al. [\[23\]](#page-12-8) highlight that transformed CPU programs are often incompatible with existing optimizations and propose novel compiler and runtime optimizations. Moses et al. [\[32\]](#page-12-9) propose to optimize the migration with a polyhedral model.

All existing GPU-to-CPU projects focus on single CPUs, whereas our solution extends migration to CPU clusters.

## 9.2 Single-Device to Multi-Device Migration

Researchers propose solutions to migrate programs written for a single device to execute on multiple devices. OmpSS [\[19\]](#page-12-27) and StarPU [\[3\]](#page-12-28) are frameworks that offload workloads to distributed nodes. These solutions focus on inter-kernel parallelism, where a single task is executed exclusively by one device. In contrast, our project focuses on intra-kernel parallelism, where multiple distributed nodes collaborate to execute a single task (i.e., GPU kernel).

Other intra-kernel parallelism solutions either rely on hardware-supported shared memory to maintain data consistency [\[12,](#page-12-29) [13\]](#page-12-30) or use peer-to-peer communication to synchronize CPU and GPU memory on the same node [\[29,](#page-12-31) [34\]](#page-13-15). Our work is the first to migrate a single GPU program to a CPU cluster, an environment with no hardware-supported shared memory and where peer-to-peer communication is too expensive for high performance.

## 9.3 Partitioned Global Address Space

PGAS is a parallel programming model that maintains a global memory space across distributed nodes. The global memory is partitioned among nodes, and PGAS provides primitives that allow each node to access memory located on other nodes. PGAS is a widely used model with many implementations (e.g., UPC++ [\[4\]](#page-12-10), SHMEM [\[11\]](#page-12-11)).

These solutions are designed for general programs, utilizing flexible but costly communication operations. However, for GPU-to-CPU-cluster migration, communication overhead is a significant concern due to the high volume of communication. Our work analyzes common patterns in GPU programs and proposes the use of coarse-grained collective communication to reduce network overhead.

