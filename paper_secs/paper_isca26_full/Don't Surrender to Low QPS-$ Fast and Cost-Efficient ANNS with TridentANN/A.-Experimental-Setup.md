# *A. Experimental Setup*

Platforms. We use a machine with the EPYC 7453 28 core CPU [13], 8×32 GiB DDR4 (actual < 32 GiB memory usage as [32, 40, 66, 69]), 8× 1-TiB Samsung PCIe-Gen4 NVMe SSDs [19], and 2 NVIDIA A2000-6GB GPUs [10]. The system runs on Ubuntu 22.04 with Linux 6.8, CUDA 12.1, GCC 11.4. KMeans (hierarchical KMeans built on it) we used is from Faiss 1.12 with default parameters [4].

Datasets. We use two widely-used billion-scale datasets for experiments, SIFT1B [51] and SPACEV1B [12], as [32, 75]. SIFT1B consists of descriptors extracted from images. Its train sets are 128-dim uint8 vectors. SPACEV1B consists of documents encoded by Microsoft SpaceV Superior model. The train sets are 100-dim int8 vectors. The metric is L2 distance. Additionally, we include GloVe [57] and NYTimes [1] datasets to study the impact of skewed data distributions. They contain 1.2M and 0.3M original vectors, with dimensionalities of 100 and 256, respectively. We limit the memory usage of GloVe and NYTimes proportionally to ensure a fair comparison.

Comparisons. We use four SSD-based ANNS systems which can be applied in mature commercial hardware as comparisons, including DiskANN [66], SPANN [32], PipeANN [40], and FusionANNS [69]. Parameters of [32, 40,

![](_page_8_Figure_15.jpeg)

Fig. 9: *Peak throughput and single-threaded latency using different 1B datasets, under Recall@10=90%.*

66] are set up according to their papers. Since the implementation of FusionANNS is not available, we report its performance and cost based on its paper, and additionally include a reproduction following [64]. The original FusionANNS is based on an NVIDIA V100-32GB GPU [7] with dual 32 core CPUs. Our reproduction uses an NVIDIA RTX A6000- 48GB GPU [11] (same-generation as our A2000, with required memory and reasonable price reference) on a comparable dualsocket platform with 56 cores of two EPYC 7453 CPUs.

Notably, this reproduction of FusionANNS can serve as the optimistic upper bound performance proven in [64]. First, to approximate its best-case performance without access to the proprietary GPU kernels, we precompute PQ distance results of each query, store them in device memory, and only move necessary candidates' resulted sorted to host memory for the coarse-grained filtering. This removes the runtime overhead of PQ computation on the GPU and yields the optimal GPU performance in theory. Second, to precisely control I/O overhead, we restrict the number of reads to match the recall-dependent I/O counts reported in the original paper and implement read operations with SPDK [3] as our TRIDENTANN.

# *A. Experimental Setup*

Platforms. We use a machine with the EPYC 7453 28 core CPU [13], 8×32 GiB DDR4 (actual < 32 GiB memory usage as [32, 40, 66, 69]), 8× 1-TiB Samsung PCIe-Gen4 NVMe SSDs [19], and 2 NVIDIA A2000-6GB GPUs [10]. The system runs on Ubuntu 22.04 with Linux 6.8, CUDA 12.1, GCC 11.4. KMeans (hierarchical KMeans built on it) we used is from Faiss 1.12 with default parameters [4].

Datasets. We use two widely-used billion-scale datasets for experiments, SIFT1B [51] and SPACEV1B [12], as [32, 75]. SIFT1B consists of descriptors extracted from images. Its train sets are 128-dim uint8 vectors. SPACEV1B consists of documents encoded by Microsoft SpaceV Superior model. The train sets are 100-dim int8 vectors. The metric is L2 distance. Additionally, we include GloVe [57] and NYTimes [1] datasets to study the impact of skewed data distributions. They contain 1.2M and 0.3M original vectors, with dimensionalities of 100 and 256, respectively. We limit the memory usage of GloVe and NYTimes proportionally to ensure a fair comparison.

Comparisons. We use four SSD-based ANNS systems which can be applied in mature commercial hardware as comparisons, including DiskANN [66], SPANN [32], PipeANN [40], and FusionANNS [69]. Parameters of [32, 40,

![](_page_8_Figure_15.jpeg)

Fig. 9: *Peak throughput and single-threaded latency using different 1B datasets, under Recall@10=90%.*

66] are set up according to their papers. Since the implementation of FusionANNS is not available, we report its performance and cost based on its paper, and additionally include a reproduction following [64]. The original FusionANNS is based on an NVIDIA V100-32GB GPU [7] with dual 32 core CPUs. Our reproduction uses an NVIDIA RTX A6000- 48GB GPU [11] (same-generation as our A2000, with required memory and reasonable price reference) on a comparable dualsocket platform with 56 cores of two EPYC 7453 CPUs.

Notably, this reproduction of FusionANNS can serve as the optimistic upper bound performance proven in [64]. First, to approximate its best-case performance without access to the proprietary GPU kernels, we precompute PQ distance results of each query, store them in device memory, and only move necessary candidates' resulted sorted to host memory for the coarse-grained filtering. This removes the runtime overhead of PQ computation on the GPU and yields the optimal GPU performance in theory. Second, to precisely control I/O overhead, we restrict the number of reads to match the recall-dependent I/O counts reported in the original paper and implement read operations with SPDK [3] as our TRIDENTANN.

