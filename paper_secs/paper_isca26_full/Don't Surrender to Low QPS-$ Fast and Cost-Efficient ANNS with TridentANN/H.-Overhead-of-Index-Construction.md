# *H. Overhead of Index Construction*

We report overhead of index building on SIFT1B and SPACEV1B in Table IV. 28 CPU cores and 256 GB DRAM are used. GPU builds (GPU-Acc) additionally use an NVIDIA A6000-48GB GPU. SPANN takes the longest due to hierarchical clustering and dataset-wide RNG checking [70] for boundary vectors. Our CPU-only version avoids RNG checks by using candidate sets for boundaries, reducing 1B build time to about 2 to 2.5 days. With GPU acceleration of KMeans, it takes time within 1 day, and 10B-scale duplications (i.e., SIFT10B and SPACEV10B) build within 1 week.

# *H. Overhead of Index Construction*

We report overhead of index building on SIFT1B and SPACEV1B in Table IV. 28 CPU cores and 256 GB DRAM are used. GPU builds (GPU-Acc) additionally use an NVIDIA A6000-48GB GPU. SPANN takes the longest due to hierarchical clustering and dataset-wide RNG checking [70] for boundary vectors. Our CPU-only version avoids RNG checks by using candidate sets for boundaries, reducing 1B build time to about 2 to 2.5 days. With GPU acceleration of KMeans, it takes time within 1 day, and 10B-scale duplications (i.e., SIFT10B and SPACEV10B) build within 1 week.

