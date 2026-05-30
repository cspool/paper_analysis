# VI. RELATED WORK

SpMM has been widely studied, with research focusing on computer architecture [31], [32], data formats [33], [34], and other aspects [35]–[38]. From the architectural perspective, Spade integrates an accelerator with the CPU to reduce data transfers [31], later evolving into HotTile for heterogeneous architectures [32]. Mentor employs a column-wise dataflow and software–hardware co-design to boost performance [39]. SpCache introduces an access pattern–aware cache to reduce bank conflicts [40], while Sparkle targets deep-learning SpMM with a flexible accelerator design [41].

From the data format perspective, researchers often integrate format characteristics with computing platform features. Embark dynamically allocates memory based on CSR and CSC formats, leveraging main and non-volatile memory properties [42]. FastSpMM adopts the ELLPACK format to improve the computation-to-memory ratio [43]. A CSR-based tiling approach enhances SpMM performance by improving tensor core utilization and computational density [44]. EC-SpMM optimizes GPU SpMM kernels [45], while DA-SpMM adaptively tunes GPU execution based on matrix characteristics [46].

