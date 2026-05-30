# *C. Baseline*

CAIS is evaluated against 9 baselines in four categories, including 6 existing works and 3 NVLS-enhanced baselines.

• *Tensor Parallelism with NVLS* includes 1) Basic TP (TP-NVLS) [49] that partitions model layers across GPUs and applies AllReduce to merge intermediate results, and 2) TP with Sequence Parallelism (SP-NVLS) [25] that enhances TP by splitting AllReduce into ReduceScatter and AllGather phases, with layer normalization and

![](_page_9_Figure_0.jpeg)

Fig. 11: End-to-End Model Speedup Across Training and Inference.

![](_page_9_Figure_2.jpeg)

Fig. 12: Sub-layer Performance Speedup.

dropout/Add operations interleaved to reduce memory footprint. NVLS accelerates these collectives.

- *Overlap Solutions* includes 3) CoCoNet [19] and 4) FuseLib [44], both of which enable GEMM-AllReduce overlapping through software scheduling techniques, and 5) T3 [43] that introduces hardware-assisted fine-grained overlapping between GEMM and ReduceScatter. We extend T3 to also support AG-GEMM overlap in our evaluation. These solutions do not leverage NVLS.
- *Overlap Solutions with NVLS* includes 6) CoCoNet-NVLS, 7) FuseLib-NVLS, and 8) T3-NVLS, which are enhanced variants of overlap solutions by integrating NVLS support. As introduced in Sec. IV-A, CoCoNet-NVLS and FuseLib-NVLS utilize extended multimem instructions, T3-NVLS adopts a DMA-based NVLS design.
- *Locality-aware TB schedule* places TBs across GPUs/dies for reducing remote access, where we adopt the SOTA, 9) LADM [22]. LADM cannot utilize NVLS because of its communication-centric design.

