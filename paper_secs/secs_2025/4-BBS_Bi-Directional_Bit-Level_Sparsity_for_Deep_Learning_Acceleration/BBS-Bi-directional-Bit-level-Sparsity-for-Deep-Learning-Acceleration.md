# BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration

Yuzong Chen *Cornell University* New York, NY, USA yc2367@cornell.edu

Jian Meng *Cornell University* New York, NY, USA jm2787@cornell.edu

Jae-sun Seo *Cornell University* New York, NY, USA js3528@cornell.edu

Mohamed S. Abdelfattah *Cornell University* New York, NY, USA mohamed@cornell.edu

*Abstract*—Bit-level sparsity methods skip ineffectual zero-bit operations and are typically applicable within bit-serial deep learning accelerators. This type of sparsity at the bit-level is especially interesting because it is both orthogonal and compatible with other deep neural network (DNN) efficiency methods such as quantization and pruning. Furthermore, it comes at little or no accuracy degradation and can be performed completely post-training. However, current bit-sparsity approaches lack practicality because of (1) load imbalance from the random distribution of zero bits, (2) unoptimized external memory access because all bits are fetched from off-chip memory, and (3) high hardware implementation overhead, including large multiplexers and shifters to support sparsity at the bit level.

In this work, we improve the practicality and efficiency of bitlevel sparsity through a novel algorithmic bit-pruning, averaging, and compression method, and a co-designed efficient bit-serial hardware accelerator. On the algorithmic side, we introduce bidirectional bit sparsity (BBS). The key insight of BBS is that we can leverage bit sparsity in a symmetrical way to prune either zero-bits or one-bits. This significantly improves the load balance of bit-serial computing and guarantees the level of sparsity to be more than 50%. On top of BBS, we further propose two bit-level binary pruning methods that require no retraining, and can be seamlessly applied to quantized DNNs. Combining binary pruning with a new tensor encoding scheme, BBS can both skip computation and reduce the memory footprint associated with bi-directional sparse bit columns. On the hardware side, we demonstrate the potential of BBS through *BitVert*, a bitserial architecture with an efficient PE design to accelerate DNNs with low overhead, exploiting our proposed binary pruning. Evaluation on seven representative DNN models shows that our approach achieves: (1) on average 1.66× reduction in model size with negligible accuracy loss of < 0.5%; (2) up to 3.03× speedup and 2.44× energy saving compared to prior DNN accelerators.

*Index Terms*—Deep learning accelerator, bit-serial computing, hardware-software co-design, sparsity, model compression

