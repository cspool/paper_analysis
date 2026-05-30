# 5 Training SMOA with Cross-Layer Generalization (CLAG)

With the cross-layer shared pool of adapters, SMOA enables experts to be sparsely activated and reused across layers, thereby improving the use efficiency and reducing redundancy. To encourage the Cross-Layer Generalization (CLAG), we train a pool of adapters to gain diverse and specialized expertise complementary to the backbone base model. To this end, we introduce a training objective to enforce adapters' complementarity via a redundancy regularization in Section 5.1. The training algorithm is detailed in Section 5.2, with a curriculum learning strategy outlined in Section 5.3 to balance expert specialization and generalization.

