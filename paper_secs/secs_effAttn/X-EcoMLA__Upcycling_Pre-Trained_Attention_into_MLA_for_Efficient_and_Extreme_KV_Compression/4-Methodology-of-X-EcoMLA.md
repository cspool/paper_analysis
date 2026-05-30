# 4 Methodology of X-EcoMLA

We begin with a pre-trained Transformer model referred to as the "base model". Our methodology in this paper concerns upcycling the attention modules in the base model into MLA modules to save the KV cache memory while remaining minimum training efforts and performance degradation. To achieve that, we first propose our SVD-based weight initialization approach to better inherit the knowledge from the pre-trained model. Additionally, our initialization approach offers both static and dynamic rank selection. After initialization, we adopt the knowledge distillation training process in MambaInLlama [Wang et al.](#page-12-5) [\(2024a\)](#page-12-5), which includes: end-to-end knowledge distillation, and direct preference optimization (DPO).

