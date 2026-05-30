# **2 Model**

### **2.1 Fine-Grained Mixture-of-Experts**

MoE has emerged as a preferred architecture over dense models for building computeefficient large language models [\[Fedus et al.,](#page-22-1) [2022,](#page-22-1) [Jiang et al.,](#page-22-2) [2024,](#page-22-2) [Dai et al.,](#page-22-3) [2024,](#page-22-3) [Ludziejewski et al.,](#page-23-1) [2024\]](#page-23-1). The core idea of MoE is to replace each feed-forward layer (FFN) in a Transformer with a set of experts, where each expert is structurally identical to an FFN. Each input token is routed to only a subset of experts in each layer. The sparsity of expert activation ensures computational efficiency of an MoE layer.

Due to the vast diversity of multimodal data, we hypothesize that *expert specialization* is important for an multimodal MoE to understand input from different data distributions. To this end, we use a large number of fine-grained experts with smaller FFN hidden dimension than standard FFNs, similar to [\[Dai et al.,](#page-22-3) [2024\]](#page-22-3). In particular, ARIA has 66 experts in each MoE layer, 2 of the 66 experts are shared among all inputs to capture common knowledge, whereas 6 more experts are activated for each token by a router module. Table [2](#page-2-0) shows the detailed architectural configuration.

ARIA is significantly different from previous multimodal MoEs which either design modality-specific expert architectures or rely on upcycling from dense models [\[Lin et al.,](#page-23-2) [2024b,](#page-23-2) [Shen et al.,](#page-23-3) [2023,](#page-23-3) [Lin et al.,](#page-23-4) [2024a\]](#page-23-4). Our multimodal native MoE is pre-trained from scratch with modality-generic experts. In Section [4.2,](#page-5-0) we show that multimodal expert specialization naturally arises after pre-training.

| #total parameters | #activated parameters | #experts | #activated experts | expert FFN dim | hidden dim | #layers |
|-------------------|-----------------------|----------|--------------------|----------------|------------|---------|
| 24.9B             | 3.5B                  | 2△+64    | 2△+6               | 1664           | 2560       | 28      |

<span id="page-2-0"></span>Table 2: Architectural configuration of our MoE decoder. △ denotes shared experts.

### **2.2 Visual Encoder**

We design a lightweight visual encoder to convert visual inputs (i.e. images or video frames) into continuous visual tokens with the same feature dimension as word embeddings, which enables the MoE to seamlessly integrate visual and language inputs.

Drawing inspiration from previous work [\[Li et al.,](#page-23-5) [2023,](#page-23-5) [Bai et al.,](#page-22-4) [2023,](#page-22-4) [Laurençon et al.,](#page-22-5) [2024\]](#page-22-5), our visual encoder consists of a Vision Transformer (ViT) and a projection module. The ViT accepts images in their native aspect ratio as variable-length sequences of patches [\[Lee](#page-22-6) [et al.,](#page-22-6) [2023,](#page-22-6) [Dehghani et al.,](#page-22-7) [2023\]](#page-22-7), which preserves the inherent information structure in images. We categorize image size into three ranges: (1) medium-resolution images, where the longer edge is resized to 490 pixels; (2) high-resolution images, where the longer edge is resized to 980 pixels; (3) ultra-high-resolution images, where an image is dynamically decomposed into multiple high-res images, following a strategy similar to [Liu et al.](#page-23-6) [\[2024\]](#page-23-6). We initialize the weights of our ViT using the SigLIP-SO400M model [\[Zhai et al.,](#page-23-7) [2023\]](#page-23-7) and continue pre-train the ViT on our multimodal data.

Our projection module transforms the sequence of image embeddings from the ViT into a sequence of visual tokens. It comprises a single cross-attention layer and a FFN layer. The cross-attention layer employs a set of trainable vectors as queries and the image embeddings as keys. Medium-resolution images are processed by 128 queries, whereas high-resolution images are processed by an additional 128 queries (256 queries in total). The outputs from the cross-attention layer are then fed to an FFN, which then outputs visual tokens for the MoE decoder to further process.

#### **2.3 Infrastructure**

ARIA is trained on an extensively modified Megatron framework [\[Shoeybi et al.,](#page-23-8) [2019\]](#page-23-8). We eschew pipeline parallelism and instead implement a combination of expert parallelism [\[Lepikhin et al.,](#page-22-8) [2020\]](#page-22-8) and ZeRO-1 data parallelism [\[Rajbhandari et al.,](#page-23-9) [2020\]](#page-23-9) to optimize performance. Due to the carefully designed parallelism method and the small model size, ARIA can be effectively trained without using tensor parallelism, which significantly reduces communication overhead and enhances training efficiency.

We implement a load balancing loss to prevent routing collapse and encourage balanced expert activation. We find that the expert-level load balancing loss in previous work [\[Fedus](#page-22-1) [et al.,](#page-22-1) [2022,](#page-22-1) [Dai et al.,](#page-22-3) [2024\]](#page-22-3) is overly restrictive for our MoE due to the large number of

experts. Therefore, we relax the load balancing to groups of experts, where each group contains 8 fine-grained experts. We also employ z-loss [\[Zoph et al.,](#page-23-10) [2022\]](#page-23-10) to stabilize training.

