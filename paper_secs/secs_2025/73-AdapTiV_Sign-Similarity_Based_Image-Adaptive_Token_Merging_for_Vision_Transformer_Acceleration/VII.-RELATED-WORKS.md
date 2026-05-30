# VII. RELATED WORKS

Accelerator for Transformer/ViT Model. Various hardware accelerators [38]–[46] have been proposed since the advent of neural networks. Recent developments in hardware accelerators have predominantly targeted the attention mechanism [47]–[53], focusing on optimizing its computational efficiency. However, dedicated hardware accelerators for ViT models are still rare. A few proposed solutions, such as ViTCoD [34], HeatViT [54], and ViTALiTy [35], primarily focus on pruning or refining the attention layer. Consequently, AdapTiV, this work distinguishes itself by pioneering a TM scheme that aims to accelerate ViT by reducing computational load across the entire model.

Token Merging and its variations. Various adaptations of ToMe [12] have been actively explored. For instance,

ToFu [19] is a collaborative effort that combines token pruning and TM, observing that shallow and deep layers benefit differently from each optimization strategy, respectively. Despite numerous studies exploring and optimizing TM, to the best of our knowledge, AdapTiV represents an early attempt to explore the hardware efficiency and implementation of TM.

