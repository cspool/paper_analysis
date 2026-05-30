# <span id="page-16-0"></span>**Strict Inequalities: GQA < MLA**<sub>Factorized</sub> < **MQA**

The relationships are strict:

 $\mathbf{GQA} < \mathbf{MLA_{Factorized}}$  When  $\mathbf{GQA}$  is represented as an  $\mathbf{MLA_{Factorized}}$  model, the up-projection matrices  $W^{UK}$  and  $W^{UV}$  must adopt specific sparse, block-selector structures. A general  $\mathbf{MLA_{Factorized}}$  model imposes no such constraints;  $W^{UK}$  and  $W^{UV}$  are typically dense and fully learnable. This allows a general  $MLA_{Factorized}$  to create h distinct key (and value) vectors by combining features from the  $r_{kv}$ -dimensional latent space in complex ways. GQA is restricted to g unique key (and value) vectors that are merely replicated h/g times. If h>g,  $MLA_{Factorized}$  can generate a richer set of interaction patterns. Thus,  $\mathrm{MLA}_{\mathrm{Factorized}}$  has strictly greater expressive power.

 $\mathbf{MLA_{Factorized}} < \mathbf{MQA}$  Consider the bilinear form  $\mathbf{x}_t^{\top} \mathbf{M} \mathbf{x}_j$  in the attention score. In  $\mathbf{MLA_{Factorized}}$ , for head i,  $\mathbf{M}_{MLA,i} = (W_i^Q)^\top W_i^{UK} W^{DKV}$ . The maximum rank of the transformation is determined by the smallest one among the ranks of  $W_i^Q \in \mathbb{R}^{d \times D}$ ,  $W_i^{UK} \in \mathbb{R}^{d \times 2gd}$ , and  $W^{DKV} \in \mathbb{R}^{2gd \times D}$ , which is at most d.

However, in the MQA form derived from MLA<sub>Factorized</sub>, the rank of the interaction matrix here,  $(W_i'^Q)^{\top}W^{DKV}$ , is determined by the smallest one among the ranks of  $W_i'^Q \in \mathbb{R}^{2gd \times D}$  and  $W^{DKV} \in \mathbb{R}^{2gd \times D}$ , which is at most 2gd.

Since  $2gd \ge d$ , MQA allows for a potentially higher-rank interaction between the (modified) query and the shared key representations compared to the per-head effective rank in MLA<sub>Factorized</sub>'s original formulation. This indicates that MQA has a greater representational capacity for the scoring mechanism.

