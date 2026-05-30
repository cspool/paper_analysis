# <span id="page-6-4"></span>4.3 A Balanced Approach to Joint Low-Rank Approximation of NoPE Keys and Values

In the previous section, we split the key heads into one carrying positional information and the others without positional information, achieving minimal loss. We then apply Principal Component Analysis

(PCA) jointly on the values and the non-positional components of the keys (i.e. NoPE-Key), using activations collected from a small calibration dataset, thereby compressing the projection matrices into a low-rank latent space. However, we observed that although the principal components of the keys were effectively separated with RoRoPE, the norm of the residual key features remained significantly larger than that of the Value. This imbalance caused the direct decomposition to favor principal component directions dominated by the keys.

To mitigate this, we scale  $W^{DK}$  by dividing it by

$$\alpha = \frac{\mathbb{E}_t[\|W_{\text{NoPE}}^{DK} \mathbf{x}_t\|_2]}{\mathbb{E}_t[\|W^{DV} \mathbf{x}_t\|_2]}$$
(20)

and correspondingly scale  $W^{UK}$  by multiplying it by  $\alpha$ . Here,  $W^{DK}_{\text{RoPE}} \in \mathbb{R}^{d \times D}$ ,  $W^{DK}_{\text{NoPE}} \in \mathbb{R}^{(g-1)d \times D}$  represent the parts of  $W^{DK}$  obtained after the operations described in the previous section, where  $W^{DK}_{\text{RoPE}}$  corresponds to one head that uses RoPE, and  $W^{DK}_{\text{NoPE}}$  corresponds to the remaining heads that do not use RoPE.

This transformation is mathematically equivalent and does not affect the overall model outputs, while significantly enhancing the effectiveness of KV cache compression in subsequent steps. More details is provided in the Appendix D.

