# <span id="page-3-2"></span>3.2 Group Ouerv Attention

Let the t-th token of the input sequence be  $\mathbf{x}_t \in \mathbb{R}^D$ , where D denotes the hidden dimension. To reduce the memory overhead of the KV cache, GQA divides the h query heads uniformly into g groups, with all query heads within a group sharing the same key and value vectors. Specifically, let  $W^Q \in \mathbb{R}^{hd \times D}$ ,  $\mathbf{W}^K$ ,  $\mathbf{W}^V \in \mathbb{R}^{gd \times D}$  and  $W^Q \in \mathbb{R}^{D \times hd}$  be the projection matrices for the query, key, value and output, where d = D/h denotes the dimension per head. GQA first computes the concatenated queries  $\mathbf{q}_t$ , keys  $\mathbf{k}_t$ , and values  $\mathbf{v}_t$ , and then slices them into heads or groups for attention computation:

$$[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; \dots; \mathbf{q}_{t,h}] = \mathbf{q}_t = W^Q \mathbf{x}_t, \tag{2}$$

$$[\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; ...; \mathbf{k}_{t,q}] = \mathbf{k}_t = W^K \mathbf{x}_t,$$
 (3)

$$[\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; ...; \mathbf{v}_{t,a}] = \mathbf{v}_t = W^V \mathbf{x}_t,$$
 (4)

where each  $\mathbf{q}_{t,i} \in \mathbb{R}^d$  corresponds to the query vector of the i-th head, and  $\mathbf{k}_{t,j}, \mathbf{v}_{t,j} \in \mathbb{R}^d$  correspond to the key and value vectors of the j-th group.

Using the notation in Section 3.1, after applying RoPE to  $\mathbf{q}_{t,i}$ ,  $\mathbf{k}_{t,i}$ , we can obtain the attention output for the *t*-th token as follows:

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \operatorname{softmax}_{j} \left( \frac{\mathbf{q}_{t,i}^{R^{\top}} \mathbf{k}_{j,\lceil i/\frac{h}{g} \rceil}^{R}}{\sqrt{d}} \right) \mathbf{v}_{j,\lceil i/\frac{h}{g} \rceil}, \tag{5}$$

$$\mathbf{y}_t = W^O[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,h}]. \tag{6}$$

As we can see, in GQA, each key and value head corresponds to  $\frac{h}{g}$  query heads. When g = h, GQA becomes MHA, and when g = 1, GQA becomes Multi-Query Attention (MQA).

