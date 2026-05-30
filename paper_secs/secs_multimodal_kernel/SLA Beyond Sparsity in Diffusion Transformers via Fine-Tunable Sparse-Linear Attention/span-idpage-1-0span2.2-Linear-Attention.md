# <span id="page-1-0"></span>2.2 Linear Attention

Linear attention methods reduce the complexity of standard attention from  $\mathcal{O}(N^2d)$  to  $\mathcal{O}(Nd^2)$ . A key idea is to decouple the softmax operation by introducing a feature map  $\phi(\cdot)$  applied to Q and K. Specifically, it replaces the attention weights in standard attention with  $\frac{\phi(Q)\phi(K)^{\top}}{\operatorname{rowsum}(\phi(Q)\phi(K)^{\top}}$ . This reformulation enables reordering of the matrix multiplications: instead of explicitly computing the attention weights, it first computes  $\phi(K)^{\top}V$ , and then applies this intermediate result to  $\phi(Q)$ :

$$H = \phi(K)^{\top} V$$
,  $Z = \text{rowsum}(\phi(K)^{\top}) \in \mathbb{R}^{d \times 1}$ ,  $O = \frac{\phi(Q)H}{\phi(Q)Z}$ .

The mapping  $\phi(\cdot)$  is usually an activation function (e.g.,  $\mathrm{ELU}+1$  or  $\mathrm{ReLU}$  (Clevert et al., 2016; Xavier et al., 2011)). This formulation avoids explicitly constructing the  $N\times N$  matrices S,P and achieves linear computational complexity.

## 3 MOTIVATION AND ANLYSIS

