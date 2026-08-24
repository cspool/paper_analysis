# A.4. Tail Bound for Global Attention

Theorem A.5 (Tail Bound for Global Attention Mass). *Suppose* Ag[i, j] ∼ exp(−O(R)· *d*(i, j))*. Let* k<sup>g</sup> *denote the number of global attention entries below* ϵ *in each row. Then*

$$\epsilon \ge \exp\left(-O(R) \cdot \sqrt{\log\left(\frac{(Cw + w_q)(w_q - k_{\mathfrak{g}})}{\delta}\right)}\right).$$

*Proof.* Applying the union bound and exponential decay form as in the local case,

$$\mathbb{P}(A_{\mathfrak{g}}[i,j] > \epsilon) \le \frac{\delta}{(Cw + w_q)(w_q - k_{\mathfrak{g}})},$$

which yields the stated bound for ϵ.

## A.5. Sparsity of Global Attention

Theorem A.6. *For any fixed* ϵ > 0 *and* R = O( p log(Cw + wq))*, as* w<sup>q</sup> → ∞*,*

$$\lim_{w_q \to \infty} |\mathcal{S}_{\epsilon}(A_{\mathfrak{g}}[i,:])| = (Cw + w_q) - k_{\mathfrak{g}}, \qquad k_{\mathfrak{g}} = o(Cw + w_q).$$

*Proof.* Since the global attention decays exponentially in distance, as in the local case the number of entries above ϵ is k<sup>g</sup> = o(Cw + wq) for R = O( p log(Cw + wq)), ensuring overall sparsity as sequence length grows.

Discussion. While global attention has a larger support span, attention values still decay exponentially with distance; only a vanishingly small fraction of entries exceed a fixed threshold for sufficiently large R. This structure ensures the computational tractability and sparsity of the attention matrices, where R modulates the effective receptive field between local and global modeling.

