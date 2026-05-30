# <span id="page-13-11"></span>A Appendix: introduction to bi-level optimization

The bi-level optimization [\(Liu et al.,](#page-10-14) [2019\)](#page-10-14) optimize Θ conditioned on the optimized parameters of Ω ∗ . Denote the training set as Dtrain, and the validation set as Dval. The inner and outer levels of optimization are conducted on these two separate splits of the task dataset, which is analogous to validating architectures trained on Dtrain using a different split Dval to avoid over-fitting. Thus the optimization objective is:

$$\min_{\Theta} \mathcal{L}(\mathcal{D}_{val}, \Omega^*, \Theta),$$
s.t.  $\Omega^* = \arg\min_{\Omega} \mathcal{L}(\mathcal{D}_{train}, \Omega, \Theta),$  (7)

where L() is the objective function on a given downstream task, such as cross entropy loss. The above bi-level optimization problem is approximated with an alternating optimization strategy. The gradients of Ω are calculated with batches of samples from Dtrain, and the gradients of Θ are calculated on Dval.

