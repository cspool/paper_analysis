# A.7 ALGORITHMIC PROCEDURE OF DiffTopK

Following Equation 3, Differentiable Top-K solves the sigmoid threshold t during the forward pass via the method of undetermined coefficients and obtains t by binary search, yielding a smooth approximation to Top-K so that each probability  $M_i$  lies in [0,1]. Owing to the strict monotonicity of  $\sigma(\cdot)$ , its compensability under global shifts (via adjusting t), and its saturation behavior in the low-temperature or extreme-threshold limit, the mapping  $s \to M_{soft} = \sigma(s+t)$  satisfies monotonicity, invariance to global shifts, and convergence toward standard Top-K.

During backpropagation, the threshold t varies with s but they satisfy an implicit constraint. Differentiating this implicit equation yields a closed-form gradient, and the resulting backward computation appears in Algorithm 1.

