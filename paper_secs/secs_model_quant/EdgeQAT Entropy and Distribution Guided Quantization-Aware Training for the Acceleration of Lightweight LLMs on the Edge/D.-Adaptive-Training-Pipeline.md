# *D. Adaptive Training Pipeline*

We visualize our training pipeline in Figure 4. We use the FP16 model (colored in yellow) to distill the quantized model (colored in blue) during QAT. We apply soft distillation, which trains a student model to mimic a teacher model by minimizing the KL divergence between their softmax outputs [12]. The distillation loss is defined as:

$$\mathcal{L}_{distill} = (1 - \gamma) \cdot \mathcal{L}_{CE} + \gamma \tau^2 \cdot \mathcal{L}_{KL}, \tag{15}$$

where τ is the temperature for the distillation, and γ is the coefficient balancing the KL divergence loss LKL and the cross-entropy loss LCE. In the quantization modules, the tokens are adaptively quantized with either 8 bits or 4 bits based on their scores (colored in red in Figure 4) generated from the most recent attention map.

The entropy loss L<sup>E</sup> and the distribution loss L<sup>D</sup> (both colored in green) are added to the total loss for optimization during training as follows,

$$\mathcal{L}_{total} = \mathcal{L}_{distill} + r_E \cdot \mathcal{L}_E + r_D \cdot \mathcal{L}_D, \tag{16}$$

where the ratios r<sup>E</sup> and r<sup>D</sup> are used to scale the entropy and distribution losses, respectively. In our experiments, we set r<sup>E</sup> = 0.5 and r<sup>D</sup> = 1 to facilitate better optimization.

