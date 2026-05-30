# <span id="page-15-0"></span>F. Theoretical Explanation of Attention Router Optimization

The primary challenge in optimizing an attention router lies in the non-differentiable nature of discrete selection. To enable the model to learn which attention heads are essential for a given task, we employ a continuous relaxation technique based on the Gumbel-Softmax (specifically, the Gumbel-Sigmoid for binary decisions) and the Straight-Through Estimator (STE).

<span id="page-15-1"></span><sup>12</sup>https://github.com/mit-han-lab/Block-Sparse-Attention

## F.1. Latent Representation and Logit Generation

Given the Key hidden states x<sup>K</sup> ∈ R s×H×d ′ , where s is the sequence length and d is the hidden dimension, the router first extracts a task-aware representation via a pooling operation and a two-stage MLP:

$$\mathbf{x}_{K}' = \text{Pooling}(\mathbf{x}_{K}), \mathbf{x}_{K}' \in \mathbb{R}^{H \times d'}$$
 (10)

The routing logits z for each head are then computed as:

$$\mathbf{z} = \mathrm{MLP}_{\mathrm{router}}(\mathrm{MLP}_{\mathrm{task}}(\boldsymbol{x}_K')), \tag{11}$$

where z ∈ R <sup>H</sup> represents the unnormalized preference for a specific attention mode (e.g., FA vs. SA) for each head.

### F.2. Differentiable Sampling via Gumbel-Sigmoid

To simulate the stochasticity of discrete routing while maintaining differentiability, we apply the reparameterization trick [\(Bhaskar et al.,](#page-8-6) [2025\)](#page-8-6). We introduce i.i.d. noise samples u ∼ Uniform(0, 1) and transform them into Gumbel noise g:

$$g = -\log(-\log(u + \epsilon) + \epsilon),\tag{12}$$

where ϵ is a small constant for numerical stability.

The discrete binary decision is then relaxed into a continuous approximation zˆsoft using a temperature-dependent Sigmoid function:

$$\hat{z}_{\text{soft}} = \sigma\left(\frac{\mathbf{z}+g}{\tau}\right) = \frac{1}{1 + \exp\left(-\frac{\mathbf{z}+g}{\tau}\right)}$$
(13)

Here, τ ∈ (0, ∞) is the temperature parameter. When τ → ∞, the distribution becomes uniform; as τ → 0, the output zˆsoft approaches a discrete Bernoulli distribution. In Section [F.3,](#page-16-1) we introduce the strategy for parameter τ .

## <span id="page-16-1"></span>F.3. Temperature Annealing Schedule

The optimization process utilizes an annealing schedule for τ to bridge the gap between the continuous relaxation and the discrete reality. We define the decay as:

$$\tau^{(t)} = \max(\tau_{\min}, \tau_{\text{init}} \cdot \exp(-r \cdot p)), \tag{14}$$

where p is the training step and r is the decay rate (we set r = 0.6). In the early stages of training, a high τ encourages exploration by providing dense gradients; in later stages, a low τ forces the router to converge toward the "hard" binary decisions used during actual deployment.

