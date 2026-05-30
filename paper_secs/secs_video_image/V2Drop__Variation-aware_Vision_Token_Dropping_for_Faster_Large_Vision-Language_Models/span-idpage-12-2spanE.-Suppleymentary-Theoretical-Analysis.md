# <span id="page-12-2"></span>E. Suppleymentary Theoretical Analysis

Here, we present the complete theoretical proof that rigorously establishes the connection between token variation and model output through first-order analysis.

### E.1. Smoothness Assumption

We assume the model f has sufficient local smoothness in the representation space, such that the second-order remainder term in the Taylor expansion is bounded, satisfies:

$$||R_j|| = \mathcal{O}(||\Delta x_j^{(t)}||^2).$$
 (14)

This assumption is well- justified in Transformer-based LVLMs due to three architectural properties:

- Residual connections limit layer-wise changes, ensuring  $\|\Delta x_j^{(t)}\|$  remains small relative to  $\|x_j^{(t)}\|$ ;
- Layer normalization constrains the range of token representations, bounding higher-order derivatives;
- Smooth activations (e.g., GELU, SiLU) provide continuous second derivatives, ensuring Taylor expansion validity.

Under this assumption, for sufficiently small  $\|\Delta x_j^{(t)}\|$ , the quadratic term is negligible compared to the linear term, yielding:

$$\|\Delta f_i\| \approx \|J_i\|_{\text{op}} \cdot \|\Delta x_i^{(t)}\| \tag{15}$$

#### E.2. Justification of Bounded Jacobian Assumption

In the proof of Corollary, we assume that for all tokens j, the Jacobian operator norm is bounded below:  $||J_j||_{\text{op}} \ge \mu > 0$  for some constant  $\mu$ . Here is the proof for this assumption.

**Assumption (Non-degenerate Gradients).** The function f has non-degenerate gradients with respect to token representations, i.e., there exists  $\mu > 0$  such that:

$$||J_j||_{\text{op}} = \left\| \frac{\partial f}{\partial x_j^{(t+1)}} \right\|_{\text{op}} \ge \mu, \quad \forall j \in [n]$$
 (16)

This assumption is reasonable for the following reasons:

1. Information Flow in Transformers. In Transformer architectures, each token contributes to the final output through multi-head attention and feed-forward layers. The attention mechanism ensures that:

$$\frac{\partial \text{Output}}{\partial x_j} = \sum_{i=1}^n \frac{\partial \text{Output}}{\partial h_i} \cdot \frac{\partial h_i}{\partial x_j}$$
 (17)

where  $h_i$  are intermediate representations. Due to the soft-max normalization in attention, each token  $x_j$  receives non-zero attention weights from at least some positions, ensuring  $\|\frac{\partial \text{Output}}{\partial x_j}\| > 0$ .

**2. Residual Connections Preserve Gradients.** The residual structure  $x^{(t+1)} = x^{(t)} + \operatorname{Block}(x^{(t)})$  ensures that gradients flow directly through identity mappings:

$$\frac{\partial f}{\partial x_{j}^{(t)}} = \frac{\partial f}{\partial x_{j}^{(t+1)}} \cdot \left( I + \frac{\partial \text{Block}}{\partial x_{j}^{(t)}} \right) \tag{18}$$

The identity component I guarantees that gradients do not vanish, thus  $||J_j||_{\text{op}} \ge \mu$  for some  $\mu$  related to the minimum singular value of the identity component.

3. Layer Normalization Stabilizes Gradients. Layer normalization prevents gradient explosion and vanishing by maintaining bounded gradient norms across layers, ensuring  $\|J_j\|_{\text{op}} \in [\mu, M]$  for constants  $0 < \mu < M < \infty$ .

**Discussion:** What if  $||J_j||_{op} \to 0$ ?

What if some tokens have  $||J_j||_{op} \approx 0$ ? This would indicate that these tokens have negligible influence on the output. In such cases:

- These tokens can be safely dropped regardless of their variation magnitude
- Our method naturally handles this case: if  $\|J_j\|_{\text{op}} \approx 0$ , then  $\|\Delta f_j\| \approx 0$  regardless of  $\|\Delta x_j^{(t)}\|$ , so dropping them causes minimal performance degradation

Therefore, Assumption ( $||J_j||_{\text{op}} \ge \mu > 0$ ) is theoretically justified for vast majority of vision tokens in LVLMs.

