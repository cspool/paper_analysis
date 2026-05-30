# E.3. Connection to V<sup>2</sup>Drop Algorithm

**Proposition 1 (Dropping Strategy Justification).** Given n tokens at layer t, we aim to select  $|\mathcal{S}_{drop}| = \alpha n$  tokens to drop while minimizing total output perturbation:

$$S_{\text{drop}}^* = \underset{\substack{S \subseteq [n] \\ |S| = \alpha n}}{\operatorname{arg \, min}} \sum_{j \in S} \|\Delta f_j\| \tag{19}$$

**Proof.** By Theorem 1,  $\|\Delta f_j\| \approx \|J_j\|_{\text{op}} \cdot \|\Delta x_j^{(t)}\|$ . Under Assumption 2 ( $\mu \leq \|J_j\|_{\text{op}} \leq M$ ), we have:

$$\sum_{j \in \mathcal{S}} \|\Delta f_j\| \approx \sum_{j \in \mathcal{S}} \|J_j\|_{\text{op}} \cdot \|\Delta x_j^{(t)}\|$$

$$\in \left[\mu \sum_{j \in \mathcal{S}} \|\Delta x_j^{(t)}\|, M \sum_{j \in \mathcal{S}} \|\Delta x_j^{(t)}\|\right]$$
(20)

Since  $||J_j||_{\text{op}}$  varies within a bounded range, minimizing  $\sum_{j \in \mathcal{S}} ||\Delta f_j||$  is approximately equivalent to:

$$S_{\text{drop}}^* \approx \underset{\substack{S \subseteq [n] \\ |S| = \alpha n}}{\operatorname{arg\,min}} \sum_{j \in S} \|\Delta x_j^{(t)}\| \tag{21}$$

Therefore, V<sup>2</sup>Drop's strategy of selecting tokens with minimal variation  $\|\Delta x_j^{(t)}\|$  for dropping approximately minimizes total output perturbation, while computationally efficient (only requiring simple L2 norm computation).

#### **E.4.** Connection to information flow

In Transformer layers with residual connections:

$$x_{j}^{(t+1)} = x_{j}^{(t)} + \operatorname{Attn}(x_{j}^{(t)}) + \operatorname{FFN}(x_{j}^{(t)}), \tag{22}$$

the variation  $\Delta x_j^{(t)} = \operatorname{Attn}(x_j^{(t)}) + \operatorname{FFN}(x_j^{(t)})$  represents the *effective update* applied by the layer. Tokens with large  $\|\Delta x_j^{(t)}\|$  are those being actively refined by the network, indicating they carry task-relevant information being extracted and propagated to subsequent layers.

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 10. More visualization of token compression by V<sup>2</sup>Drop. The presented examples are from TextVQA, where grey masks indicate discarded visual tokens.