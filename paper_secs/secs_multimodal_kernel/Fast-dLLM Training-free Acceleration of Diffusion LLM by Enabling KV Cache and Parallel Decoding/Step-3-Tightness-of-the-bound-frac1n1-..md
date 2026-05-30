# Step 3: Tightness of the bound $\frac{1}{n+1}$ .

The bound  $\epsilon \leq \frac{1}{n+1}$  is tight. This means if  $\epsilon > \frac{1}{n+1}$ , one can construct a scenario where the marginal conditions  $p_j(X_{i_j} = x_{i_j}|E) > 1 - \epsilon$  hold, but  $\operatorname{argmax}_{\boldsymbol{z}} p(\boldsymbol{z}|E) \neq \boldsymbol{x}^*$  (which is  $\operatorname{argmax}_{\boldsymbol{z}} q(\boldsymbol{z}|E)$  as long as  $\epsilon \leq 1/2$ ).

Consider a vocabulary  $\mathcal{V}=\{0,1\}$  and let  $x_{i_j}=0$  for all j, so  $\boldsymbol{x}^*=(0,\dots,0)$ . For each  $j\in\{1,\dots,n\}$ , let  $\mathbf{e}_j$  be the vector with 1 at position j and 0 elsewhere. Let  $\eta=\frac{1}{n+1}(\epsilon-\frac{1}{n+1})>0$ . Set  $p(\mathbf{e}_j|E)=\frac{1}{n+1}+\frac{1}{n}\eta,\ \forall 1\leq j\leq n$  and  $p(\boldsymbol{x}^*|E)=\frac{1}{n+1}-\eta$ , then  $\boldsymbol{x}^*\notin \operatorname{argmax}_{\boldsymbol{z}}p(\boldsymbol{z}|E)$ . The marginal probabilities are:

$$p_j(X_{i_j} = 1|E) = p(\mathbf{e}_j|E) = \frac{1}{n+1} + \frac{1}{n}\eta, \ \forall 1 \le j \le n.$$

$$p_j(X_{i_j} = 0|E) = 1 - p_j(X_{i_j} = 1|E) = 1 - \epsilon_c = \frac{n}{n+1} - \frac{1}{n}\eta > 1 - \epsilon,$$

because

$$\frac{1}{n}\eta = \frac{1}{n(n+1)}(\epsilon - \frac{1}{n+1}) < \epsilon - \frac{1}{n+1}$$

So, the marginal condition  $p_j(X_{i_j}=x_{i_j}|E)>1-\epsilon$  (with  $x_{i_j}=0$ ) holds. As shown,  $\operatorname{argmax}_{\boldsymbol{z}} p(\boldsymbol{z}|E)$  can be made different from  $\boldsymbol{x}^*$ . Thus, if  $\epsilon>\frac{1}{n+1}$ , the argmax of p and q may not be the same.

**Step 4: Bound the**  $L_p$  **distance.** Let  $A_j$  be the event  $\{X_{i_j} = x_{i_j}\}$ .

$$D_p(p,q)^p = |p(x^*|E) - q(x^*|E)|^p + \sum_{z \neq x^*} |p(z|E) - q(z|E)|^p.$$

The term  $|p(\cap_{j=1}^n A_j|E) - \prod_{j=1}^n p(A_j|E)|$  (using  $p(A_j|E)$  for  $p_j(X_{i_j} = x_{i_j}|E)$ ) can be bounded. Since

$$1 - \sum_{j=1}^{n} \epsilon_{j}' \le p(\bigcap_{j=1}^{n} A_{j} | E) \le \min_{1 \le j \le n} p(A_{j} | E) = 1 - \max_{1 \le j \le n} \epsilon_{j}',$$

$$1 - \sum_{j=1}^{n} \epsilon_j' \le \prod_{j=1}^{n} (1 - \epsilon_j') = \prod_{j=1}^{n} p(A_j | E) \le 1 - \max_{1 \le j \le n} \epsilon_j'.$$

Thus.

$$|p(\boldsymbol{x}^*|E) - q(\boldsymbol{x}^*|E)| < (n-1)\epsilon.$$

For  $z \neq x^*$ :  $p(z|E) < \epsilon$  and  $q(z|E) < \epsilon$ . So,

$$|p(z|E) - q(z|E)| < \epsilon.$$

The sum  $\sum_{z \neq x^*} |p(z|E) - q(z|E)|$  can be bounded:

$$\sum_{\boldsymbol{z} \neq \boldsymbol{x}^*} |p(\boldsymbol{z}|E) - q(\boldsymbol{z}|E)| \leq \sum_{\boldsymbol{z} \neq \boldsymbol{x}^*} (p(\boldsymbol{z}|E) + q(\boldsymbol{z}|E)) = p(\boldsymbol{X} \neq \boldsymbol{x}^*|E) + q(\boldsymbol{X} \neq \boldsymbol{x}^*|E).$$

$$p(X \neq x^*|E) = 1 - p(x^*|E) < 1 - (1 - \sum_{j=1}^n \epsilon'_j) = \sum_{j=1}^n \epsilon'_j < n\epsilon.$$

$$q(X \neq x^*|E) = 1 - q(x^*|E) < 1 - \prod_{j=1}^{n} (1 - \epsilon'_j) \le \sum_{j=1}^{n} \epsilon'_j < n\epsilon.$$

So,

$$\sum_{z \neq x^*} |p(z|E) - q(z|E)| < 2n\epsilon.$$

Then.

$$\sum_{\boldsymbol{z} \neq \boldsymbol{x}^*} |p(\boldsymbol{z}|E) - q(\boldsymbol{z}|E)|^p \le (\sup_{\boldsymbol{z} \neq \boldsymbol{x}^*} |p(\boldsymbol{z}|E) - q(\boldsymbol{z}|E)|)^{p-1} \sum_{\boldsymbol{z} \neq \boldsymbol{x}^*} |p(\boldsymbol{z}|E) - q(\boldsymbol{z}|E)|$$
$$< \epsilon^{p-1} (2n\epsilon) = 2n\epsilon^p.$$

Therefore,

$$D_p(p,q)^p < ((n-1)\epsilon)^p + 2n\epsilon^p = ((n-1)^p + 2n)\epsilon^p.$$

So,

$$D_p(p,q) < ((n-1)^p + 2n)^{1/p}\epsilon.$$

For p = 1.

$$D_1(p,q) < (n-1+2n)\epsilon = (3n-1)\epsilon.$$

And for Total Variation Distance,

$$D_{TV}(p,q) = \frac{1}{2}D_1(p,q) < \frac{3n-1}{2}\epsilon.$$

#### Step 4: Bound the forward KL divergence.

$$D_{\mathrm{KL}}(p||q) = \sum_{\mathbf{z}} p(\mathbf{z}|E) \log \frac{p(\mathbf{z}|E)}{q(\mathbf{z}|E)} = I(X_{i_1}; \dots; X_{i_n}|E).$$

The conditional total correlation can be expanded using the chain rule:

$$I(X_{i_1}; \dots; X_{i_n}|E) = \sum_{k=2}^n I(X_{i_k}; X_{i_1}, \dots, X_{i_{k-1}}|E).$$

Each term is bounded by the conditional entropy:

$$I(X_{i_k}; X_{i_1}, \dots, X_{i_{k-1}}|E) \le H(X_{i_k}|E).$$

The conditional entropy  $H(X_{i_k}|E)$  is bounded. Since  $p_k(X_{i_k}=x_{i_k}|E)>1-\epsilon$ , it implies  $p_k(X_{i_k}\neq x_{i_k}|E)=\epsilon'_k<\epsilon$ . The entropy is maximized when the remaining probability  $\epsilon'_k$  is spread uniformly, leading to:

$$H(X_{i_k}|E) \le H_b(\epsilon'_k) + \epsilon'_k \ln(|\mathcal{V}| - 1) < H_b(\epsilon) + \epsilon \ln(|\mathcal{V}| - 1).$$

Summing (n-1) such terms (for  $k=2,\ldots,n$ ):

$$D_{\mathrm{KL}}(p||q) < (n-1)[H_b(\epsilon) + \epsilon \ln(|\mathcal{V}|-1)].$$

Remark 1. Assumption of a Well-Defined Joint  $p_{\theta}(X_{i_1}, \dots, X_{i_n} | E)$ : The theorem and proof rely on  $p_{\theta}(X_{i_1}, \dots, X_{i_n} | E)$  being a well-defined joint probability mass function from which the marginals  $p_{\theta}(X_{i_j} | E)$  are consistently derived. This implies that the joint PMF is coherent and its definition does not depend on a specific factorization order beyond what is captured by the conditioning on E. In practice, while MDM may not strictly satisfy this property, its behavior typically offers a close approximation. The theorem holds for an idealized  $p_{\theta}$  that possesses these properties. As MDMs become larger and more powerful, their learned distributions might better approximate such consistency.

Worst-Case Analysis: The conditions and bounds provided in the theorem (e.g.,  $(n+1)\epsilon \leq 1$ ) are derived from a worst-case analysis. This means the bounds are guaranteed to hold if the conditions are met, regardless of the specific structure of  $p_{\theta}(X|E)$  beyond the high-confidence marginal property. In practice, the actual case might be "better behaved" than the worst-case scenario. For instance, the dependencies between  $X_{i_j}$  and  $X_{i_k}$  (given E) might be weaker than what the worst-case construction assumes. Consequently, the argmax equivalence (Result 1) might still hold frequently even if  $(n+1)\epsilon$  is slightly greater than 1 (but not much larger). The condition identifies a threshold beyond which guarantees break down in the worst case, but practical performance can be more robust. Similarly, the actual  $L_p$  distances or KL divergence might be smaller than the upper bounds suggest if the true joint  $p_{\theta}(X|E)$  is closer to the product of marginals q(X|E) than the worst-case configurations.

