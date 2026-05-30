# Appendix

### <span id="page-17-1"></span>A Proofs for theorems

#### A.1 Proof for Theorem 3.2

Proof.

$$\mathbb{E}(\bar{o}) = \frac{1}{\mathcal{B}} \sum_{i=1}^{\mathcal{B}} \mathbb{E}[v_{i_j}] = \frac{1}{\mathcal{B}} \sum_{i=1}^{n} w_i v_i = o$$

$$\tag{12}$$

Assume  $\Sigma_1$  is the covariance matrix of  $\bar{o}$ ,  $\Sigma_2$  is the covariance matrix of  $v_i$ 

$$Tr(\Sigma_1) = \frac{1}{\mathcal{B}} Tr(\Sigma_2) = \frac{1}{\mathcal{B}} (\mathbb{E}[||v_i||^2] - ||\mathbb{E}[v_i]||^2) = \frac{1}{\mathcal{B}} (\mathbb{E}[||v_i||^2] - ||o||^2)$$
(13)

 $\mathbb{E}[||v_X||^2] - ||o||^2$  is a constant, so the trace of covariance matrix monotonically decreases with  $\mathcal{B}$ .

#### A.2 Proof for Theorem 3.3

Proof.

$$\mathbb{E}[|S|] = \mathbb{E}\left[\sum_{i=1}^{n} \mathbf{1}_{i \in S}\right] = \sum_{i=1}^{n} \mathbb{E}[\mathbf{1}_{i \in S}] = \sum_{i=1}^{n} (1 - (1 - w_i)^{\mathcal{B}}) = n - \sum_{i=1}^{n} (1 - w_i)^{\mathcal{B}}$$
(14)

Without loss of generality, let  $a_i = 1 - w_i$  and  $a_1 = \min_{1 \le i \le n} a_i = \epsilon$ , then

$$\mathbb{E}[|S|] = n - \sum_{i=1}^{n} a_i^{\mathcal{B}} = n - a_1^{\mathcal{B}} - \sum_{i=2}^{n} a_i^{\mathcal{B}}$$
(15)

$$= n - \epsilon^{\mathcal{B}} - \sum_{i=2}^{n} a_i^{\mathcal{B}} \tag{16}$$

 $f(x) = x^{\mathcal{B}}$  is convex function with  $\mathcal{B} \geq 1$  and  $x \geq 0$ . Then with Jensen's inequality, we have

$$\sum_{i=2}^{n} a_i^{\mathcal{B}} \ge (n-1) \left( \frac{\sum_{i=2}^{n} a_i}{n-1} \right)^{\mathcal{B}} = (n-1) \left( \frac{\left(\sum_{i=1}^{n} a_i\right) - a_1}{n-1} \right)^{\mathcal{B}}$$
(17)

$$= (n-1)(\frac{n-1-\epsilon}{n-1})^{\mathcal{B}} = (n-1)(1-\frac{\epsilon}{n-1})^{\mathcal{B}}$$
(18)

Let  $g(x) = (1-x)^{\mathcal{B}} + \mathcal{B}x - 1$ . We can prove  $g(x) \ge 0$  for any  $x \in (0,1), \mathcal{B} \ge 1$ . Then we have

$$\sum_{i=2}^{n} a_i^{\mathcal{B}} \ge (n-1)(1 - \frac{\epsilon \mathcal{B}}{n-1}) = n - 1 - \epsilon \mathcal{B}$$
(19)

Then we finally have

$$\mathbb{E}[|S|] = n - \epsilon^{\mathcal{B}} - \sum_{i=2}^{n} a_i^{\mathcal{B}} \le 1 + \epsilon \mathcal{B}$$
 (20)

