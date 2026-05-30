# <span id="page-12-0"></span>A.2. Proof of Theorem [4.4](#page-5-3)

Theorem A.2 (Restated, [4.4\)](#page-5-3). *Let* A(·) *be a linear operator and consider a sequence of inputs* a<sup>T</sup> , a<sup>T</sup> <sup>−</sup>1, . . . , a1*, with corresponding outputs* o<sup>T</sup> , o<sup>T</sup> <sup>−</sup>1, . . . , o1*. Given a quantization operator* Q*, we estimate the outputs using standard modulation:*

<span id="page-12-2"></span><span id="page-12-1"></span>
$$\tilde{\mathbf{o}}_t = \mathcal{A}(Q(\mathbf{a}_t - \mathbf{a}_{t+1})) + \tilde{\mathbf{o}}_{t+1},\tag{43}$$

$$\tilde{\mathbf{o}}_T = \mathcal{A}(\mathbf{a}_T),\tag{44}$$

*where* t = T − 1, . . . , 2, 1*. Similarly, we estimate the outputs using error-compensated modulation:*

$$\hat{\mathbf{o}}_t = \mathcal{A}(Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1})) + \hat{\mathbf{o}}_{t+1},\tag{45}$$

$$\hat{\mathbf{a}}_t = Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1}) + \hat{\mathbf{a}}_{t+1},\tag{46}$$

$$\hat{\mathbf{o}}_T = \mathcal{A}(\mathbf{a}_T), \quad \hat{\mathbf{a}}_T = \mathbf{a}_T, \tag{47}$$

*where* t = T − 1, . . . , 2, 1*. Suppose the quantization operator* Q *satisfies the following error bound:*

<span id="page-12-3"></span>
$$\|\mathbf{x} - Q(\mathbf{x})\|_2^2 \le c \|\mathbf{x}\|_2^2, \quad 0 < c < \frac{1}{2}.$$
 (48)

*Then, the estimation errors are bounded as follows:*

$$\|\mathbf{o}_t - \tilde{\mathbf{o}}_t\|_2^2 \le \sum_{k=t}^{T-1} 2^{T-k-1} c \|\mathcal{A}\|_2^2 \|\mathbf{a}_k - \mathbf{a}_{k+1}\|_2^2, \tag{49}$$

$$\|\mathbf{o}_t - \hat{\mathbf{o}}_t\|_2^2 \le \sum_{k=t}^{T-1} (2c)^{T-k-1} \|\mathcal{A}\|_2^2 \|\mathbf{a}_k - \mathbf{a}_{k+1}\|_2^2.$$
 (50)

*Proof:* Denote the error for standard modulation in Equation [\(43\)](#page-12-1) as e˜<sup>t</sup> and for error-compensation modulation in Equation [\(44\)](#page-12-2) as eˆ<sup>t</sup> at time step t. We first compute the error for standard modulation:

$$\tilde{\mathbf{e}}_t^2 = \|\mathbf{o}_t - \tilde{\mathbf{o}}_t\|_2^2 \tag{51}$$

$$= \|\mathbf{o}_t - \mathcal{A}(Q(\mathbf{a}_t - \mathbf{a}_{t+1})) - \tilde{\mathbf{o}}_{t+1}\|_2^2$$

$$(52)$$

$$= \|\mathbf{o}_t - \mathbf{o}_{t+1} - \mathcal{A}(Q(\mathbf{a}_t - \mathbf{a}_{t+1})) + (\mathbf{o}_{t+1} - \tilde{\mathbf{o}}_{t+1})\|_2^2$$
(53)

$$= \|\mathcal{A}(\mathbf{a}_t - \mathbf{a}_{t+1}) - \mathcal{A}(Q(\mathbf{a}_t - \mathbf{a}_{t+1})) + (\mathbf{o}_{t+1} - \tilde{\mathbf{o}}_{t+1})\|_2^2$$
(54)

$$= \|\mathcal{A}(\mathbf{a}_t - \mathbf{a}_{t+1} - Q(\mathbf{a}_t - \mathbf{a}_{t+1})) + (\mathbf{o}_{t+1} - \tilde{\mathbf{o}}_{t+1})\|_2^2$$
(55)

$$\leq 2|\mathcal{A}(\mathbf{a}_{t} - \mathbf{a}_{t+1} - Q(\mathbf{a}_{t} - \mathbf{a}_{t+1}))\|_{2}^{2} + 2\|\mathbf{o}_{t+1} - \tilde{\mathbf{o}}_{t+1}\|_{2}^{2}$$
(56)

Since ∥(ot+1 − o˜t+1)∥ 2 2 represents the error from the previous time step, applying the submultiplicative inequality yields:

$$\tilde{\mathbf{e}}_t^2 = \|\mathbf{o}_t - \tilde{\mathbf{o}}_t\|_2^2 \tag{57}$$

$$\leq 2\|\mathcal{A}\|_{2}^{2}\|\mathbf{a}_{t} - \mathbf{a}_{t+1} - Q(\mathbf{a}_{t} - \mathbf{a}_{t+1})\|_{2}^{2} + 2\mathbf{e}_{t+1}^{2}$$
(58)

$$\leq 2c\|\mathcal{A}\|_{2}^{2}\|\mathbf{a}_{t} - \mathbf{a}_{t+1}\|_{2}^{2} + 2\mathbf{e}_{t+1}^{2},\tag{59}$$

Accumulating the error from time T to t, we obtain Equation [\(49\)](#page-12-3).

For the error-compensation modulation, we compute:

$$\hat{\mathbf{e}}_t^2 = \|\mathbf{o}_t - \hat{\mathbf{o}}_t\|_2^2 \tag{60}$$

$$= \|\mathbf{o}_t - \mathcal{A}(Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1})) - \hat{\mathbf{o}}_{t+1}\|_2^2$$
(61)

$$= \|\mathcal{A}(\mathbf{a}_t) - \mathcal{A}(Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1})) - \mathcal{A}(\hat{\mathbf{a}}_{t+1})\|_2^2$$
(62)

$$= \|\mathcal{A}(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1} - Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1}))\|_2^2$$
(63)

<span id="page-12-4"></span>
$$\leq c\|\mathcal{A}\|_{2}^{2}\|\mathbf{a}_{t} - \hat{\mathbf{a}}_{t+1}\|_{2}^{2} \tag{64}$$

Next, we expand a<sup>t</sup> − aˆt+1:

$$\|\mathbf{a}_{t} - \hat{\mathbf{a}}_{t+1}\|_{2}^{2} = \|\mathbf{a}_{t} - Q(\mathbf{a}_{t+1} - \hat{\mathbf{a}}_{t+2}) - \hat{\mathbf{a}}_{t+2}\|_{2}^{2}$$
(65)

$$= \|\mathbf{a}_{t} - \mathbf{a}_{t+1} - Q(\mathbf{a}_{t+1} - \hat{\mathbf{a}}_{t+2}) + \mathbf{a}_{t+1} - \hat{\mathbf{a}}_{t+2}\|_{2}^{2}$$
(66)

$$\leq 2\|\mathbf{a}_{t} - \mathbf{a}_{t+1}\|_{2}^{2} + 2\|Q(\mathbf{a}_{t+1} - \hat{\mathbf{a}}_{t+2}) + \mathbf{a}_{t+1} - \hat{\mathbf{a}}_{t+2}\|_{2}^{2}$$

$$(67)$$

$$\leq 2\|\mathbf{a}_{t} - \mathbf{a}_{t+1}\|_{2}^{2} + 2c\|\mathbf{a}_{t+1} - \hat{\mathbf{a}}_{t+2}\|_{2}^{2}$$

$$\tag{68}$$

Substituting this into Equation [\(64\)](#page-12-4), we complete the proof. □

## <span id="page-13-0"></span>A.3. Proof of Corollary

Corollary A.3. *Let* x ∈ R <sup>d</sup> *be a vector, and let the quantization bandwidth be* b ∈ N*. Define the max-min dynamic quantizer as follows:*

$$s = \frac{\max(\mathbf{x}) - \min(\mathbf{x})}{2^b - 1},\tag{69}$$

$$\mathbf{z} = \left| -\frac{\min(\mathbf{x})}{s} \right|,\tag{70}$$

$$\mathbf{x}_{int} = clamp(\left\lfloor \frac{\mathbf{x}}{s} \right\rfloor + \mathbf{z}, 0, 2^b - 1). \tag{71}$$

*The corresponding dequantization is given by:*

$$Q(\mathbf{x}) = s(\mathbf{x}_{int} - \mathbf{z}). \tag{72}$$

*For any* 0 < c < <sup>1</sup> 2 *, we can revise* Q *with a new bandwidth* ˆb *satisfying:*

$$\|\mathbf{x} - Q(\mathbf{x})\|_2^2 \le c \|\mathbf{x}\|_2^2.$$
 (73)

*Proof:* From Theorem [4.3,](#page-5-2) we have:

$$\|\mathbf{x} - Q(\mathbf{x})\|_{2}^{2} \le \frac{(\max(\mathbf{x}) - \min(\mathbf{x}))^{2} d}{(2^{b} - 1)^{2}}$$
 (74)

$$\leq \frac{4\|\mathbf{x}\|_{\infty}^2 d}{(2^b - 1)^2} \tag{75}$$

$$\leq \frac{4\|\mathbf{x}\|_2^2 d}{(2^b - 1)^2} \tag{76}$$

To satisfy the desired bound, we choose ˆb such that:

$$\hat{b} \ge \log_2\left(\sqrt{\frac{4d}{c}} + 1\right). \tag{77}$$

Thus, the proof is complete. □

