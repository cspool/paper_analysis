# <span id="page-17-0"></span>**A. Lossless Nature of Speculative Decoding**

The speculative decoding [\(Leviathan et al.,](#page-14-3) [2023;](#page-14-3) [Chen et al.,](#page-12-1) [2023\)](#page-12-1) can easily be justified to be lossless and identical to sample from *qtarget* alone, *i.e*., *pSD* = *qtarget*. Note that, given prefix *X*1:*<sup>j</sup>* , the next token sampled from:

$$x_{j+1} \sim \begin{cases} p_{draft}(x|X_{1:j}), & \text{if } \mathcal{U}(0,1) > \alpha, \\ norm(\max(0, q_{target}(x|X_{1:j}) - p_{draft}(\hat{x}|X_{1:j}))), & \text{otherwise,} \end{cases}$$

where *α* is the acceptance rate given by

$$\alpha(x) = \min\left(1.0, \frac{q_{target}(x)}{p_{draft}(x)}\right).$$

If the draft token is accepted, we have

$$p_{SD}(x|X_{1:j}; accepted) = p_{draft}(x|X_{1:j})\alpha(x|X_{1:j}) = \min(p_{draft}, q_{target}).$$

If the token is rejected, we have

$$\begin{split} p_{SD}(x|X_{1:j};rejected) &= (1-\alpha(x|X_{1:j}))norm(\max(0,q_{target}(x|X_{1:j})-p_{draft}(\hat{x}|X_{1:j}))) \\ &= (1-\alpha)\frac{q_{target}-\min(p_{draft},q_{target})}{1-\alpha} \\ &= q_{target}-\min(p_{draft},q_{target}) \end{split}$$

Therefore, the overall probability is given by

$$p_{SD}(x|X_{1:j}) = p_{SD}(x|X_{1:j}; accepted) + p_{SD}(x|X_{1:j}; rejected) = q_{target}((x|X_{1:j}))$$

Proved.

