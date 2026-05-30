# APPENDICES

## A LOAD-BALANCING LOSS

As discussed in section 4, for load-balancing purposes, we want to define an additional loss function to encourage experts to receive roughly equal numbers of training examples. Unfortunately, the number of examples received by an expert is a discrete quantity, so it can not be used in backpropagation. Instead, we define a smooth estimator Load(X) of the number of examples assigned to each expert for a batch X of inputs. The smoothness allows us to back-propagate gradients through the estimator. This is the purpose of the noise term in the gating function. We define P(x, i) as the probability that G(x)<sup>i</sup> is nonzero, given a new random choice of noise on element i, but keeping the already-sampled choices of noise on the other elements. To compute P(x, i), we note that the G(x)<sup>i</sup> is nonzero if and only if H(x)<sup>i</sup> is greater than the k th-greatest element of H(x) excluding itself. The probability works out to be:

$$P(x,i) = Pr\Big((x \cdot W_g)_i + StandardNormal() \cdot Softplus((x \cdot W_{noise})_i) \\ > kth\_excluding(H(x), k, i)\Big)$$
(8)

Where kth\_excluding(v, k, i) means the kth highest component of v, excluding component i. Simplifying, we get:

$$P(x,i) = \Phi\left(\frac{(x \cdot W_g)_i - kth\_excluding(H(x), k, i)}{Softplus((x \cdot W_{noise})_i)}\right)$$
(9)

Where Φ is the CDF of the standard normal distribution.

$$Load(X)_i = \sum_{x \in X} P(x, i)$$
 (10)

We can now define the load loss to be the square of the coefficient of variation of the load vector, multiplied by a hand-tuned scaling factor wload.

$$L_{load}(X) = w_{load} \cdot CV(Load(X))^2 \tag{11}$$

Initial Load Imbalance: To avoid out-of-memory errors, we need to initialize the network in a state of approximately equal expert load (since the soft constraints need some time to work). To accomplish this, we initialize the matrices W<sup>g</sup> and Wnoise to all zeros, which yields no signal and some noise.

Experiments: We trained a set of models with identical architecture (the MoE-256 model described in Appendix C), using different values of wimportance and wload. We trained each model for 10 epochs, then measured perplexity on the test set. We also measured the coefficients of variation in Importance and Load, as well as ratio of the load on the most overloaded expert to the average load. This last value is significant for load balancing purposes on distributed hardware. All of these metrics were averaged over several training batches.

Table 6: Experiments with different combinations of losses.

|      |      |      | wimportance wload Test Perplexity CV (Importance(X)) CV (Load(X)) |      | max(Load(X))<br>mean(Load(X)) |
|------|------|------|-------------------------------------------------------------------|------|-------------------------------|
| 0.0  | 0.0  | 39.8 | 3.04                                                              | 3.01 | 17.80                         |
| 0.2  | 0.0  | 35.6 | 0.06                                                              | 0.17 | 1.47                          |
| 0.0  | 0.2  | 35.7 | 0.22                                                              | 0.04 | 1.15                          |
| 0.1  | 0.1  | 35.6 | 0.06                                                              | 0.05 | 1.14                          |
| 0.01 | 0.01 | 35.7 | 0.48                                                              | 0.11 | 1.37                          |
| 1.0  | 1.0  | 35.7 | 0.03                                                              | 0.02 | 1.07                          |

Results: Results are reported in Table 6. All the combinations containing at least one the two losses led to very similar model quality, where having no loss was much worse. Models with higher values of wload had lower loads on the most overloaded expert.

