# <span id="page-19-0"></span>D.1 The Error of Randomized Uniform Scalar Quantization

As is discussed above, the error introduced by the uniform scalar quantization is

$$\left|\left\langle \bar{\mathbf{x}}, \bar{\mathbf{q}} - \mathbf{q}' \right\rangle \right| = \left| \sum_{i=1}^{D} \bar{\mathbf{x}}[i] \cdot (\bar{\mathbf{q}}[i] - \mathbf{q}'[i]) \right| \tag{63}$$

Due to the randomized uniform scalar quantization presented in Section 3.3.1, each term of the error, i.e.,  $\bar{\mathbf{x}}[i] \cdot (\bar{\mathbf{q}}[i] - \mathbf{q'}[i])$ , is a random variable. The D random variables are independent to each other. Each term has the expected value of 0 (see Section 3.3.1) and has their values bounded by  $[-\Delta/\sqrt{D}, +\Delta/\sqrt{D}]$ . Now the question is to analyze the summation of D such random variables. We note that the Hoeffding's inequality immediately answers the question [82]. We restate the inequality in the following lemma.

LEMMA D.1 (HOEFFDING'S INEQUALITY [82]). Let  $X_1, ..., X_n$  be independent random variables, such that  $X_i \in [a_i, b_i], \forall 1 \leq i \leq n$ .

Let 
$$S_n = \sum_{i=1}^n X_i$$
. Then

$$\mathbb{P}\{|S_n - \mathbb{E}[S_n]| \ge t\} \le 2 \exp\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$
 (64)

In our case, we note that  $a_i = -\Delta/\sqrt{D}$ ,  $b_i = +\Delta/\sqrt{D}$ .  $\mathbb{E}[S_n] = \mathbb{E}\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n \mathbb{E}[X_i] = 0$ . It immediately yields the following conclusion.

$$\mathbb{P}\left\{\left|\sum_{i=1}^{D} \tilde{\mathbf{x}}[i] \cdot (\tilde{\mathbf{q}}[i] - \mathbf{q}'[i])\right| \ge t\right\} \le 2 \exp\left(-\frac{t^2}{2\Delta^2}\right) \tag{65}$$

<span id="page-19-3"></span>
$$\mathbb{P}\left\{\left|\sum_{i=1}^{D} \tilde{\mathbf{x}}[i] \cdot (\tilde{\mathbf{q}}[i] - \mathbf{q}'[i])\right| \ge \Delta u\right\} \le 2 \exp\left(-\frac{u^2}{2}\right) \tag{66}$$

where (66) is by letting  $u = t/\Delta$ . The conclusion shows that the error is bounded by  $O(\Delta)$  with high probability.

