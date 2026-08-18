# C. Logarithmic Number System

The Logarithmic Number System (LNS) simplifies various arithmetic operations [46], [48], as shown in Table II. However, addition and subtraction are more complex in LNS, so we adopt a hybrid approach: addition and subtraction are handled using ordinary arithmetic, while other operations are performed in LNS. Expression (2) shows a polynomial with five terms, where  $c_i$  is the coefficient and x is the variable. This polynomial is transformed into LNS using the equations in Table II, resulting in the expression in (3). To compute each term, we first apply a logarithmic transformation to x, then perform addition and multiplication in LNS. Finally, the antilogarithm transformation is applied to  $log_2c_i+k_i\times log_2x$ , and the terms are summed using ordinary arithmetic. This approach simplifies the original power operations into addition and multiplication. For instance,  $c_i x^9$  becomes  $2^{log_2 c_i + 9 \times log_2 x}$ , eliminating the need for numerous costly multipliers.

$$c_0 x^{k_0} + c_1 x^{k_1} + c_2 x^{k_2} + c_3 x^{k_3} + c_4 x^{k_4}$$
 (2)

$$2^{\log_2 c_0 + k_0 \times \log_2 x} + 2^{\log_2 c_1 + k_1 \times \log_2 x} + 2^{\log_2 c_2 + k_2 \times \log_2 x} + 2^{\log_2 c_3 + k_3 \times \log_2 x} + 2^{\log_2 c_4 + k_4 \times \log_2 x}$$
(3)

A related work [48] adopts the LNS to approximate nonlinear functions, but with the following limitations: (1) it only supports fixed point data; (2) it only supports Taylor series, lacking an efficient approximation algorithm.

## III. ERROR ANALYSIS

## A. Problem Formulation

Utilizing Chebyshev polynomials of at most n degree to approximate the given function f(x) can be defined as equation (4), where  $c_i$  is a constant coefficient and  $T_i(x)$  is the i-th order Chebyshev polynomial.

$$\tilde{f}(x) = \sum_{i=0}^{n-1} c_i T_i(x)$$
 (4)

Next, the software approximation  $\tilde{f}(x)$  is computed using LNS-based hardware to generate the result  $\tilde{f}_{HW}(x)$ . Consequently, the problem can be formulated as finding an efficient implementation of  $\tilde{f}_{HW}(x)$  to minimize the absolute approximation error  $\varepsilon(x)$ , defined by equation (5).

$$\varepsilon(x) = |f(x) - \tilde{f}_{HW}(x)| \tag{5}$$

By introducing  $\tilde{f}(x)$ , the error can be decomposed and bounded using the triangle inequality as follows:

$$\varepsilon(x) = |f(x) - \tilde{f}_{HW}(x)| \le \underbrace{|f(x) - \tilde{f}(x)|}_{\text{model error}} + \underbrace{|\tilde{f}(x) - \tilde{f}_{HW}(x)|}_{\text{implementation error}}$$
(6)

