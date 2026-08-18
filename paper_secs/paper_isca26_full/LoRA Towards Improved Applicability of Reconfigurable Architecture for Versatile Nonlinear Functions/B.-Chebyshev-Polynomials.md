# B. Chebyshev Polynomials

Compared to the Taylor series, Chebyshev polynomials converge faster, minimize the maximum error over the entire interval, and require a lower degree for the same sup-norm error. They also avoid Runge oscillations and offer better numerical conditioning [13], [31]. As a result, Chebyshev polynomials are more efficient and stable for approximation.

Chebyshev polynomials of the first kind, defined as  $T_n(x)$ , form a key family of orthogonal polynomials on the interval [-1, 1]. An *n*-degree polynomial polynomial  $T_n(x)$  is defined by the following recurrence relations:

$$\begin{cases}
T_0(x) = 1 \\
T_1(x) = x \\
T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
\end{cases}$$
(1)

TABLE II
OPERATIONS IN ORDINARY AND LOGARITHMIC ARITHMETIC

| Operation      | Ordinary<br>Arithmetic | Transforming<br>Equation                          | Logarithmic<br>Arithmetic        |  |  |
|----------------|------------------------|---------------------------------------------------|----------------------------------|--|--|
| Multiplication | $x \times y$           | $2^{\log_2(x\times y)} = 2^{\log_2 x + \log_2 y}$ | $log_2x + log_2y$                |  |  |
| Division       | $x \div y$             | $2^{\log_2(x \div y)} = 2^{\log_2 x - \log_2 y}$  | $log_2x - log_2y$                |  |  |
| Logarithmic    | $log_b x$              |                                                   | $\frac{1}{log_2b} \times log_2x$ |  |  |
| Power          | $x^y$                  | $2^{\log_2(x^y)} = 2^{y \times \log_2 x}$         | $y \times log_2x$                |  |  |

