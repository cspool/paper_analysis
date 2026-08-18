# A. Problem Formulation: Single Interval Approximation

As discussed in Section III-A, the software approximation process involves finding a polynomial  $\tilde{f}(x)$  of degree  $\leq n$  that minimizes the error between  $\tilde{f}(x)$  and the target function f(x). For a given f(x) with an input range [a, b], the first step is to sample f(x) in this interval. Here, we set the maximum number of sample points to m, so there are m pairs of data  $\{(x_i, f(x_i))|i=1,...,m\}$ . For the sake of simplicity, we set n=5, leading to a polynomial that can have up to six terms (one constant term), which can be computed directly by one XCore. Subsequently, the approximation process is as follows.

**Step 1: Interval transformation.** Based on equation (11), the interval transformation is performed to map the input range [a, b] to [-1, 1], since Chebyshev polynomials are naturally

defined and orthogonal on [-1, 1]. In addition, the inverse transformation is defined by equation (12).

$$x \in [a,b] \to x^{'} = \frac{2x - (b+a)}{b-a}, x^{'} \in [-1,1] \tag{11}$$

$$x^{'} \in [-1, 1] \to x = \frac{(x^{'} + 1)(b - a)}{2} + a, x \in [a, b]$$
 (12)

Consequently, the m pairs of data is transformed into  $\{(x_i',f(\frac{(x_i'+1)(b-a)}{2}+a)|i=1,...,m\}.$  Step 2: Construct Chebyshev matrix. Since the approxi-

Step 2: Construct Chebyshev matrix. Since the approximation value  $\tilde{f}(x)$  is the sum of Chebyshev polynomials, we can construct the Chebyshev matrix (V) as equation (13).

$$V = \begin{bmatrix} T_0(x_1^{'}) & T_1(x_1^{'}) & T_2(x_1^{'}) & T_3(x_1^{'}) & T_4(x_1^{'}) & T_5(x_1^{'}) \\ \dots & \dots & \dots & \dots & \dots & \dots \\ T_0(x_m^{'}) & T_1(x_m^{'}) & T_2(x_m^{'}) & T_3(x_m^{'}) & T_4(x_m^{'}) & T_5(x_m^{'}) \end{bmatrix}$$

$$(13)$$

Subsequently, the approximation value for each x' can be obtained as follows, where  $c_{0\sim 5}$  are constant coefficients.

$$V \cdot [c_0, ..., c_5]^T = \left[\tilde{f}(x_1'), ..., \tilde{f}(x_m')\right]^T$$
 (14)

Consider the algebraic property: When the target function exhibits symmetry, exploiting its parity improves approximation efficiency. For example, there are only odd-order terms for an odd function, leading to a simplified V in equation (15).

$$V = \begin{bmatrix} 0 & T_1(x_1^{'}) & 0 & T_3(x_1^{'}) & 0 & T_5(x_1^{'}) \\ \dots & \dots & \dots & \dots & \dots \\ 0 & T_1(x_m^{'}) & 0 & T_3(x_m^{'}) & 0 & T_5(x_m^{'}) \end{bmatrix}$$
(15)

In addition, exploiting the parity can achieve a higher polynomial degree for the same number of terms, leading to improved accuracy and better numerical stability.

**Step 3: Solved by least squares.** To have a smoother and more stable approximation, we use the evaluation function in equation (16) to minimize the squared error.

$$minimize\left(\sum_{i=0}^{m-1} \left(f\left(\frac{(x_{i}^{'}+1)(b-a)}{2}+a\right)-\tilde{f}(x_{i}^{'})\right)^{2}\right)$$
(16)

**Problem formulation:** Finding coefficients  $[c_0, ..., c_5]^T$  to form a polynomial and minimize the evaluation function. This problem can be solved using the least squares method.

**Step 4: Coefficient transformation.** After obtaining  $[c_0,...,c_5]^T$ , the  $\tilde{f}(x)$  can be transformed as a standard polynomial based on the recurrence equations shown in equation (1). The transformed standard polynomial is shown in equation (17), where  $p_i'$  is a constant coefficient for each term.

$$\tilde{f}(x') = \sum_{i=0}^{5} c_i T_i(x') = \sum_{i=0}^{5} p'_i \times x'^i$$
(17)

It's worth noting that  $p_i'$  is obtained based on x'. Hence, to obtain the required  $\tilde{f}(x)$ , the coefficient  $p_i'$  should be transformed based on equation (11). Finally, the required  $\tilde{f}(x)$  can be obtained based on equation (18).

$$\tilde{f}(x) = \sum_{i=0}^{5} p'_{i} \times \left(\frac{2x - (b+a)}{b-a}\right)^{i} = \sum_{i=0}^{5} p_{i} x^{i}$$
 (18)

**Hardware Constraint:** *XCore* sums all the terms in the ordinary arithmetic system to compute a polynomial. Therefore, when the target format is fixed-point,  $p_i$  must be constrained to avoid overflow. Accordingly, the constraints in inequality (19) are incorporated into the least-squares solving process, where  $Q_{m.n}^{max}$  is the upper bound of the target fixed-point format.

$$|p_i||x^i|_{max} < |Q_{(m,n)}^{max}|, i = 0, 1, ..., 5$$
 (19)

The above steps demonstrate the core process of approximating f(x) in a given interval [a, b].

