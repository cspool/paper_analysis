# <span id="page-4-0"></span>3.3 Computing the Estimator

Recall that we target to estimate the inner product  $\langle \mathbf{o}, \mathbf{q} \rangle$  to further estimate the squared distances (Section 2.2). We adopt the estimator of RaBitQ to inherit its unbiasedness and error bound <sup>8</sup> (see empirical verification in Section 5.2.1), i.e., we use  $\langle \bar{\mathbf{o}}, \mathbf{q} \rangle / \langle \bar{\mathbf{o}}, \mathbf{o} \rangle$  to estimate  $\langle \mathbf{o}, \mathbf{q} \rangle$ . The denominator  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$  is only related to the data vector and its quantized vector, so it can be pre-computed in the index phase.

<span id="page-4-3"></span> $<sup>^6\</sup>mathrm{It}$  is  $2^{B-1}$  instead of  $2^B$  because  $\bar{\mathbf{y}}$  is in the same orthant of  $\mathbf{o}'.$ 

<span id="page-4-5"></span> $<sup>^{7}</sup>$ For  $y_{cur}$ , we use two variables to store  $\langle y_{cur}, o' \rangle$  and  $\|y_{cur}\|$ . Whenever  $y_{cur}$  is updated, we can update these variables to obtain the new values of  $\langle y_{cur}, o' \rangle$  and  $\|y_{cur}\|$  efficiently.

<span id="page-4-6"></span><sup>&</sup>lt;sup>8</sup>This conclusion can be directly yielded from the proof in the original RaBitQ paper [27], i.e., Lemma 2.1 holds if (1) the codebook is a set of randomly rotated unit vectors; and (2)  $\bar{\mathbf{o}}$  is the nearest vector of  $\bar{\mathbf{o}}$  in the codebook.

Thus, we only need to compute  $\langle \bar{\mathbf{o}}, \mathbf{q} \rangle$ . Recall that  $\bar{\mathbf{o}} = P \frac{\bar{\mathbf{y}}}{\|\bar{\mathbf{y}}\|}$ . The following equations illustrate how it can be computed.

$$\langle \bar{\mathbf{o}}, \mathbf{q} \rangle = \left\langle P \frac{\bar{\mathbf{y}}}{\|\bar{\mathbf{y}}\|}, \mathbf{q} \right\rangle = \left\langle \frac{\bar{\mathbf{y}}}{\|\bar{\mathbf{y}}\|}, P^{-1} \mathbf{q} \right\rangle = \frac{1}{\|\bar{\mathbf{y}}\|} \left\langle \bar{\mathbf{y}}, \mathbf{q}' \right\rangle$$
 (11)

$$= \frac{1}{\|\bar{\mathbf{y}}\|} \left( \langle \bar{\mathbf{y}}_u, \mathbf{q}' \rangle - \frac{2^B - 1}{2} \sum_{i=1}^D \mathbf{q}'[i] \right) \tag{12}$$

Here  $\mathbf{q}'$  denotes  $P^{-1}\mathbf{q}$ ; Equation (11) applies an orthonormal matrix  $P^{-1}$  to both sides of the inner product; and Equation (12) expresses  $\bar{\mathbf{y}}$  with its quantization code  $\bar{\mathbf{y}}_u$ , i.e.,  $\bar{\mathbf{y}} = \bar{\mathbf{y}}_u - (2^B - 1)/2 \cdot 1_D$ .

Note that  $\|\bar{\mathbf{y}}\|$  is only related to the quantized vectors and can be pre-computed in the index phase.  $\sum_{i=1}^D \mathbf{q}'[i]$  is only related to the query vector. It can be computed once and its time costs can be shared by many data vectors. Thus, the remaining task is the computation of  $\langle \bar{\mathbf{y}}_u, \mathbf{q}' \rangle$ , i.e., the inner product between a vector of unsigned integers and a vector of floating-point numbers. When B=1 (the case of the original RaBitQ), RaBitQ's implementation can be directly applied [27]. When B equals to 4 or 8, the implementations in existing systems (for computing the inner product between a vector of 4-bit or 8-bit unsigned integers and a vector of floating-point numbers) can be directly applied [1, 17]. Other settings of B's can be implemented by splitting a vector of B-bit unsigned integers into several parts, where each part has the size of the power of 2 (e.g., a vector of 9-bit unsigned integers can be split into a binary vector and a vector of 8-bit unsigned integers). We will discuss the details of the idea later in Section 4.2.

