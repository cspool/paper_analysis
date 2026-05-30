# B THE ANALYSIS FOR THE CONCENTRATION PHENOMENON

In this part, we provide rigorous analysis for the concentration phenomenon presented in Section 3.2.1. In particular, we will analyze the expected value of  $\langle\bar{\mathbf{o}},\mathbf{o}\rangle$  (Section B.1), the extent of concentration of  $\langle\bar{\mathbf{o}},\mathbf{o}\rangle$  (Section B.2) and the joint distribution of  $(\langle\bar{\mathbf{o}},\mathbf{o}\rangle,\langle\bar{\mathbf{o}},\mathbf{e}_1\rangle)$  (Section B.3). We summarize the conclusions in Lemma B.3 and empirically verify them in Figure 8.

## <span id="page-16-2"></span>B.1 The Expected Value of $\langle \bar{o}, o \rangle$

As is analyzed in Section 3.2.1,  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$  and  $\langle \bar{\mathbf{o}}, \mathbf{e}_1 \rangle$  correspond to the projection of the random vector  $\bar{\mathbf{o}}$  onto two mutually orthogonal directions. In order to analyze the joint distribution of the random variables, let us first revisit the process of the generation of  $\bar{\mathbf{o}}$ . The generation of  $\bar{\mathbf{o}}$  involves two steps. <u>First</u>, the algorithm randomly transforms a deterministic codebook C into  $C_{rand}$  with a random orthogonal transformation P. <u>Second</u>, it chooses the vector  $\bar{\mathbf{o}}$  which has the largest inner product with  $\mathbf{o}$  from the vectors in  $C_{rand}$ . We next deduce from the definition (generation) of  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$  as follows.

$$\langle \bar{\mathbf{o}}, \mathbf{o} \rangle = \max_{\mathbf{x} \in C} \langle P\mathbf{x}, \mathbf{o} \rangle$$
 (28)

$$= \max_{\mathbf{x} \in C} \langle \mathbf{x}, P^{-1} \mathbf{o} \rangle = \max_{\mathbf{x} \in C} \sum_{i=1}^{D} \mathbf{x}[i] \cdot (P^{-1} \mathbf{o})[i]$$
 (29)

$$= \frac{1}{\sqrt{D}} \sum_{i=1}^{D} \left| (P^{-1} \mathbf{o})[i] \right| = \frac{1}{\sqrt{D}} \|P^{-1} \mathbf{o}\|_{\ell_1}$$
 (30)

where (28) is due to the process of generation of  $\bar{\mathbf{o}}$ . (29) is because the inner product is invariant to orthogonal transformation (rotation). (30) is due to the definition of our codebook C and the definition of  $\ell_1$  norm. Specifically, as is analyzed in Section 3.1.3, the entry of  $\mathbf{x} \in C$  can only be  $1/\sqrt{D}$  or  $-1/\sqrt{D}$ . To maximize the inner

product, we only need to pick the **x** which has its signs of the entries match the vector  $P^{-1}$ **o**. Thus, the result of the inner product is the summation the absolute values as is presented in (30), where  $\|\cdot\|_{\ell_1}$  is the  $\ell_1$  norm.

We note that  $\mathbf{o}$  is a unit vector. P is a random orthogonal transformation matrix (i.e., random rotation), whose inverse matrix (inverse rotation) is also a random orthogonal transformation matrix. Thus, the vector  $P^{-1}\mathbf{o}$  follows the uniform distribution on the unit sphere  $\mathbb{S}^{D-1}$  in the D-dimensional space  $\mathbb{R}^D$ . We note that the distribution is well studied [52, 82]. We restate some conclusions about the distribution with the following lemma.

<span id="page-16-10"></span><span id="page-16-0"></span>LEMMA B.1. ([52, 82]) For a D-dimensional random vector  $\mathbf{x} = (\mathbf{x}[1], \mathbf{x}[2], ..., \mathbf{x}[D])$  which follows the uniform distribution on the unit sphere, the probability density function of its every coordinate  $\mathbf{x}[1], \mathbf{x}[2], ..., \mathbf{x}[D]$  is given as

$$p_D(x) = \frac{\Gamma(\frac{D}{2})}{\sqrt{\pi}\Gamma(\frac{D-1}{2})} (1 - x^2)^{\frac{D-3}{2}}, x \in [-1, 1]$$
 (31)

where  $\Gamma(\cdot)$  is the Gamma function. The tail bound is given as

$$\mathbb{P}\left\{|\mathbf{x}[i]| > \frac{t}{\sqrt{D}}\right\} \le 2\exp\left(-c_0 t^2\right) \tag{32}$$

where  $c_0$  is a constant, i = 1, 2, ..., D.

Based on the explicit expression of the probabilistic density function, we next derive the expected value of  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$  as follows.

$$\mathbb{E}\left[\langle \bar{\mathbf{o}}, \mathbf{o} \rangle\right] = \frac{1}{\sqrt{D}} \cdot \mathbb{E}\left[\sum_{i=1}^{D} \left| (P^{-1}\mathbf{o})[i] \right|\right]$$
(33)

<span id="page-16-8"></span><span id="page-16-7"></span>
$$= \sqrt{D} \cdot \mathbb{E}\left[\left| (P^{-1}\mathbf{o})[1] \right|\right] \tag{34}$$

<span id="page-16-9"></span>
$$=\sqrt{\frac{D}{\pi}} \frac{\Gamma(\frac{D}{2})}{\Gamma(\frac{D-1}{2})} \cdot \int_{-1}^{1} (1-x^2)^{\frac{D-3}{2}} |x| dx \qquad (35)$$

<span id="page-16-11"></span>
$$=\sqrt{\frac{D}{\pi}}\frac{2\Gamma(\frac{D}{2})}{(D-1)\Gamma(\frac{D-1}{2})}$$
(36)

where (33) is due to (30). (34) is due to the linearity of expectation. (35) is due to Lemma B.1. (36) is by elementary calculus.

<span id="page-16-4"></span>We note that although the expected value of  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$  has a complicated form, i.e., (36), its numerical value is highly stable. When D ranges from  $10^2$  to  $10^6$ , the value ranges from 0.798 to 0.800, which is verified by the observations in Section 3.2.1 perfectly.

