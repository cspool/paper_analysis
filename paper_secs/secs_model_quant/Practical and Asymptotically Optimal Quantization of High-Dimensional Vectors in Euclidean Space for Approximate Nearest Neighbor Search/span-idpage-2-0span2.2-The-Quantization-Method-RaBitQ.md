# <span id="page-2-0"></span>2.2 The Quantization Method RaBitQ

A recent paper proposes a new quantization method called *Ra-BitQ* [27], which quantizes a *D*-dimensional real vector into a *D*-bit string. It provides an unbiased estimator of squared distances and guarantees that the estimator has an asymptotically optimal error bound, which always holds regardless of the data distribution.

Specifically, given a raw data vector  $\mathbf{o}_r$  and a raw query vector  $\mathbf{q}_r$ , it first normalizes the vectors based on a vector  $\mathbf{c}$  (e.g., the centroid of a set of data vectors). Let  $\mathbf{o} := \frac{\mathbf{o}_r - \mathbf{c}}{\|\mathbf{o}_r - \mathbf{c}\|}$  and  $\mathbf{q} := \frac{\mathbf{q}_r - \mathbf{c}}{\|\mathbf{q}_r - \mathbf{c}\|}$  be the *normalized* data and query vectors. The (squared) Euclidean distance between  $\mathbf{o}_r$  and  $\mathbf{q}_r$  can be expressed as follows.

$$\|\mathbf{o}_r - \mathbf{q}_r\|^2 = \|(\mathbf{o}_r - \mathbf{c}) - (\mathbf{q}_r - \mathbf{c})\|^2$$
 (1)

$$= \|\mathbf{o}_r - \mathbf{c}\|^2 + \|\mathbf{q}_r - \mathbf{c}\|^2 - 2 \cdot \|\mathbf{o}_r - \mathbf{c}\| \cdot \|\mathbf{q}_r - \mathbf{c}\| \cdot \langle \mathbf{q}, \mathbf{o} \rangle$$
 (2)

Note that the distance  $\|\mathbf{o_r} - \mathbf{c}\|$  can be pre-computed in the index phase and  $\|\mathbf{q_r} - \mathbf{c}\|$  can be computed when a query comes and can be shared by many data vectors. Therefore, the computation of the distances between the raw vectors can be reduced to that of

the inner product of their normalized vectors. Then, it focuses on estimating the inner product of the normalized vectors <sup>2</sup>.

During the <u>index</u> phase, RaBitQ constructs a set C of all possible bi-valued unit vectors, each consisting of values of  $+\frac{1}{\sqrt{D}}$  and  $-\frac{1}{\sqrt{D}}$ . Then, it randomly rotates all vectors in C by multiplying them with a random *orthogonal matrix* [39] (a type of Johnson-Lindenstrauss Transformation) to form a quantization codebook  $C_r$ . The process can be described with equations as follows.

$$C_r := \left\{ P\mathbf{x} \mid \mathbf{x} \in C \right\}, \text{ where } C := \left\{ +\frac{1}{\sqrt{D}}, -\frac{1}{\sqrt{D}} \right\}^D$$
 (3)

where P is a random orthogonal matrix [39]. Note that the codebook is solely determined by the random orthogonal matrix P since the set of bi-valued unit vectors is pre-defined and does not rely on the data. Thus, it maintains the codebook  $C_r$  conceptually only by sampling and storing the matrix P. For each data vector  $\mathbf{o}$ , it finds the nearest vector  $\mathbf{o}_0$  in  $C_r$  as its quantized vector. The quantized vector is represented and stored as a quantization code  $\mathbf{v}_b \in \{0,1\}^D$  (a D-bit string) - recall that each quantized vector has a corresponding bi-valued unit vector, denoted by  $\mathbf{v}_0$ , in C. Specifically, we have  $\mathbf{v}_0 = P\mathbf{v}_0 = P\left(\frac{2}{\sqrt{D}}\mathbf{v}_b - \frac{1}{\sqrt{D}}\mathbf{1}_D\right)$  where  $\mathbf{1}_D$  is the D-dimensional vector whose coordinates are all ones.

During the **query** phase, it constructs an unbiased estimator for the inner product. The estimator has a theoretical error bound. We restate the estimator and its bound as follows.

<span id="page-2-5"></span>Lemma 2.1 (Restating Theorem 3.2 in [27]).  $\frac{\langle \bar{\mathbf{o}}_0, \mathbf{q} \rangle}{\langle \bar{\mathbf{o}}_0, \mathbf{o} \rangle}$  is an unbiased estimator of  $\langle \mathbf{o}, \mathbf{q} \rangle$ . With the probability of at least  $1 - \exp(-c_0 \epsilon_0^2)$ , its error bound is presented as

$$\left| \frac{\langle \bar{\mathbf{o}}_0, \mathbf{q} \rangle}{\langle \bar{\mathbf{o}}_0, \mathbf{o} \rangle} - \langle \mathbf{o}, \mathbf{q} \rangle \right| \le \sqrt{\frac{1 - \langle \bar{\mathbf{o}}_0, \mathbf{o} \rangle^2}{\langle \bar{\mathbf{o}}_0, \mathbf{o} \rangle^2}} \cdot \frac{\epsilon_0}{\sqrt{D - 1}} \tag{4}$$

where  $c_0$  is a constant and  $\epsilon_0$  is a parameter which controls the failure probability of the bound.

It is proven that using RaBitQ to quantize a D-dimensional vector to a D-bit string, the inner product  $\langle \bar{\mathbf{o}}_0, \mathbf{o} \rangle$  is highly concentrated around 0.8 [27]. Thus, the above lemma indicates that for estimating the inner product of two D-dimensional unit vectors, it guarantees a probabilistic error bound of  $O(1/\sqrt{D})$  with high probability, which achieves the asymptotic optimality [2]. In terms of the computation of the estimator, we note that  $\langle \mathbf{o}, \bar{\mathbf{o}}_0 \rangle$  is independent of the query and can be pre-computed before querying. The computation of  $\langle \mathbf{q}, \bar{\mathbf{o}}_0 \rangle$  can be conducted as follows.

$$\langle \mathbf{q}, \bar{\mathbf{o}}_0 \rangle = \left\langle \mathbf{q}, P\left(\frac{2}{\sqrt{D}}\bar{\mathbf{x}}_b - \frac{1}{\sqrt{D}}\mathbf{1}_D\right) \right\rangle$$
 (5)

<span id="page-2-4"></span><span id="page-2-3"></span>
$$= \frac{2}{\sqrt{D}} \left\langle \mathbf{q}', \bar{\mathbf{x}}_b \right\rangle - \frac{1}{\sqrt{D}} \sum_{i=1}^{D} \mathbf{q}'[i] \tag{6}$$

<span id="page-2-6"></span>Here,  $\mathbf{q}'$  denotes  $P^{-1}\mathbf{q}$  and  $\mathbf{q}'[i]$  denotes the i-th dimension of the vector  $\mathbf{q}'$ ; (5) plugs in the definition of  $\bar{\mathbf{o}}$  and (6) applies  $P^{-1}$  on both sides of the inner product. Note that  $\sum_{i=1}^{D}\mathbf{q}'[i]$  depends only on the query vector. Thus, its computation can be conducted once and shared by many data vectors. For the computation of  $\langle \mathbf{q}', \bar{\mathbf{x}}_b \rangle$ ,

<span id="page-2-2"></span><sup>&</sup>lt;sup>2</sup>Without further specification, in this paper, by data and query vectors, we refer to their normalized vectors.

[27] introduces two versions of implementation. One is based on a SIMD-based implementation called *FastScan* [4], which can efficiently compute the estimated distances for data vectors batch by batch. The other is based on bitwise operations, which supports to efficiently estimate distances for individual vectors. We refer readers to the original papers of RaBitQ [27] and FastScan [3–5] for more technical details and theoretical analysis. With all the proposed techniques, RaBitQ supports to unbiasedly estimate the inner product (and further unbiasedly estimate the squared distances) with both promising accuracy and efficiency.

