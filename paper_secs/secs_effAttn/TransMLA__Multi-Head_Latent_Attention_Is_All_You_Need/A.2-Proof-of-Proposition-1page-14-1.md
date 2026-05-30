# A.2 Proof of Proposition [1](#page-14-1)

Let D be the hidden dimension of the input token x<sup>t</sup> ∈ R <sup>D</sup>, h be the number of query heads, and d = D/h be the dimension per head. In GQA, query heads are divided into g groups. For fair KV cache comparison, the latent dimension for keys and values in MLAFactorized (rkv) and the head dimension of MQA will be related to gd. Specifically, if the KV cache per token in GQA is 2gd for both keys and values, then in MLAFactorized, rkv = 2gd, and in MQA, the head dimension is also 2gd; this ensures the KV cache sizes are aligned.

#### A.2.1 $GQA \leq MLA_{Factorized}$

In GQA, query head  $\mathbf{q}_{t,i}$  attends to key  $\mathbf{k}_{j,\lceil i/(h/g)\rceil}$  and value  $\mathbf{v}_{j,\lceil i/(h/g)\rceil}$ . The GQA key projection  $W^K \in \mathbb{R}^{gd \times D}$  produces g distinct key vectors  $[\mathbf{k}_{t,1};\ldots;\mathbf{k}_{t,g}]$ . Similarly,  $W^V \in \mathbb{R}^{gd \times D}$  produces value vectors. We define effective per-query-head projection matrices  $W'^K \in \mathbb{R}^{hd \times D}$  and  $W'^V \in \mathbb{R}^{hd \times D}$  for GQA:

$$W^{\prime K} = \begin{pmatrix} W_1^{\prime K} \\ \vdots \\ W_h^{\prime K} \end{pmatrix}, \text{ where } W_i^{\prime K} = W_{\lceil i/(h/g) \rceil}^K, \tag{21}$$

$$W^{\prime V} = \begin{pmatrix} W_1^{\prime V} \\ \vdots \\ W_h^{\prime V} \end{pmatrix}, \text{ where } W_i^{\prime V} = W_{\lceil i/(h/g) \rceil}^{V}.$$
 (22)

Here,  $W_k^K$  is the k-th  $d \times D$  block of  $W^K$ . Thus,  $\mathbf{k}'_{j,i} = W_i^{\prime K} \mathbf{x}_j = \mathbf{k}_{j,\lceil i/(h/g)\rceil}$ , and similarly for values. The matrices  $W^{\prime K}$  and  $W^{\prime V}$  have ranks at most gd.

An  $\mathrm{MLA}_{\mathrm{Factorized}}$  mechanism generates keys via  $\mathbf{k}_{j,i} = (W^{UK}(W^{DKV}\mathbf{x}_j))_i$ , where  $W^{DKV} \in \mathbb{R}^{hd \times r_{kv}}$  and  $W^{UK} \in \mathbb{R}^{hd \times r_{kv}}$ . A similar formulation applies for values with  $W^{UV} \in \mathbb{R}^{hd \times r_{kv}}$ .

To demonstrate expressive capability, GQA  $\leq$  MLA<sub>Factorized</sub>, we set  $r_{kv}=2gd$ . Let  $W^{DKV}=\begin{pmatrix} W^K \\ W^V \end{pmatrix} \in \mathbb{R}^{2gd \times D}$ . We seek  $W^{UK}, W^{UV} \in \mathbb{R}^{hd \times 2gd}$  such that  $W'^K=W^{UK}W^{DKV}, W'^V=W^{UV}W^{DKV}$ . This is achieved by setting  $W_i^{UK}, W_i^{UV} \in \mathbb{R}^{d \times 2gd}$  (the block for head i) as selector matrices:

$$W_i^{UK} = [\underbrace{\mathbf{0}_{d \times d}, \dots, \mathbf{0}_{d \times d}}_{k-1 \text{ blocks}}, \mathbf{I}_{d \times d}, \underbrace{\mathbf{0}_{d \times d}, \dots, \mathbf{0}_{d \times d}}_{2g-k \text{ blocks}}], \tag{23}$$

$$W_i^{UV} = [\underbrace{\mathbf{0}_{d \times d}, \dots, \mathbf{0}_{d \times d}}_{g+k-1 \text{ blocks}}, \mathbf{I}_{d \times d}, \underbrace{\mathbf{0}_{d \times d}, \dots, \mathbf{0}_{d \times d}}_{g-k \text{ blocks}}], \tag{24}$$

where  $k = \lceil i/(h/g) \rceil$ . Thus, GQA's key/value generation can be replicated by an  $\operatorname{MLA}_{\operatorname{Factorized}}$  model with  $r_{kv} = 2gd$  and specific sparse structures for  $W^{UK}$  and  $W^{UV}$ . The KV cache size  $2gd \times (\text{sequence length})$  is preserved since we will be caching  $\mathbf{c}_j^{KV} = W^{DKV}\mathbf{x}_j \in \mathbb{R}^{2gd}$ . On that account, the theoretical expressive power of GQA is less than or equal to that of  $\operatorname{MLA}_{\operatorname{Factorized}}$  given the same KV cache size.

