# B Further theoretical results to support model folding

Lemma B.1. Let x ∈ R <sup>k</sup> and let U ∈ {0, 1} n×k be a binary clustering matrix with P j uij = 1. Then with any element-wise nonlinear function σ(·) we have

$$\sigma(\mathbf{U}\mathbf{x}) = \mathbf{U}\sigma(\mathbf{x})$$

Proof of Lemma B.1. Define y = Ux, z = σ(Ux) and v = σ(x), w = Uσ(x). Note that in any row of U just one element satisfies uij = 1. We define such an element by a function p with uij = 1 ⇔ p(i) = j.

Therefore, y<sup>i</sup> = xp(i) and z<sup>i</sup> = σ(yi) = σ(xp(i)) for all 1 ≤ i ≤ n. Moreover, v<sup>i</sup> = σ(xi) and w<sup>i</sup> = vp(i) = σ(xp(i)). Therefore, z<sup>i</sup> = w<sup>i</sup> and z = w.

<sup>4</sup>https://pytorch.org/hub/

<sup>5</sup>https://huggingface.co/docs/hub/index

<sup>6</sup>https://wandb.ai

<span id="page-19-0"></span>**Lemma B.2.** Let  $\mathbf{x} \in \mathbb{R}^k$ , let  $\mathbf{U} \in \{0,1\}^{n \times k}$  be a binary clustering matrix with  $\sum_j u_{ij} = 1$ , let  $\sigma(\cdot)$  be an element-wise nonlinear function, and define  $\mathbf{C} = \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T$ . Then

$$\sigma(\mathbf{C}\mathbf{x}) = \mathbf{C}^T \sigma(\mathbf{C}\mathbf{x})$$

Proof of Lemma B.2. We can write

$$\begin{split} \sigma(\mathbf{C}\mathbf{x}) &= \sigma(\mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{x}) \\ &= \mathbf{U}\sigma((\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{x}) \qquad \text{(Lemma B.1)} \\ &= \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}(\mathbf{U}^T\mathbf{U})\sigma((\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{x}) \\ &= \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\sigma(\mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{x}) \qquad \text{(Lemma B.1)} \\ &= \mathbf{C}^T\sigma(\mathbf{C}\mathbf{x}). \end{split}$$

**Lemma B.3.** Let  $\mathbf{U}^T$  be a clustering matrix and let  $\mathbf{D}$  be a diagonal matrix, then the following is true

$$(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{D}\mathbf{U} = Diag((\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^Tdiag(\mathbf{D}))$$

Proof of Theorem B.3. The clustering matrix  $\mathbf{U}^T$  can be expressed as:

$$\mathbf{U}^{T} = \begin{bmatrix} \mathbf{u}_{1}^{T} \\ \mathbf{u}_{2}^{T} \\ \vdots \\ \mathbf{u}_{k}^{T} \end{bmatrix} = \begin{bmatrix} u_{11} & u_{12} & \dots & u_{1n} \\ u_{21} & u_{22} & \dots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ u_{k1} & u_{k2} & \dots & u_{kn} \end{bmatrix},$$

where  $\mathbf{u}_i^T$  represents the rows of the clustering matrix. Each row corresponds to cluster i, and the entries  $u_{ij}$  satisfy the binary clustering property:  $u_{ij} = 1$  if the j-th data point belongs to cluster i, and  $u_{ij} = 0$  otherwise.

The product DU is given by:

$$\mathbf{DU} = \begin{bmatrix} d_1 & 0 & \dots & 0 \\ 0 & d_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & d_n \end{bmatrix} \begin{bmatrix} u_{11} & u_{12} & \dots & u_{1k} \\ u_{21} & u_{22} & \dots & u_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ u_{n1} & u_{n2} & \dots & u_{nk} \end{bmatrix}.$$

This simplifies to:

$$\mathbf{DU} = \begin{bmatrix} d_1 u_{11} & d_1 u_{12} & \dots & d_1 u_{1k} \\ d_2 u_{21} & d_2 u_{22} & \dots & d_2 u_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ d_n u_{n1} & d_n u_{n2} & \dots & d_n u_{nk} \end{bmatrix}.$$

Using the clustering property of **U**, it follows that:

$$u_{ij}u_{i'j} = \begin{cases} 1, & \text{if } i = i', \\ 0, & \text{otherwise.} \end{cases}$$

From this, the product  $\mathbf{U}^T \mathbf{D} \mathbf{U}$  simplifies to:

$$\mathbf{U}^T \mathbf{D} \mathbf{U} = \mathrm{Diag}(\mathbf{U}^T \mathrm{diag}(\mathbf{D})).$$

This result holds because only the diagonal entries remain due to the clustering matrix's orthogonality and binary properties.

<span id="page-20-0"></span>Finally, using the above result, we compute:

$$(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{D}\mathbf{U} = (\mathbf{U}^T\mathbf{U})^{-1}\mathrm{Diag}(\mathbf{U}^T\mathrm{diag}(\mathbf{D})).$$

By the property diag(Diag( $\mathbf{x}$ )) =  $\mathbf{x}$  for any  $\mathbf{x} \in \mathbb{R}^n$ , we obtain:

$$(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{D}\mathbf{U} = \mathrm{Diag}((\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathrm{diag}(\mathbf{D})).$$

The lemma demonstrates that projecting the diagonal matrix  $\mathbf{D}$  through the clustering matrix  $\mathbf{U}^T$  preserves its diagonal structure. The diagonal entries are determined by the clustering matrix's mapping of the original diagonal values diag( $\mathbf{D}$ ), ensuring efficient computation and alignment with clustering properties.

**Lemma B.4.** Let  $\mathbf{U}^T$  be a clustering matrix and let  $\mathbf{w} \in \mathbb{R}^n$  and  $\mathbf{x} \in \mathbb{R}^n$ , then the following is true

$$UDiag(\mathbf{w})\mathbf{x} = Diag(\mathbf{U}\mathbf{w})\mathbf{U}\mathbf{x}$$

Proof of Lemma B.4. The clustering matrix U can be expressed as:

$$\mathbf{U} = \begin{bmatrix} \mathbf{v}_1^T \ \mathbf{v}_2^T \ \vdots \ \mathbf{v}_n^T \end{bmatrix},$$

where each row  $\mathbf{v}_m^T$  is defined by a mapping function  $f:\{1,2,\ldots,n\}\to\{1,2,\ldots,k\}$ . For each row  $\mathbf{v}_m^T$ , the entries are defined as:

 $v_{m,j} = \begin{cases} 1, & \text{if } j = f(m), \\ 0, & \text{otherwise.} \end{cases}$ 

This representation indicates that the clustering matrix **U** assigns each element m to a specific cluster f(m). Each row  $\mathbf{v}_m^T$  has a single non-zero element corresponding to the cluster index f(m).

Calculation of the Left-Hand Side (LHS). The left-hand side of the equality is:

$$UDiag(\mathbf{w})\mathbf{x}$$
.

First, compute  $Diag(\mathbf{w})\mathbf{x}$ , which scales each element of  $\mathbf{x}$  by the corresponding element of  $\mathbf{w}$ :

$$\operatorname{Diag}(\mathbf{w})\mathbf{x} = \begin{bmatrix} w_1 x_1 \\ w_2 x_2 \\ \vdots \\ w_n x_n \end{bmatrix}.$$

Then, multiplying by **U** aggregates these scaled values according to the clusters defined by f. Specifically, the j-th element of  $\mathbf{U}\mathrm{Diag}(\mathbf{w})\mathbf{x}$  is given by:

$$(\mathbf{U}\mathrm{Diag}(\mathbf{w})\mathbf{x})_j = \sum_{m: f(m)=j} w_m x_m.$$

Calculation of the Right-Hand Side (RHS). The right-hand side of the equality is:

$$Diag(Uw)Ux$$
.

First, compute  $\mathbf{U}\mathbf{w}$ . The j-th element of  $\mathbf{U}\mathbf{w}$  is:

$$(\mathbf{U}\mathbf{w})_j = \sum_{m:f(m)=j} w_m,$$

<span id="page-21-0"></span>which sums the  $w_m$  values for all elements assigned to cluster j.

Next, construct  $Diag(\mathbf{U}\mathbf{w})$ , a diagonal matrix with entries  $(\mathbf{U}\mathbf{w})_i$  along the diagonal:

$$Diag(\mathbf{U}\mathbf{w}) = \begin{bmatrix} (\mathbf{U}\mathbf{w})_1 & 0 & \dots & 0 \\ 0 & (\mathbf{U}\mathbf{w})_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & (\mathbf{U}\mathbf{w})_k \end{bmatrix}.$$

Finally, compute  $\mathbf{U}\mathbf{x}$ . The j-th element of  $\mathbf{U}\mathbf{x}$  is:

$$(\mathbf{U}\mathbf{x})_j = \sum_{m:f(m)=j} x_m,$$

which sums the  $x_m$  values for all elements assigned to cluster j. Multiplying Diag( $\mathbf{U}\mathbf{w}$ ) by  $\mathbf{U}\mathbf{x}$  gives:

$$(\operatorname{Diag}(\mathbf{U}\mathbf{w})\mathbf{U}\mathbf{x})_j = (\mathbf{U}\mathbf{w})_j(\mathbf{U}\mathbf{x})_j = \left(\sum_{m:f(m)=j} w_m\right) \left(\sum_{m:f(m)=j} x_m\right).$$

**Verification of Equality.** Both the LHS and RHS compute the same aggregated sums  $\sum_{m:f(m)=j} w_m x_m$  for each cluster j. The LHS directly performs the aggregation of  $w_m x_m$  within clusters, while the RHS separates the computation into two steps: summing  $w_m$  and  $x_m$  for each cluster, followed by multiplying these sums. Since multiplication distributes over addition, the two expressions are equivalent:

$$UDiag(\mathbf{w})\mathbf{x} = Diag(\mathbf{U}\mathbf{w})\mathbf{U}\mathbf{x}.$$

The lemma is proven, as both sides of the equation compute the same weighted aggregation of  $w_m x_m$  over the clusters defined by the clustering matrix **U**.

**Lemma B.5.** Let  $\mathbf{C}^T$  be a clustering matrix and let  $\mathbf{D}$  be a diagonal matrix, then the following is true

$$\|\mathbf{W} - Diag(\mathbf{C}diag(\mathbf{W}))\|_F^2 = \|diag(\mathbf{W}) - \mathbf{C}diag(\mathbf{W})\|_2^2$$

Proof of Lemma B.5. Let  $\tilde{\mathbf{W}} = \text{Diag}(\mathbf{C}\text{diag}(\mathbf{W}))$ , where  $\tilde{\mathbf{W}}$  represents the diagonal matrix obtained by clustering the diagonal entries of  $\mathbf{W}$  using the clustering matrix  $\mathbf{C}$ . Both  $\mathbf{W}$  and  $\tilde{\mathbf{W}}$  are diagonal matrices, so their difference  $\mathbf{W} - \tilde{\mathbf{W}}$  is also diagonal. The entries of this difference are:

$$w_{i,j} - \tilde{w}_{i,j} = \begin{cases} w_{i,i} - \tilde{w}_{i,i}, & \text{if } i = j, \\ 0, & \text{otherwise.} \end{cases}$$

The Frobenius norm of the difference  $\mathbf{W} - \tilde{\mathbf{W}}$  is:

$$\|\mathbf{W} - \tilde{\mathbf{W}}\|_F^2 = \sum_{i,j} (w_{i,j} - \tilde{w}_{i,j})^2.$$

Since  $\mathbf{W}$  and  $\tilde{\mathbf{W}}$  are diagonal matrices, this simplifies to:

$$\|\mathbf{W} - \tilde{\mathbf{W}}\|_F^2 = \sum_i (w_{i,i} - \tilde{w}_{i,i})^2.$$

The diagonal entries of  $\mathbf{W}$  can be represented as a vector diag( $\mathbf{W}$ ), and the diagonal entries of  $\tilde{\mathbf{W}}$  are given by  $\mathbf{C}$ diag( $\mathbf{W}$ ). Substituting these representations, we have:

$$\|\mathbf{W} - \tilde{\mathbf{W}}\|_F^2 = \sum_i (\operatorname{diag}(\mathbf{W})_i - (\mathbf{C}\operatorname{diag}(\mathbf{W}))_i)^2.$$

This is equivalent to the squared  $\ell_2$ -norm of the difference between the vectors diag(**W**) and Cdiag(**W**), giving:

$$\|\mathbf{W} - \tilde{\mathbf{W}}\|_F^2 = \|\operatorname{diag}(\mathbf{W}) - \mathbf{C}\operatorname{diag}(\mathbf{W})\|_2^2.$$

Substituting back  $\tilde{\mathbf{W}} = \text{Diag}(\mathbf{C}\text{diag}(\mathbf{W}))$ , we conclude that:

$$\|\mathbf{W} - \operatorname{Diag}(\mathbf{C}\operatorname{diag}(\mathbf{W}))\|_F^2 = \|\operatorname{diag}(\mathbf{W}) - \mathbf{C}\operatorname{diag}(\mathbf{W})\|_2^2$$
.

**Lemma B.6.** Let  $\mathbf{A} \in \mathbb{R}^{n \times n}$  and  $\mathbf{B} \in \mathbb{R}^{n \times n}$  be diagonal matrices, then:

$$AB = Diag(Adiag(B))$$

*Proof of Lemma B.6.* Since both **A** and **B** are diagonal matrices, their product **AB** is also a diagonal matrix. The entries of the product **AB** are given by:

$$(\mathbf{AB})_{i,j} = a_{i,j}b_{i,j}.$$

For diagonal matrices, all off-diagonal entries are zero, so:

$$(\mathbf{AB})_{i,j} = \begin{cases} a_{i,i}b_{i,i}, & \text{if } i = j, \\ 0, & \text{otherwise.} \end{cases}$$

Thus, the diagonal entries of **AB** are  $a_{i,i}b_{i,i}$ , and the matrix **AB** is:

$$\mathbf{AB} = \begin{bmatrix} a_1b_1 & 0 & \dots & 0 \\ 0 & a_2b_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & a_nb_n \end{bmatrix},$$

where  $a_i = a_{i,i}$  and  $b_i = b_{i,i}$  represent the diagonal entries of **A** and **B**, respectively.

Now, let  $diag(\mathbf{B})$  denote the vector of diagonal entries of  $\mathbf{B}$ , i.e.,

$$\operatorname{diag}(\mathbf{B}) = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}.$$

The operation Adiag(B) represents the element-wise multiplication of the diagonal entries of A and B:

$$\mathbf{A} \operatorname{diag}(\mathbf{B}) = \begin{bmatrix} a_1 b_1 \\ a_2 b_2 \\ \vdots \\ a_n b_n \end{bmatrix}.$$

Next, using the function  $Diag(\cdot)$ , we can construct a diagonal matrix from this vector:

$$\operatorname{Diag}(\mathbf{A}\operatorname{diag}(\mathbf{B})) = \begin{bmatrix} a_1b_1 & 0 & \dots & 0 \\ 0 & a_2b_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & a_nb_n \end{bmatrix}.$$

Clearly,  $\mathbf{AB}$  and  $\mathrm{Diag}(\mathbf{A}\mathrm{diag}(\mathbf{B}))$  are identical, as they both produce the same diagonal matrix with entries  $a_ib_i$  along the diagonal. Therefore:

$$\mathbf{AB} = \mathrm{Diag}(\mathbf{A}\mathrm{diag}(\mathbf{B})).$$

### <span id="page-23-0"></span>C Channel similarity

Models learned by SGD trend to have correlated patterns or similar parameters in the weight space. Fig. 9 shows 3×3 filter weights in conv1 of a pre-trained ResNet18. These filters across the first 3 input channels and first 16 output channels ordered by the entropy of filter weight. From the plot, most filters of a channel can find at least one another similar filter in other channels, which means filter similarity may lead to structured redundancy.

> **[图片提取文字 (无描述)]:**
> 经决定金 化苯基甲化苯基化苯基 逐醇的 医克斯里尔 电相极电影电影 外
![](_page_23_Figure_2.jpeg)

Figure 9: Similar patterns in weight map of conv1 layer in ResNet18 pre-trained on ImageNet [\(Deng et al.,](#page-14-0) [2009\)](#page-14-0). Each small square represents the weights of a single filter in cool-warm color map, where each color of grid corresponds to a weight value.

To investigate the filter redundancy within a layer, we apply weight matching activation matching from the literature [\(Jordan et al.,](#page-15-0) [2022\)](#page-15-0) to each layer of ResNet18 pretrained on CIFAR10 [\(Krizhevsky et al.,](#page-15-0) [2009a\)](#page-15-0) in Fig. [2](#page-3-0) and on ImageNet [\(Deng et al.,](#page-14-0) [2009\)](#page-14-0) in Fig. 10. We observe two findings: (1) The correlation score distribution varies across layers. The earlier and narrower the lay ers are, the more scattered the correlation coefficients are, and only a few have high correlation coefficients. The wider and later the layers are, the more compact the correlation coefficients are, and most of the matching channels have high correlation coefficients. (2) In the same layer, the distribution of correlation coefficients among matched channels differs across various pre-training datasets. This observation does not fully align with the claim by [Chen et al.](#page-13-0) [\(2023\)](#page-13-0) regarding the downward trend of similarity before a reversal. It appears that this characterization might not consistently hold across different models and pre-trained dataset.

> **[图片提取文字 (无描述)]:**
> layer3.0.conv1 convl layer1.0.conv1 layer1.1.conv1 layer2.0.conv1 layer2.0.conv2 layer2.1.conv1 layer3.0.conv2 layer3.1.conv1 layer4.0.conv1 layer4.0.conv2 layer4.1.conv1 1.0 thannels 1.0 thannels 64 channels 64 channels 128 channels 128 channels 128 channels 256 channels 256 channels 512 channels 512 channels 512 channels 1.0-T.0 -1.0 -1.0 − 1.0 ⊤ 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 value 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 Correlation 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.2-0.2-0.2-0.2-0.2-0.2-0.2-0.2-0.2-0.2-0.2-0.2Frequency
![](_page_23_Figure_5.jpeg)

Figure 10: Layer-wise correlation between matched channels in ResNet18 trained on ImageNet. We compute a layer-wise correlation matrix by matching activations between channels, then assign each channel its best match in the same layer using a greedy pairing based on the correlation matrix.

#### C.1 The impact of regularization

In Fig. [6,](#page-8-0) the models on CIFAR10 were trained without regularization, while the pre-trained ImageNet models were sourced from torchvision. In Fig. [11,](#page-24-0) we extend the comparison of folding and pruning methods on CIFAR10, including ResNet18 (left column) and VGG11 (right column) models trained with explicit L<sup>1</sup> and L<sup>2</sup> regularization. L<sup>1</sup> regularization, in particular, promotes neuron sparsity, leading structured magnitude pruning methods to outperform model folding under these conditions. However, a comparison between Fig. [6](#page-8-0)

<span id="page-24-0"></span>and Fig. 11 shows that model folding with L<sup>2</sup> regularization maintains the highest accuracy at higher sparsity levels, surpassing 80% accuracy. In contrast, the accuracy of the pruned network trained with L<sup>1</sup> drops significantly, reaching just 33% at 75% sparsity.

> **[图片提取文字 (无描述)]:**
> 0.9 0.9 ─ Fold-R ─ Fold-R 0.8 0.8 Fold-Naive Fold-Naive Test Accuracy 6.0 7.0 8.0 8.0 8.0 0.7 SPL1-R SPL1-R Test Accuracy SPL1 SPL1 SPL2-R SPL2-R SPL2 SPL2 0.2 0.2 0.1 0.1 0.0 0.1 0.2 0.6 0.7 0.0 0.1 0.2 0.3 0.3 0.4 0.5 0.4 0.5 0.6 0.7 Sparsity Sparsity 0.9 0.8 0.8 Accuracy 6.0 7.0 8.0 Test Accuracy 70 0.0 Fold-R Fold-R Fold-Naive Fold-Naive 0.4 0.3 SPL1-R SPL1-R - SPL1 SPL1 SPL2-R SPL2-R 0.2 0.2 SPL2 SPL2 0.1 0.1 0.2 0.3 0.5 0.6 0.7 0.2 0.3 0.4 0.5 0.0 0.4 0.0 0.1 0.6 0.7 Sparsity Sparsity
![](_page_24_Figure_1.jpeg)

Figure 11: ResNet18 (left column) and VGG11 (right column) models trained with L<sup>1</sup> (top row) and L<sup>2</sup> (bottom row) regularization. Structured magnitude pruning outperforms model folding only if training explicitly regularizes for model sparsity (L<sup>1</sup> norm). REPAIR is hardly beneficial for all structural pruning methods.

#### C.2 Folding wider models

Do wider networks present more opportunities for model folding? We first examine the layer-wise correlation among matched channels in VGG11 and its wider variants on CIFAR10, as shown in Fig. [8.](#page-11-0) This ablation study reveals that increasing the layer width strengthens the matched correlations, suggesting greater potential for folding. Building on this, Fig. [12](#page-25-0) demonstrates the application of model folding also to 1x/2x/3x wider MLP and ResNet50 architectures, trained on CIFAR10 and CIFAR100, showing consistent performance gains as width increases.

