# D.1 KV Balancing: Purpose and Formulation

**Purpose** The primary goal of KV balancing is to ensure that the principal component analysis (PCA), when applied jointly to the NoPE key and value activations, is not disproportionately influenced by components with larger norms. We observed that the activations derived from  $W_{\text{NoPE}}^{DK}$  (i.e.,  $\mathbf{k_{NoPE}}_{,t} = W_{\text{NoPE}}^{DK}\mathbf{x}_{t}$ ) often have a significantly larger average norm than those from  $W^{DV}$  (i.e.,  $\mathbf{v}_{t} = W^{DV}\mathbf{x}_{t}$ ). Without balancing, PCA would predominantly capture the variance within the NoPE key components, potentially neglecting important variations in the value components.

**Formulation** To address this imbalance, we introduce a scaling factor  $\alpha$ . This factor is computed as the ratio of the expected L2 norms of the NoPE key activations to the value activations, based on a calibration dataset:

$$\alpha = \frac{\mathbb{E}_t[\|W_{\text{NoPE}}^{DK} \mathbf{x}_t\|_2]}{\mathbb{E}_t[\|W^{DV} \mathbf{x}_t\|_2]}$$
(33)

where  $\mathbf{x}_t \in \mathbb{R}^D$  is the *t*-th input token.

While the main paper states scaling  $W^{DK}_{\text{NoPE}}$  by  $1/\alpha$  and  $W^{UK}$  by  $\alpha$  for mathematical equivalence in the model's output, for the purpose of deriving the PCA projection, we effectively use scaled NoPE

key activations. That is, the activations used to compute the PCA basis are  $\mathbf{k}'_{NoPE,t} = 1/\alpha \cdot W_{NoPE}^{DK} \mathbf{x}_t$ and  $\mathbf{v}_t = W^{DV}\mathbf{x}_t$ . This ensures that the PCA process considers features from keys and values on a more equitable footing with respect to their magnitudes. The subsequent low-rank decomposition will then be applied to  $W^{DK}_{\text{NoPE}}$  and  $W^{DV}$ , using the PCA basis derived from these balanced activations.

#### D.2 Joint Low-Rank Approximation of NoPE Keys and Values using PCA

After determining the scaling factor  $\alpha$ , we proceed to compress the projection matrices associated with the NoPE keys  $(W_{\text{NoPE}}^{DK})$  and all values  $(W^{DV})$  jointly.

The process is as follows:

1. Collect Calibrated Activations: A small calibration dataset (WikiText-2) is used. For each input  $\mathbf{x}_t$  from this dataset, we compute the scaled NoPE key activations  $\mathbf{k}'_{\text{NoPE},t}$  and the value activations  $\mathbf{v}_t$ . These are concatenated to form combined activation vectors:

$$\mathbf{c}_{\text{NoPE},t} = \begin{pmatrix} \mathbf{k}'_{\text{NoPE},t} \\ \mathbf{v}_t \end{pmatrix} \in \mathbb{R}^{(2g-1)d}$$
(34)

- 2. **Perform PCA**: PCA is performed on the set of collected combined activation vectors  $\{\mathbf{c}_{\text{NoPE},t}\}$ . This involves computing the covariance matrix of these vectors and finding its principal components. The eigenvectors (corresponding to the largest eigenvalues) are selected to form the columns of a projection matrix  $R_{KV} \in \mathbb{R}^{((2g-1)d) \times r_{kv}}$ , where  $r_{kv}$  is the reduced rank. This matrix  $R_{KV}$  captures the directions of highest variance in the (balanced) combined NoPE key and value activation space.

3. Low-Rank Decomposition of Projection Matrices: Let  $W^{DKV} = \begin{pmatrix} W_{\text{NoPE}}^{DK} \\ W^{DV} \end{pmatrix} \in \mathbb{R}^{((2g-1)d) \times D}$  be the initial projection matrix that transforms the input  $\mathbf{x}_t$  into an intermediate NoPE Key and Value representation  $\mathbf{c}_{\text{NoPE},t} = W^{DKV}\mathbf{x}_t$ . Further, let  $W^{UKV} = \begin{pmatrix} W_{\text{NoPE}}^{UK} & 0 \\ 0 & W^{UV} \end{pmatrix} \in \mathbb{R}^{2hd \times ((2g-1)d)}$ 

represent the subsequent collective projection matrix that takes  $\mathbf{c}_{\text{NoPE},t}$  and processes it to produce the actual keys and values required by the attention mechanism for the NoPE components, where  $W_{\text{RoPE}}^{UK} \in \mathbb{R}^{hd \times gd}$  and  $W_{\text{NoPE}}^{UK} \in \mathbb{R}^{hd \times (g-1)d}$  are two parts of  $W^{UK}$  hat participate in and do not participate in the RoPE computation, respectively. The original sequence of operations for these components can be expressed as  $W^{UKV}W^{DKV}\mathbf{x}_t \in \mathbb{R}^{2hd}$ , in which the first hd elements correspond to the keys and the following hd elements correspond to the values.

To introduce a low-rank bottleneck, we modify both  $W^{DKV}$  and  $W^{UKV}$  using the PCA projection matrix  $R_{KV}$ .

• The initial projection matrix  $W^{DKV}$  is transformed into  $W^{DKV'} \in \mathbb{R}^{r_{kv} \times D}$ :

$$W^{DKV'} = R_{KV}^T W^{DKV} (35)$$

This new matrix  $W^{DKV'}$  takes the original input  $\mathbf{x}_t$  and projects it into a compressed  $r_{kv}$ -dimensional latent space, which is the actual content stored in the KV cache for the NoPE components.

• The subsequent projection matrix  $W^{UKV}$  is transformed into  $W^{UKV} \in \mathbb{R}^{2hd \times r_{kv}}$ :

$$W^{UKV'} = W^{UKV} R_{KV} \tag{36}$$

This new matrix  $W^{UKV'}$  now takes the compressed latent representation as input and produces the final representations for the NoPE components that are used in the attention calculation. As we can see,  $W^{UKV'}$  is actually the concatenated form of  $W^{UK}$  and  $W^{UV}$ in MLA:

$$W^{UKV'} = \begin{pmatrix} W^{UK} \\ W^{UV} \end{pmatrix} \tag{37}$$

This joint decomposition allows for a more holistic compression by identifying shared latent structures between NoPE keys and values, guided by the balanced PCA.

Table 2: Composition of the training dataset.

<span id="page-24-1"></span>

| Dataset           | Sampling Weight |  |  |
|-------------------|-----------------|--|--|
| fineweb-edu-dedup | 0.70            |  |  |
| cosmopedia-v2     | 0.15            |  |  |
| python-edu        | 0.06            |  |  |
| open-web-math     | 0.08            |  |  |
| stackoverflow     | 0.01            |  |  |

