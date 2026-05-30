# <span id="page-15-0"></span>B.3 The MLP Form for Mixtral and DeepSeekMoE

The output of an MLP in Mixtral and DeepseekMoE can be rewritten into:

$$\begin{split} E_k(\mathbf{x}) &= \mathbf{W}_k^{(2)} \cdot \operatorname{SwiGLU}(\mathbf{x}) + \mathbf{b}_k^{(2)} \\ &= \mathbf{W}_k^{(2)} \cdot \left[ \sigma \left( \mathbf{W}_k^{(1)} \cdot \mathbf{x} + \mathbf{b}_k^{(1)} \right) \odot \left( \mathbf{W}_k^{(3)} \cdot \mathbf{x} + \mathbf{b}_k^{(3)} \right) \right] + \mathbf{b}_k^{(2)} \\ &= \sum_{i=1}^{p_I} \mathbf{W}_{k,\cdot,i}^{(2)} \cdot \left[ \sigma \left( \left\langle \mathbf{W}_{k,i,\cdot}^{(1)}, \mathbf{x} \right\rangle + \mathbf{b}_{k,i}^{(1)} \right) \cdot \left( \left\langle \mathbf{W}_{k,i,\cdot}^{(3)}, \mathbf{x} \right\rangle + \mathbf{b}_{k,i}^{(3)} \right) \right] + \mathbf{b}_k^{(2)}, \end{split}$$

where  $\sigma(\mathbf{x}) = \text{Swish}_{\beta}(\mathbf{x}) = \mathbf{x}\sigma(\beta\mathbf{x}) = \frac{\mathbf{x}}{1 + e^{-\beta\mathbf{x}}}$ , with  $\beta$  setting to

Similarly, for Mixtral and DeepSeekMoE, the extraction objective is:

$$\begin{split} \min_{\substack{\mathbf{W}_{\omega}^{(1)}, \mathbf{b}_{\omega}^{(1)}, \mathbf{W}_{\omega}^{(3)}, \mathbf{b}_{\omega}^{(3)}, \mathbf{W}_{\omega}^{(2)}}} \frac{1}{N} \sum_{k=1}^{N} \left[ \left\| \mathbf{T}_{k} \left( \mathbf{W}_{k}^{(1)}, \mathbf{b}_{k}^{(1)}, \mathbf{W}_{k}^{(3)}, \mathbf{b}_{k}^{(3)} \right) \right. \\ \left. - \left( \mathbf{W}_{\omega}^{(1)}, \mathbf{b}_{\omega}^{(1)}, \mathbf{W}_{\omega}^{(3)}, \mathbf{b}_{\omega}^{(3)} \right) \right\|_{F}^{2} \\ \left. + \left\| \mathbf{W}_{k}^{(2)} \mathbf{T}_{k}^{\mathsf{T}} - \mathbf{W}_{\omega}^{(2)} \right\|_{F}^{2} \right]. \end{split}$$

Then we can extend the  $\mathbf{W}_k$  to:

$$\mathbf{W}_{k} = \left[\mathbf{W}_{k}^{(1)}, \mathbf{b}_{k}^{(1)}, \mathbf{W}_{k}^{(3)}, \mathbf{b}_{k}^{(3)}, (\mathbf{W}_{k}^{(2)})^{\mathrm{T}}\right],$$

and ResMoE can then be similary applied to Mixtral and DeepSeek-MoE.

#### <span id="page-15-1"></span>C Proof of Proposition 4.1

For the reader's convenience, we recall Theorem 4.1 as follows.

**Proposition C.1.** Consider the solution  $\mathbf{W}_{\omega}$  to the following free-support WB problem

$$\min_{\mathbf{W}_{\omega}} \frac{1}{N} \sum_{k=1}^{N} W_2^2(\mu_k, \mu_{\omega}(\mathbf{W}_{\omega})). \tag{5}$$

Then  $\mathbf{W}_{\omega}$ , along with  $\mathbf{T}_k = p_{\mathbf{I}} \cdot \mathrm{OT}(\mu_k, \mu_{\omega}(\mathbf{W}_{\omega}))$ , is the solution to the optimization problem (4).

PROOF. We recall  $\mu_i$ ,  $\mu_\omega$  are uniformly distributed over the  $p_I$  rows of  $\mathbf{W}_i$  and  $\mathbf{W}_\omega$ , respectively. OT  $(\mu_i, \mu_\omega(\mathbf{W}_\omega))$  is the optimal transport matrix  $\mathbf{M}$  from  $\mu_i$  to  $\mu_\omega$  of the following problem:

<span id="page-15-2"></span>
$$OT(\mu, \nu) := \underset{\mathbf{M} \in U(\alpha, \beta)}{\operatorname{argmin}} \langle \mathbf{M}, \mathbf{C} \rangle, \tag{6}$$

where 
$$\mathbf{C} = \left[ \| \mathbf{W}_{k_i} - \mathbf{W}_{\omega_j} \|^2 \right]_{ij} \in \mathbb{R}^{p_I \times p_I}, U(\frac{\mathbf{1}_{p_I}}{p_I}, \frac{\mathbf{1}_{p_I}}{p_I}) := \{ \mathbf{M} \in \mathbb{R}^{p_I \times p_I}_+ \mid \mathbf{M} \mathbf{1}_{p_I} = \frac{\mathbf{1}_{p_I}}{p_I}, \mathbf{M}^T \mathbf{1}_{p_I} = \frac{\mathbf{1}_{p_I}}{p_I} \}.$$

We first relate the transport matrix to the permutation matrices  $T_k$ 's. Peyre and Cuturi [47, Proposition 2.1] shows the optimal solution to problem (6) is exactly a permutation matrix, up to a constant factor  $p_I$ . Now straightforwardly, problem (4) can be rewritten as:

<span id="page-15-3"></span>
$$\min_{\substack{\mathbf{W}_{\omega} \\ \mathbf{T}_{k} \in P, k \in [N]}} \frac{1}{N} \sum_{k=1}^{N} \left[ \| \mathbf{T}_{k} \mathbf{W}_{k} - \mathbf{W}_{\omega} \|_{F}^{2} \right], \tag{7}$$

where 
$$\mathbf{W}_k = [\mathbf{W}_k^{(1)}, \mathbf{b}_k^{(1)}, (\mathbf{W}_k^{(2)})^T] \in \mathbb{R}^{p_I \times (2p+1)}$$
, and  $\mathbf{W}_{\omega} = [\mathbf{W}_{\omega}^{(1)}, \mathbf{b}_{\omega}^{(1)}, (\mathbf{W}_{\omega}^{(2)})^T] \in \mathbb{R}^{p_I \times (2p+1)}$ .

We denote the objective function in problem (7) as  $f(\mathbf{W}_{\omega}; \{\mathbf{T}_k\}_{k=1}^N) = \sum_{k=1}^N \frac{1}{N} [\|\mathbf{T}_k \mathbf{W}_k - \mathbf{W}_{\omega}\|_F^2]$ , and take  $\mathbf{W}_{\omega}^*$  as the optimal solution to the Wasserstein barycenter problem (5). For the given  $\mathbf{W}_{\omega}^*$ , we further denote  $\mathbf{T}_k^* := \underset{T}{\operatorname{argmin}} f(\mathbf{W}_w^*; \mathbf{T}_k), \forall k \in [N]$ . The rest of the

proof is to show 
$$f(\mathbf{W}_{\omega}^*; \left\{\mathbf{T}_k^*\right\}_{k=1}^N) = \text{Equation (4)}.$$

① We start with the first side: Equation (4)  $\leq f(\mathbf{W}_{\omega}^*; \left\{\mathbf{T}_k^*\right\}_{k=1}^N$ ). We indeed immediately have:

$$(4) = \min_{\mathbf{W}_{\omega}, \mathbf{T}_{k}} f(\mathbf{W}_{\omega}; \{\mathbf{T}_{k}\}_{k=1}^{N}) \le f(\mathbf{W}_{\omega}^{*}; \{\mathbf{T}_{k}^{*}\}_{k=1}^{N}),$$

due to the definition of min in Equation (4).

② For the other direction, we first show the barycenter loss  $(5) \le (4)$ . Through the definition of  $W_2$  distance, we have

$$\begin{split} & W_2^2(\mu_i, \mu_\omega(\mathbf{W}_\omega)) \leq \|\mathbf{T}_k \mathbf{W}_k - \mathbf{W}_\omega\|_F^2, \ \forall \mathbf{T}_k, \mathbf{W}_\omega \\ \Rightarrow & \frac{1}{N} \sum_{k=1}^N W_2^2(\mu_i, \mu_\omega(\mathbf{W}_\omega)) \leq \frac{1}{N} \sum_{k=1}^N \|\mathbf{T}_k \mathbf{W}_k - \mathbf{W}_\omega\|_F^2, \ \forall \mathbf{T}_k, \mathbf{W}_\omega \\ \Rightarrow & \frac{1}{N} \sum_{k=1}^N W_2^2(\mu_i, \mu_\omega(\mathbf{W}_\omega)) \leq \frac{1}{N} \sum_{k=1}^N \min_{\mathbf{T}_k} \|\mathbf{T}_k \mathbf{W}_k - \mathbf{W}_\omega\|_F^2, \ \forall \mathbf{W}_\omega. \end{split}$$

The inequality will still hold when we minimize the two sides both over  $\mathbf{W}_{\omega}$ :

$$(5) = \min_{\mathbf{W}_{\omega}} \frac{1}{N} \sum_{k=1}^{N} W_{2}^{2}(\mu_{i}, \mu_{\omega}(\mathbf{W}_{\omega}))$$

$$\leq \min_{\mathbf{W}_{\omega}} \frac{1}{N} \sum_{k=1}^{N} \min_{\mathbf{T}_{k}} \|\mathbf{T}_{k}\mathbf{W}_{k} - \mathbf{W}_{\omega}\|_{F}^{2}$$

$$= (4).$$

<span id="page-16-0"></span>To close the proof, it suffices to show that  $f(\mathbf{W}_{\omega}^*; \mathbf{T}_k^*) =$  (5). We show the equivalence as follows:

$$\begin{split} f(\mathbf{W}_{\omega}^{*}; \mathbf{T}_{k}^{*}) &= \frac{1}{N} \sum_{k=1}^{N} \left\| \mathbf{T}_{k}^{*} \mathbf{W}_{k} - \mathbf{W}_{\omega}^{*} \right\|_{F}^{2} \\ &= \frac{1}{N} \sum_{k=1}^{N} \min_{\mathbf{T}_{k}} [\left\| \mathbf{T}_{k} \mathbf{W}_{k} - \mathbf{W}_{\omega}^{*} \right\|_{F}^{2}] \\ &= \frac{1}{N} \sum_{k=1}^{N} W_{2}^{2} (\mu_{i}, \mu_{\omega}(\mathbf{W}_{\omega}^{*})), \end{split}$$

where the last equation holds again thanks to Peyre and Cuturi [47, Proposition 2.1]. Using the fact that  $W_\omega^*$  is the optimal solution to Wasserstein barycenter problem (5), we finally attain

$$f(\mathbf{W}_{\omega}^{*}; \mathbf{T}_{k}^{*}) = \frac{1}{N} \sum_{k=1}^{N} W_{2}^{2}(\mu_{i}, \mu_{\omega}(\mathbf{W}_{\omega}^{*}))$$

$$= \min_{\mathbf{W}_{\omega}} \frac{1}{N} \sum_{i=1}^{N} W_{2}^{2}(\mu_{i}, \mu_{\omega}(\mathbf{W}_{\omega}))$$

$$= (5),$$

which completes the proof.

 $\Diamond$