# <span id="page-15-1"></span><span id="page-15-0"></span>A PROOF OF THEOREM 3.4.

*Proof of Theorem 3.4.* 

For  $\tilde{\mathbf{A}}_{s,a}^{(t)}$ , we have:

$$(\mathbf{A}_{\text{full}}^{(t)} - \mathbf{A}_{\text{s,q}}^{(t)}) - (\mathbf{A}_{\text{full}}^{(t_{\text{ref}})} - \mathbf{A}_{\text{s,q}}^{(t_{\text{ref}})}) = \hat{\Delta}_{\text{quant}}^{t}$$

$$\Rightarrow \mathbf{A}_{\text{full}}^{(t)} = \mathbf{A}_{\text{s,q}}^{(t)} + (\mathbf{A}_{\text{full}}^{(t_{\text{ref}})} - \mathbf{A}_{\text{s,q}}^{(t_{\text{ref}})}) + \hat{\Delta}_{\text{quant}}^{t}$$

$$= \mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t_{\text{ref}})} + \hat{\Delta}_{\text{quant}}^{t}.$$
(17)

Given this, we further have:

$$\mathbf{A}_{\text{full}}^{(t)} - \tilde{\mathbf{A}}_{\text{s,q}}^{(t)} = (\mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t_{\text{ref}})} + \hat{\Delta}_{\text{quant}}^{(t)}) - \tilde{\mathbf{A}}_{\text{s,q}}^{(t)}$$

$$= (\mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t_{\text{ref}})} + \hat{\Delta}_{\text{quant}}^{(t)}) - (\mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t_{\text{ref}})} + \hat{\Delta}_{\text{quant}}^{(t_{\text{ref}})})$$

$$= \hat{\Delta}_{\text{quant}}^{(t)} - \hat{\Delta}_{\text{quant}}^{(t_{\text{ref}})}.$$
(18)

Similarly, for  $\hat{\mathbf{A}}_{s,q}^{(t)}$ , we also have:

$$\mathbf{A}_{\text{full}}^{(t)} - \hat{\mathbf{A}}_{\text{s,q}}^{(t)} = (\mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t)}) - \hat{\mathbf{A}}_{\text{s,q}}^{(t)} = (\mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t)}) - (\mathbf{A}_{\text{s,q}}^{(t)} + \Delta_{\text{quant}}^{(t_{\text{ref}})}) = \Delta_{\text{quant}}^{(t)} - \Delta_{\text{quant}}^{(t_{\text{ref}})}.$$
(19)

Based on Proposition 3.3, we have:

$$\mathbb{E}_{t} \underbrace{\left[ \left\| \mathbf{A}_{\text{full}}^{(t)} - \tilde{\mathbf{A}}_{\text{s,q}}^{(t)} \right\|_{F} \right]}_{\text{second-order}} = \mathbb{E}_{t} \left[ \left\| \hat{\Delta}_{\text{quant}}^{(t)} - \hat{\Delta}_{\text{quant}}^{(t_{\text{ref}})} \right\|_{F} \right] \leq \mathbb{E}_{t} \left[ \left\| \Delta_{\text{quant}}^{(t)} - \Delta_{\text{quant}}^{(t_{\text{ref}})} \right\|_{F} \right] = \mathbb{E}_{t} \underbrace{\left[ \left\| \mathbf{A}_{\text{full}}^{(t)} - \hat{\mathbf{A}}_{\text{s,q}}^{(t)} \right\|_{F} \right]}_{\text{first-order}}.$$

Therefore, Theorem 3.4 holds.

