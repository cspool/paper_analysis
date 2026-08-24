# <span id="page-11-0"></span>A Combining SwiGLU Feed-Forward Layers

The SwiGLU variation of the Transformer FF layer is used across numerous current language models, including OLMo (Groeneveld et al., 2024). The SwiGLU FFN is:

$$FF_{SwiGLU} = W^{down}(Swish_1(W^{up}x) \otimes V^{gate}x)$$
(7)

where  $\otimes$  is the component-wise product. We exclude biases here for simplicty and lack of inclusion in OLMo. For more details, we refer the reader to Shazeer (2020). In applying our method to SwiGLU FFs, we note three things: 1) the location of our feature collection is the output of the component-wise product in order to find the best permutation for both  $W^{\rm up}$  and  $V^{\rm gate}$ , and 2) the new merged parameters are computed as:

$$W^{\text{up*}} = \frac{1}{k} \left( W_0^{\text{in}} + \sum_{i=1}^{k-1} P_i W_i^{\text{in}} \right)$$
 (8)

$$V^{\text{gate}*} = \frac{1}{k} \left( V_0^{\text{gate}} + \sum_{i=1}^{k-1} P_i V_i^{\text{gate}} \right) \tag{9}$$

$$W^{\text{down}*} = \frac{1}{k} \left( W_0^{\text{down}} + \sum_{i=1}^{k-1} W_i^{\text{down}} P_i^T \right) \quad (10)$$

#### <span id="page-11-2"></span>**B** Layer Selection Algorithm

We summarize our layer selection + fine-tuning algorithm from Section 3.4 in Algorithm 1.

#### <span id="page-11-3"></span>Algorithm 1 Feed-Forward Sublayer Merge

```
Input: Model parameters \theta_{\rm in}, collected features \{X_i\}_{i=0}^{N_{\rm layers}-1}, batched fine-tuning data D_{\rm ft}
Input constants: k, N_{\rm layers}, MAXUPDATES
Initialize: \theta_{\rm selected}, BESTSCORE \leftarrow 0
for i=0 to (N_{\rm layers}-1)-k do
\theta_{\rm merged} \leftarrow {\rm CoMPRESS}(\theta_{\rm in},\{X_i\}_{i=0}^{N_{\rm layers}-1},k)\nif EVAL(\theta_{\rm merged}) > BESTSCORE then
\theta_{\rm selected} \leftarrow \theta_{\rm merged}\nend if\nend for
for i=0 to MAXUPDATES do
\theta_{\rm selected} \leftarrow {\rm UPDATE}(\theta_{\rm selected},D_{\rm ft}(i))\nend for
```

Output:  $\theta_{\text{selected}}$ 

