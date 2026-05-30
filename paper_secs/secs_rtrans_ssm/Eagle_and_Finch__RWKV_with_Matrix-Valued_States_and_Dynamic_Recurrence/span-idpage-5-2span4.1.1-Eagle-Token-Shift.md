# <span id="page-5-2"></span>4.1.1 Eagle Token Shift

We adopt the Token Shift technique from the previous RWKV, similar to a 1D causal convolution of size = 2, as can be seen in Figure 1, center-bottom. To better introduce the Token Shift technique, we define some notation. The linear interpolation (lerp) between  $x_t$  and  $x_{t-1}$  used in RWKV-4 and Eagle Token Shift is defined as:

<span id="page-5-4"></span>
$$\operatorname{lerp}_{\square}(a,b) = a + (b-a) \circ \mu_{\square} \tag{3}$$

where each  $\mu_{\square} \in \mathbb{R}^D$  is a learnable vector.

Token Shift allows the model to learn how much new versus old information should be allocated per time step to each channel of receptance, key, value, and gate vectors (r, k, v), and g respectively) independently and uniquely for each head. This makes it possible to form induction heads (Elhage

[et al.,](#page-18-2) [2021\)](#page-18-2) within a single layer since even a single head can directly accumulate both past and current token data into separate subspaces within these vectors.

#### <span id="page-6-0"></span>**4.1.2 Eagle Time Mixing**

The formula of Eagle Time Mixing can be written as follows:

$$\square_t = \operatorname{lerp}_{\square}(x_t, x_{t-1}) \mathbf{W}_{\square}, \quad \square \in \{r, k, \nu, g\}$$
(4)

$$w = \exp(-\exp(\omega)) \tag{5}$$

$$wkv_t = \operatorname{diag}(u) \cdot k_t^{\mathrm{T}} \cdot v_t + \sum_{i=1}^{t-1} \operatorname{diag}(w)^{t-1-i} \cdot k_i^{\mathrm{T}} \cdot v_i \in \mathbb{R}^{(D/h) \times (D/h)}$$
(6)

$$o_t = \operatorname{concat}\left(\operatorname{SiLU}(g_t) \odot \operatorname{LayerNorm}(r_t \cdot wkv_t)\right) W_o \in \mathbb{R}^D$$
 (7)

Where LayerNorm operates on each of *h* heads separately, which is also equivalent to the Group-Norm [\(Wu & He](#page-25-3) [\(2018\)](#page-25-3)) operation on *h* groups. It is also worth noting that *w* is obtained from *w* = exp(−exp(*ω*)), where *ω* ∈ R *D*/*h* are the actual headwise trainable parameters. This ensures that *w* falls within the interval (0,1), guaranteeing that diag(*w*) is a contraction matrix.

The *wkv<sup>t</sup>* attention calculation can alternatively be written in a recurrent form:

$$wkv' = s + \operatorname{diag}(u) \cdot k^{\mathrm{T}} \cdot v \tag{8}$$

<span id="page-6-6"></span><span id="page-6-5"></span><span id="page-6-4"></span>
$$s' = \operatorname{diag}(w) \cdot s + k^{\mathrm{T}} \cdot v \tag{9}$$

RWKV's *wkv* term can be considered a decay-based equivalent to the normalised *k* T *v* term in Linear Attention. It is instructive to note how for a given head *j* the recurrent state *s* is a sum of *k T v* where each channel of *s* individually decays by the corresponding channel of *w* at each time step. Prior to the application of the receptance vector, gating, and output weights, a per-channel learned boost *u* is multiplied with the current token's *k* T *v* and summed with the state, as can be seen in [Figure 1,](#page-5-3) top-right. This gives the current token special treatment relative to the sum of past tokens contained within the decaying state history. The receptance is multiplied by this sum, acting like the query term in Linear Attention.

#### <span id="page-6-1"></span>**4.1.3 Channel Mixing**

In both Eagle and Finch, the Channel Mixing module is identical to the previous RWKV-4 architecture, except for a slightly reduced hidden dimension from 4*D* to 3.5*D*. This reduction accounts for new gating weights in Eagle Time Mixing to ensure an equi-parameter relation with the prior model at the same number of layers and embedding dimension. We do not further reduce the hidden dimension in Finch despite adding a small number of new parameters for LoRA weights. The formulas for Channel Mixing are the same as RWKV-4, but we restate them here to ensure notational consistency, using linear interpolation from Equation [3:](#page-5-4)

$$r'_{t} = \operatorname{lerp}_{r'}(x'_{t}, x'_{t-1}) W_{r'} \in \mathbb{R}^{D}$$
 (10)

$$k'_{t} = \operatorname{lerp}_{k'}(x'_{t}, x'_{t-1}) W_{k'} \in \mathbb{R}^{3.5D}$$
 (11)

$$v_t' = \text{ReLU}(k_t')^2 \mathbf{W}_{v'} \in \mathbb{R}^D$$
(12)

$$o_t' = \sigma(r_t') \circ v_t' \in \mathbb{R}^D \tag{13}$$

