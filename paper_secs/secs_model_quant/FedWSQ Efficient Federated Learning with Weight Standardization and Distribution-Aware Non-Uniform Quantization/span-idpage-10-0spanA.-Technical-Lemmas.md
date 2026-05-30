# <span id="page-10-0"></span>A. Technical Lemmas

<span id="page-10-1"></span>This section introduces some technical lemmas that are useful to understand our main document.

Lemma 1. *Consider any vector* v ∈ R d *. The mean subtraction of* v *is given by*

$$\bar{\mathbf{v}} = \mathbf{v} - \left(\frac{1}{d}\mathbf{1}^T\mathbf{v}\right)\mathbf{1}$$

$$= \left(\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T\right)\mathbf{v}$$

$$= (\mathbf{I} - \mathbf{P_1})\mathbf{v}$$
(11)

*where* I ∈ R d×d *is the identity matrix,* 1 ∈ R d *is a vector whose elements are all ones, and* P<sup>w</sup> *represents the projection matrix onto the vector* w*. Thus, mean subtraction is equivalent to projecting* v *onto* span{1} <sup>⊥</sup>*. In other words, this projection removes the DC (constant) component from the given vector* v*.*

Lemma 2. *Consider any vector* v¯ ∈ R <sup>d</sup> *with zero mean. Normalization of* v¯ *using its standard deviation* σ(v¯) *is given by*

$$\tilde{\mathbf{v}} = \frac{\rho}{\sigma(\bar{\mathbf{v}})} \bar{\mathbf{v}}$$

$$= \frac{\rho \sqrt{d}}{\|\bar{\mathbf{v}}\|} \bar{\mathbf{v}}.$$
(12)

*Since* v¯ *is zero-centered, its standard deviation is given by* σ(v¯) = p (v¯ <sup>T</sup> v¯)/d*.*

Lemma 3. *Consider any vector* v ∈ R d *. Let* v¯ *and* v˜ *be its mean-subtracted and standardized versions, respectively. The derivative of* v˜ *with respect to* v¯ *is then given by*

$$\frac{\partial \tilde{\mathbf{v}}}{\partial \bar{\mathbf{v}}} = \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \frac{1}{d(\sigma(\bar{\mathbf{v}}))^2} \bar{\mathbf{v}} \bar{\mathbf{v}}^T \right) 
= \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \frac{1}{\|\bar{\mathbf{v}}\|^2} \bar{\mathbf{v}} \bar{\mathbf{v}}^T \right) 
= \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \mathbf{P}_{\bar{\mathbf{v}}} \right).$$
(13)

*Also, based on Lemma [1,](#page-10-1) the derivative of* v¯ *with respect to* v *is given by*

$$\frac{\partial \bar{\mathbf{v}}}{\partial \mathbf{v}} = (\mathbf{I} - \mathbf{P_1}). \tag{14}$$

*Since* σ(v¯) = σ(v)*, by the chain rule, we can derive the gradient of a loss function* L *with respect to* v *as follows:*

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \mathbf{P_1} \right) \left( \mathbf{I} - \mathbf{P_{\bar{\mathbf{v}}}} \right) \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{v}}}.$$
 (15)

