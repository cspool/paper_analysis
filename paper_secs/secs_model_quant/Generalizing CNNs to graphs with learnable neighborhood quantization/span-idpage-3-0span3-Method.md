# <span id="page-3-0"></span>3 Method

#### 3.1 Preliminaries: Shift quantization

Shift quantization is a quantization scheme which constrains weight values to powers-of-two or zero values. A representable value in a (k + 2)-bit shift quantization is given by:

$$v = s \cdot 2^{e-b},\tag{1}$$

where  $s = \{-1,0,1\}$  denotes either zero or the sign of the value, e is an integer bounded by  $[0,2^k-1]$ , and b is the bias, a layer-wise constant which scales the magnitudes of quantized values. We use  $\hat{\theta} = Q_{n,b}^{\rm shift}[\theta]$  to denote a n-bit shift quantization with a bias b of a weight value  $\theta$  to the nearest representable value  $\hat{\theta}$ . As we have discussed earlier and illustrated in Figure 1, shift quantization on sparse layers makes poor use of the range of representable values, i.e. the resulting distribution after quantization  $q_{n,b}^{\rm shift}(\theta)$  is a poor approximation of the original layer weight distribution  $p(\theta|\mathcal{D})$ , where  $\mathcal{D}$  is the training dataset.

#### 3.2 Designing the Recentralized Quantization Function

Intuitively, it is desirable to concentrate quantization effort on the high probability regions in the weight distribution in sparse layers. By doing so, we can closely match the distribution of quantized weights with the original, and thus at the same time incur smaller round-off errors. Recentralized quantization  $Q[\theta]$  is designed specifically for this purpose, and applied in a layer-wise fashion. Assuming that  $\theta \in \theta$  is a weight value of a convolutional layer, we can define  $Q[\theta]$  as follows:

$$Q[\theta] = z_{\theta} \alpha \sum_{c \in C} \delta_{c,m_{\theta}} Q_c^{\text{rec}}[\theta], \text{ where } Q_c^{\text{rec}}[\theta] = Q_{n,b}^{\text{shift}} \left[ \frac{\theta - \mu_c}{\sigma_c} \right] \sigma_c + \mu_c.$$
 (2)

Here  $z_{\theta}$  is a predetermined constant  $\{0,1\}$  binary value to indicate if  $\theta$  is pruned, and it is used to set pruned weights to 0. The set of components  $c \in C$  determines the locations to focus quantization effort, each specified by the component's mean  $\mu_c$  and standard deviation  $\sigma_c$ . The Kronecker delta  $\delta_{c,m_{\theta}}$  evaluates to either 1 when  $c=m_{\theta}$ , or 0 otherwise. In other words, the constant  $m_{\theta} \in C$  chooses which component in C is used to quantize  $\theta$ . Finally,  $Q_c^{\rm rec}[\theta]$  locally quantizes the component c with shift quantization. Following [27] and [14], we additionally introduce a layer-wise learnable scaling factor  $\alpha$  initialized to 1, which empirically improves the task accuracy.

By adjusting the  $\mu_c$  and  $\sigma_c$  of each component c, and finding suitable assignments of weights to the components, the quantized weight distribution  $q_{\phi}(\theta)$  can thus match the original closely, where we use  $\phi$  as a shorthand to denote the relevant hyperparameters, e.g.  $\mu_c$ ,  $\sigma_c$ . The following section explains how we can optimize them efficiently.

