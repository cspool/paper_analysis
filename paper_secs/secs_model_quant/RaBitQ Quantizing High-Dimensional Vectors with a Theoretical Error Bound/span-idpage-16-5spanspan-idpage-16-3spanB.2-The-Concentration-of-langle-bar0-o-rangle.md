# <span id="page-16-5"></span><span id="page-16-3"></span>**B.2** The Concentration of $\langle \bar{0}, o \rangle$

<span id="page-16-6"></span>We next analyze the extent of the concentration of  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$ . Recall that as is shown in (30),  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle = \frac{1}{\sqrt{D}} \|P^{-1}\mathbf{o}\|_{\ell_1}$ . Let  $f(\mathbf{x}) := \frac{1}{\sqrt{D}} \|\mathbf{x}\|_{\ell_1}$ . Then  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle = f(P^{-1}\mathbf{o})$ . We note that  $f(\mathbf{x})$  is a Lipschitz function with the Lipschitz constant of 1, i.e.,

$$|f(\mathbf{x}) - f(\mathbf{y})| \le 1 \cdot ||\mathbf{x} - \mathbf{y}||$$
 (37)

for every x, y on the unit sphere.

PROOF.

$$|f(\mathbf{x}) - f(\mathbf{y})| = \frac{1}{\sqrt{D}} |\|\mathbf{x}\|_{\ell_1} - \|\mathbf{y}\|_{\ell_1}|$$
(38)

$$\leq \frac{1}{\sqrt{D}} \|\mathbf{x} - \mathbf{y}\|_{\ell_1} = \frac{1}{\sqrt{D}} \sum_{i=1}^{D} 1 \cdot |\mathbf{x}[i] - \mathbf{y}[i]|$$
(39)

$$\leq \frac{1}{\sqrt{D}} \cdot \sqrt{\sum_{i=1}^{D} 1^2} \cdot \sqrt{\sum_{i=1}^{D} (\mathbf{x}[i] - \mathbf{y}[i])^2} = \|\mathbf{x} - \mathbf{y}\|$$
 (40)

where (38) is by definition. (39) is by triangle's inequality. (40) is due to Cauchy-Schwarz inequality.

Recall that  $P^{-1}$ **o** is a random vector which follows the uniform distribution on the unit sphere. There is a well-known lemma [82] which presents that passing a random vector which follows the uniform distribution on the unit sphere through a Lipschitz function produces a highly concentrated distribution. The specific result is presented as follows.

LEMMA B.2. ([82]) Let x be a D-dimensional random vector which follows the uniform distribution on the unit sphere,  $f(\mathbf{x})$  is a Lipschitz function with the Lipschitz constant of L. Then

$$\mathbb{P}\left\{|f(\mathbf{x}) - \mathbb{E}\left[f(\mathbf{x})\right]| \ge t\right\} \le 2\exp\left(-\frac{cDt^2}{L^2}\right) \tag{41}$$

where c is a constant.

Plugging in our  $f(\mathbf{x})$  immediately yields the following result.

$$\mathbb{P}\left\{\left|\left\langle\bar{\mathbf{o}},\mathbf{o}\right\rangle - \mathbb{E}\left[\left\langle\bar{\mathbf{o}},\mathbf{o}\right\rangle\right]\right| \ge t\right\} \le 2\exp\left(-cDt^2\right) \tag{42}$$

$$\mathbb{P}\left\{\left|\left\langle\bar{\mathbf{o}},\mathbf{o}\right\rangle - \mathbb{E}\left[\left\langle\bar{\mathbf{o}},\mathbf{o}\right\rangle\right]\right| \ge \frac{u}{\sqrt{D}}\right\} \le 2\exp\left(-cu^2\right) \tag{43}$$

where (43) is by letting  $u = t\sqrt{D}$ . The conclusion shows that  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$ is highly concentrated around its expectation. It will not deviate from its expectation by  $\Omega(1/\sqrt{D})$  with high probability.

