# **B. Derivation of Quantization Errors**

In this section, we derive the expected quantization error, which measures the difference between the original LMPUs and their quantized values, where p(x) represents a standard normal distribution. The error is formulated as

$$\mathbb{E}\left[(\Delta w - \Delta \bar{w})^2\right] = \sum_{r=0}^{R} \int_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx \tag{16}$$

where  $q_r$  is the quantization level. To evaluate the integral, we expand the squared term as follows:

$$\int (x - q_r)^2 p(x) dx = \frac{1}{\sqrt{2\pi}} \int (x - q_r)^2 e^{-\frac{x^2}{2}} dx$$

$$= \frac{1}{\sqrt{2\pi}} \left( \underbrace{\int x^2 e^{-\frac{x^2}{2}} dx}_{P_1} \underbrace{-2q_r \int x e^{-\frac{x^2}{2}} dx}_{P_2} + \underbrace{q_r^2 \int e^{-\frac{x^2}{2}} dx}_{P_3} \right)$$
(17)

We now calculate each term  $P_1$ ,  $P_2$ , and  $P_3$ . Let  $t = \frac{x}{\sqrt{2}}$ , which transforms  $P_1$  into

$$P_1 = 2\sqrt{2} \int t^2 e^{-t^2} dt$$

$$= -\sqrt{2}te^{-t^2} + \sqrt{2} \int e^{-t^2} dt \quad \left(\because \int u dv = uv - \int v du \quad \text{where } u = t \text{ and } dv = te^{-t^2} dt\right)$$
(18)

The definite integral over the quantization boundaries is then given by

$$-\sqrt{2}\left[te^{-t^2}\right]_{u_r/\sqrt{2}}^{u_{r+1}/\sqrt{2}} + \sqrt{2}\int_{u_r/\sqrt{2}}^{u_{r+1}/\sqrt{2}}e^{-t^2}dt = \left(u_re^{-\frac{u_r^2}{2}} - u_{r+1}e^{-\frac{u_{r+1}^2}{2}}\right) + \sqrt{\frac{\pi}{2}}\left(\operatorname{erf}\left(\frac{u_{r+1}}{\sqrt{2}}\right) - \operatorname{erf}\left(\frac{u_r}{\sqrt{2}}\right)\right)$$
(19)

Also, we can evaluate the definite integral of  $P_2$  over the qunatization boundaries as follows:

$$-2q_r \int_{u_r}^{u_{r+1}} x e^{-\frac{x^2}{2}} dx = 2q_r \left[ e^{-x^2} \right]_{u_r}^{u_{r+1}} \qquad \left( \because \int x e^{-\frac{x^2}{2}} dx = -e^{-\frac{x^2}{2}} \right)$$
$$= 2q_r \left( e^{-\frac{u_{r+1}^2}{2}} - e^{-\frac{u_r^2}{2}} \right)$$
(20)

Finally, we can easily obtain the definite integral of  $P_3$  over the qunatization boundaries by substituting  $t = \frac{x}{\sqrt{2}}$ , as follows:

<span id="page-11-0"></span>
$$q_r^2 \int_{u_r}^{u_{r+1}} e^{-\frac{x^2}{2}} dx = \sqrt{2} q_r^2 \int_{u_r/\sqrt{2}}^{u_{r+1}/\sqrt{2}} e^{-t^2} dt$$

$$= \sqrt{\frac{\pi}{2}} q_r^2 \left( \left( \text{erf} \left( \frac{u_{r+1}}{\sqrt{2}} \right) - \text{erf} \left( \frac{u_r}{\sqrt{2}} \right) \right)$$
(21)

Combining all the above derivations and unrolling the sum, we can obtain the final expression for the expected quantizaion error as follows:

$$\sum_{r=0}^{R} \int_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx = \sum_{r=0}^{R} \left\{ \frac{1}{\sqrt{2\pi}} (2q_r - u_{r+1}) e^{-\frac{u_{r+1}^2}{2}} - \frac{1}{\sqrt{2\pi}} (2q_r - u_r) e^{-\frac{u_r^2}{2}} + \frac{1}{2} (q_r^2 + 1) \left( \operatorname{erf} \left( \frac{u_{r+1}}{\sqrt{2}} \right) - \operatorname{erf} \left( \frac{u_r}{\sqrt{2}} \right) \right) \right\} \\
= \frac{1}{2} (q_R^2 + 1) - \sqrt{\frac{2}{\pi}} q_1 e^{-\frac{u_1^2}{2}} - \frac{1}{2} q_1^2 \operatorname{erf} \left( \frac{u_1}{\sqrt{2}} \right) \\
+ \sqrt{\frac{2}{\pi}} \sum_{r=1}^{R-1} (q_r - q_{r+1}) e^{-\frac{u_{r+1}^2}{2}} + \frac{1}{2} \sum_{r=1}^{R-1} (q_r^2 - q_{r+1}^2) \operatorname{erf} \left( \frac{u_{r+1}}{\sqrt{2}} \right). \tag{22}$$

