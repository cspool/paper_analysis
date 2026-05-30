# <span id="page-26-0"></span>C Mechanics of Selective SSMs

Proof of Theorem [1.](#page-7-1) Consider a selective SSM (Algorithm [2\)](#page-5-2) with = 1, = −1, = 1, <sup>Δ</sup> = Linear(), <sup>Δ</sup> = softplus. The corresponding continuous-time SSM [\(1\)](#page-2-0) is

$$h(t) = -h(t) + x(t)$$

which is also called a leaky integrator.

The discretization step size is

$$\Delta_t = \tau_{\Delta}(\operatorname{Parameter} + s_{\Delta}(x_t))$$
= softplus(Parameter + Linear( $x_t$ ))
= softplus(Linear( $x_t$ ))

where we observe that the parameter can be viewed as a learnable bias and folded into the linear projection.

Now applying the zero-order hold (ZOH) discretization formulas:

$$\overline{A}_t = \exp(\Delta A) = \frac{1}{1 + \exp(\mathsf{Linear}(x_t))} = \sigma(-\mathsf{Linear}(x_t))$$

$$= 1 - \sigma(\mathsf{Linear}(x_t))$$

$$\overline{B}_t = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B = -(\exp(\Delta A) - I) = 1 - \overline{A}$$

$$= \sigma(\mathsf{Linear}(x_t)).$$

Thus the final discrete recurrence [\(2a\)](#page-2-4) is

$$g_t = \sigma(\mathsf{Linear}(x_t))$$
  
$$h_t = (1 - g_t)h_{t-1} + g_t x_t$$

as desired. □

