# 3 Preliminary

#### <span id="page-3-0"></span>3.1 Rotary Position Embedding (RoPE)

RoPE Su et al. [2024] is a position encoding method that encodes the absolute positions with different rotations and incorporates the explicit relative position dependency in the self-attention formulation. It applies different rotations to tokens in different positions to encode the position information.

Consider  $\mathbf{x}_t \in \mathbb{R}^d$  to be the embedding of the t-th token with the hidden size d. The RoPE operation upon  $x_t$  produces a representation  $x_t^R$  that encodes both semantic and positional information:

$$\mathbf{x}_{t}^{R} = \text{RoPE}(\mathbf{x}_{t}, t) = \begin{pmatrix} \mathbf{x}_{t}^{(1)} \\ \mathbf{x}_{t}^{(2)} \\ \mathbf{x}_{t}^{(3)} \\ \mathbf{x}_{t}^{(4)} \\ \vdots \\ \mathbf{x}_{t}^{(d-1)} \\ \mathbf{x}_{t}^{(d)} \end{pmatrix} \otimes \begin{pmatrix} \cos t\theta_{1} \\ \cos t\theta_{1} \\ \cos t\theta_{2} \\ \cos t\theta_{2} \\ \vdots \\ \cos t\theta_{d/2} \end{pmatrix} + \begin{pmatrix} -\mathbf{x}_{t}^{(2)} \\ \mathbf{x}_{t}^{(1)} \\ -\mathbf{x}_{t}^{(4)} \\ \mathbf{x}_{t}^{(3)} \\ \vdots \\ -\mathbf{x}_{t}^{(d)} \\ \mathbf{x}_{t}^{(d-1)} \end{pmatrix} \otimes \begin{pmatrix} \sin t\theta_{1} \\ \sin t\theta_{1} \\ \sin t\theta_{2} \\ \sin t\theta_{2} \\ \vdots \\ \sin t\theta_{d/2} \\ \sin t\theta_{d/2} \\ \sin t\theta_{d/2} \end{pmatrix}, \tag{1}$$

where  $\otimes$  denotes the element-wise multiplication of two vectors,  $\mathbf{x}_t^{(i)} \in \mathbb{R}$  denotes the *i*-th element of  $\mathbf{x}_t$ , and  $\theta_i = 10000^{-2(i-1)/d}$  is the *i*-th rotation angle. If we interpret every two elements in the embedding as a representation in the complex coordinate system, we can divide  $\mathbf{x}_t$  into paired dimensions, where the odd-indexed dimensions  $\mathbf{x}_t^{(2k-1)}$  represent the **real parts** and the even-indexed dimensions  $\mathbf{x}_t^{(2k)}$  represent the **imaginary parts**.

