# F STRICTLY BALANCED GATING

Due to some peculiarities in our infrastructure which have since been fixed, at the time we ran some of the machine translation experiments, our models ran faster if every expert received exactly the same batch size. To accommodate this, we used a different gating function which we describe below.

Recall that we define the softmax gating function to be:

$$G_{\sigma}(x) = Softmax(x \cdot W_{\sigma}) \tag{15}$$

**Sparse Gating (alternate formulation):** To obtain a sparse gating vector, we multiply  $G_{\sigma}(x)$  component-wise with a sparse mask  $M(G_{\sigma}(x))$  and normalize the output. The mask itself is a function of  $G_{\sigma}(x)$  and specifies which experts are assigned to each input example:

$$G(x)_{i} = \frac{G_{\sigma}(x)_{i} M(G_{\sigma}(x))_{i}}{\sum_{j=1}^{n} G_{\sigma}(x)_{j} M(G_{\sigma}(x))_{j}}$$
(16)

**Top-K Mask:** To implement top-k gating in this formulation, we would let M(v) = TopK(v, k), where:

$$TopK(v,k)_i = \begin{cases} 1 & \text{if } v_i \text{ is in the top } k \text{ elements of } v. \\ 0 & \text{otherwise.} \end{cases}$$
 (17)

**Batchwise Mask:** To force each expert to receive the exact same number of examples, we introduce an alternative mask function,  $M_{batchwise}(X,m)$ , which operates over batches of input vectors. Instead of keeping the top k values per example, we keep the top m values per expert across the training batch, where  $m = \frac{k|X|}{n}$ , so that each example is sent to an average of k experts.

$$M_{batchwise}(X,m)_{j,i} = \begin{cases} 1 & \text{if } X_{j,i} \text{ is in the top } m \text{ values for to expert } i \\ 0 & \text{otherwise} \end{cases}$$
 (18)

As our experiments suggest and also observed in (Ioffe & Szegedy, 2015), using a batchwise function during training (such as  $M_{batchwise}$ ) requires modifications to the inference when we may not have a large batch of examples. Our solution to this is to train a vector T of per-expert threshold values to approximate the effects of the batchwise mask. We use the following mask at inference time:

$$M_{threshold}(x,T)_i = \begin{cases} 1 & \text{if } x_i > T_i \\ 0 & \text{otherwise} \end{cases}$$
 (19)

To learn the threshold values, we apply an additional loss at training time which is minimized when the batchwise mask and the threshold mask are identical.

$$L_{batchwise}(X,T,m) = \sum_{j=1}^{|X|} \sum_{i=1}^{n} (M_{threshold}(X,T)_i - M_{batchwise}(X,m)_{j,i})(X_{j,i} - T_i)$$
 (20)

#### G ATTENTION FUNCTION

The attention mechanism described in GNMT (Wu et al., 2016) involves a learned "Attention Function"  $A(x_i, y_j)$  which takes a "source vector"  $x_i$  and a "target vector"  $y_j$ , and must be computed for every source time step i and target time step j. In GNMT, the attention function is implemented as a feed forward neural network with a hidden layer of size n. It can be expressed as:

$$A_{GNMT}(x_i, y_j) = \sum_{d=1}^{n} V_d tanh((x_i U)_d + (y_j W)_d)$$
 (21)

Where U and W are trainable weight matrices and V is a trainable weight vector.

For performance reasons, in our models, we used a slightly different attention function:

$$A(x_i, y_j) = \sum_{d=1}^{n} V_d tanh((x_i U)_d) tanh((y_j W)_d)$$
(22)

With our attention function, we can simultaneously compute the attention function on multiple source time steps and multiple target time steps using optimized matrix multiplications. We found little difference in quality between the two functions.