# <span id="page-32-0"></span>F. Pseudo Code for Switch Transformers

Pseudocode for Switch Transformers in Mesh Tensorflow [\(Shazeer et al.,](#page-38-3) [2018\)](#page-38-3). No model parallelism is being used for the below code (see [5.4](#page-21-0) for more details).

```
import mesh tensorflow as mtf
def load balance loss(router probs, expert mask):
   """Calculate load−balancing loss to ensure diverse expert routing."""
   # router probs is the probability assigned for each expert per token.
   # router probs shape: [num cores, tokens per core, num experts]
   # expert index contains the expert with the highest router probability in one−hot format.
   # expert mask shape: [num cores, tokens per core, num experts]
   # For each core, get the fraction of tokens routed to each expert.
   # density 1 shape: [num cores, num experts]
   density 1 = mtf.reduce mean(expert mask, reduced dim=tokens per core)
   # For each core, get fraction of probability mass assigned to each expert
   # from the router across all tokens.
   # density 1 proxy shape: [num cores, num experts]
   density 1 proxy = mtf.reduce mean(router probs, reduced dim=tokens per core)
   # density l for a single core: vector of length num experts that sums to 1.
   # density l proxy for a single core: vector of length num experts that sums to 1.
   # Want both vectors to have uniform allocation (1/num experts) across all num expert elements.
   # The two vectors will be pushed towards uniform allocation when the dot product is minimized.
   loss = mtf.reduce mean(density 1 proxy ∗ density 1) ∗ (num experts ˆ 2)
   return loss
```

Figure 14: Pseudo code for the load balance loss for Switch Transformers in Mesh Tensorflow.

```
import mesh tensorflow as mtf
def router(inputs, capacity factor):
   """Produce the combine and dispatch tensors used for sending and
   receiving tokens from their highest probability expert. """
   # Core layout is split across num cores for all tensors and operations.
   # inputs shape: [num cores, tokens per core, d model]
   router weights = mtf.Variable(shape=[d model, num experts])
   # router logits shape: [num cores, tokens per core, num experts]
   router logits = mtf.einsum([inputs, router weights], reduced dim=d model)
   if is training:
       # Add noise for exploration across experts.
       router logits += mtf.random uniform(shape=router logits.shape, minval=1−eps, maxval=1+eps)
   # Convert input to softmax operation from bfloat16 to float32 for stability.
   router logits = mtf.to float32(router logits)
   # Probabilities for each token of what expert it should be sent to.
   router probs = mtf.softmax(router logits, axis=−1)
   # Get the top−1 expert for each token. expert gate is the top−1 probability
   # from the router for each token. expert index is what expert each token
   # is going to be routed to.
   # expert gate shape: [num cores, tokens per core]
   # expert index shape: [num cores, tokens per core]
   expert gate, expert index = mtf.top 1(router probs, reduced dim=num experts)
   # expert mask shape: [num cores, tokens per core, num experts]
   expert mask = mtf.one hot(expert index, dimension=num experts)
   # Compute load balancing loss.
   aux loss = load balance loss(router probs, expert mask)
   # Experts have a fixed capacity, ensure we do not exceed it. Construct
   # the batch indices, to each expert, with position in expert
   # make sure that not more that expert capacity examples can be routed to
   # each expert.
   position in expert = mtf.cumsum(expert mask, dimension=tokens per core) ∗ expert mask
   # Keep only tokens that fit within expert capacity.
   expert mask ∗= mtf.less(position in expert, expert capacity)
   expert mask flat = mtf.reduce sum(expert mask, reduced dim=experts dim)
   # Mask out the experts that have overflowed the expert capacity.
   expert gate ∗= expert mask flat
   # combine tensor used for combining expert outputs and scaling with router probability.
   # combine tensor shape: [num cores, tokens per core, num experts, expert capacity]
   combine tensor = (
       expert gate ∗ expert mask flat ∗
       mtf.one hot(expert index, dimension=num experts) ∗
       mtf.one hot(position in expert, dimension=expert capacity))
   # Cast back outputs to bfloat16 for the rest of the layer.
   combine tensor = mtf.to bfloat16(combine tensor)
   # Create binary dispatch tensor that is 1 if the token gets routed to the corresponding expert.
   # dispatch tensor shape: [num cores, tokens per core, num experts, expert capacity]
   dispatch tensor = mtf.cast(combine tensor, tf.bool)
   return dispatch tensor, combine tensor, aux loss
```

<span id="page-33-0"></span>Figure 15: Pseudo code for the router for Switch Transformers in Mesh Tensorflow.

```
import mesh tensorflow as mtf
def switch layer(inputs, n, capacity factor, num experts):
   """Distributed switch transformer feed−forward layer."""
   # num cores (n) = total cores for training the model (scalar).
   # d model = model hidden size (scalar).
   # num experts = total number of experts.
   # capacity factor = extra buffer for each expert.
   # inputs shape: [batch, seq len, d model]
   batch, seq len, d model = inputs.get shape()
   # Each core will route tokens per core tokens to the correct experts.
   tokens per core = batch ∗ seq len / num cores
   # Each expert will have shape [num cores, expert capacity, d model].
   # Each core is responsible for sending expert capacity tokens
   # to each expert.
   expert capacity = tokens per core ∗ capacity factor / num experts
   # Reshape to setup per core expert dispatching.
   # shape: [batch, seq len, d model] −> [num cores, tokens per core, d model]
   # Core layout: [n, 1, 1] −> [n, 1, 1]
   inputs = mtf.reshape(inputs, [num cores, tokens per core, d model])
   # Core Layout: [n, 1, 1] −> [n, 1, 1, 1], [n, 1, 1, 1]
   # dispatch tensor (boolean) shape: [num cores, tokens per core, num experts, expert capacity]
   # dispatch tensor is used for routing tokens to the correct expert.
   # combine tensor (float) shape: [num cores, tokens per core, num experts, expert capacity]
   # combine tensor used for combining expert outputs and scaling with router
   # probability.
   dispatch tensor, combine tensor, aux loss = router(inputs, expert capacity)
   # Matmul with large boolean tensor to assign tokens to the correct expert.
   # Core Layout: [n, 1, 1], −> [1, n, 1, 1]
   # expert inputs shape: [num experts, num cores, expert capacity, d model]
   expert inputs = mtf.einsum([inputs, dispatch tensor], reduce dims=[tokens per core])
   # All−to−All communication. Cores split across num cores and now we want to split
   # across num experts. This sends tokens, routed locally, to the correct expert now
   # split across different cores.
   # Core layout: [1, n, 1, 1] −> [n, 1, 1, 1]
   expert inputs = mtf.reshape(expert inputs, [num experts, num cores, expert capacity, d model])
   # Standard feed forward computation, where each expert will have its own
   # unique set of parameters.
   # Total unique parameters created: num experts ∗ (d model ∗ d ff ∗ 2).
   # expert outputs shape: [num experts, num cores, expert capacity, d model]
   expert outputs = feed forward(expert inputs)
   # All−to−All communication. Cores are currently split across the experts
   # dimension, which needs to be switched back to being split across num cores.
   # Core Layout: [n, 1, 1, 1] −> [1, n, 1, 1]
   expert outputs = mtf.reshape(expert outputs, [num experts, num cores, expert capacity, d model])
   # Convert back to input shape and multiply outputs of experts by the routing probability.
   # expert outputs shape: [num experts, num cores, tokens per core, d model]
   # expert outputs combined shape: [num cores, tokens per core, d model]
   # Core Layout: [1, n, 1, 1] −> [n, 1, 1]
   expert outputs combined = mtf.einsum([expert outputs, combine tensor], reduce dims=[tokens per core])
   # Remove tokens per core shapes used for local routing dispatching to match input shape.
   # Core Layout: [n, 1, 1] −> [n, 1, 1]
   outputs = mtf.reshape(expert outputs combined, [batch, seq len, d model])
   return outputs, aux loss
```

Figure 16: Pseudo code of the Switch Transformer layer in Mesh Tensorflow.

