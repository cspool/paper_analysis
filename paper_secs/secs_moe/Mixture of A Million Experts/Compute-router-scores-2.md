 # Compute router scores (2)

Finally, we compute the output by linearly combining the expert outputs weighted by the router scores.

$$f(x) = \sum_{i \in \mathbb{I}} g_i(x)e_i(x)$$
