 # Retrieve top k experts (1)

Then we apply nonlinear activations (such as softmax or sigmoid) to the query-key inner products of these top k experts to obtain the router scores.

$$g_i(x) = s(q(x)^T k_i)$$
