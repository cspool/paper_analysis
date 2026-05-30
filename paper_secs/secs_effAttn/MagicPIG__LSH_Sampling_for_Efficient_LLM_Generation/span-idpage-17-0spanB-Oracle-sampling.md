# <span id="page-17-0"></span>B Oracle sampling

The optimal sampling probability to guarantee estimation is unbiased in terms of lowest variance is not directly using attention score distribution  $w_i$ , but  $u_i' \propto w_i ||v_i||$ . However, this sampling probability is not optimal in terms of downstream accuracy and efficiency. We attribute this to two reasons. First, we observe the value norm of the sink token is significantly smaller than others (Figure 11), given its lower probability of being sampled, which may influence the functionality of attention. Second, due to the same reason,  $u_i' \propto w_i ||v_i||$  is flatter than  $w_i$ , resulting larger computation cost (as analyzed by Theorem 3.3).

