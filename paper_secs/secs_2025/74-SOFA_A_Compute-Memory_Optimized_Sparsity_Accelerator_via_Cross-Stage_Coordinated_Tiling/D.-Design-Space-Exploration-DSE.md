# D. Design Space Exploration (DSE)

In the SOFA algorithm mechanism, the tiling size, i.e.,  $B_c$  in each layer and top-k form an interesting design space. For larger  $B_c$ , i.e. smaller  $T_c$  ( $S = B_c \times T_c$ ), inference accuracy tends to increase. However, sorting complexity escalates significantly, yet the computation complexity of SU-FA decreases. On the contrary, when  $B_c$  decreases, it will lead to opposite effects. We provide each of the hyperparameters with plenty of options as 1)  $Tc_i$ : 2 – 32, step=2; 2) Top-k:

