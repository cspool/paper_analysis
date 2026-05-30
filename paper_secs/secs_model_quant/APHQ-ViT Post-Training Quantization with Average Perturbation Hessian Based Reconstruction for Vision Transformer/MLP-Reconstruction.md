# MLP Reconstruction:

- 6: Replace the GELU activation of MLP by ReLU.
- 7: **for**  $i = 0, \dots, \max_{i}$  **ter do**
- 8: Calculate  $O_{\mathrm{Direct}}$  and  $O_{\mathrm{Clamp}}$  by Eqs. (11) and (12).
- 9: Calculate  $\mathcal{L}_{\text{Distill}}$  by Eq. (14).
- 10: Perform backward propagation and update MLP.

11: end for

