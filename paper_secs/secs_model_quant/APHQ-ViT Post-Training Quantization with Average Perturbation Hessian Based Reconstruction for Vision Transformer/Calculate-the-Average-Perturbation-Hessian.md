# Calculate the Average Perturbation Hessian:

- 1: Compute the raw output O of  $\mathcal{B}$  based on  $\mathcal{D}_{calib}$ .
- 2: Calculate the perturbed outputs  $O^+$  and  $O^-$ .
- 3: Compute  $f(O)/f(O^+)/f(O^-)$  by forward passing  $O/O^+/O^-$  through the remaining blocks of  $\mathcal{M}$ .
- 4: Calculate  $\mathcal{L}(f(\boldsymbol{O}), f(\boldsymbol{O}^+))$  and  $\mathcal{L}(f(\boldsymbol{O}), f(\boldsymbol{O}^-))$  and obtain  $\bar{\boldsymbol{J}}^{(\boldsymbol{O}^+)}$  and  $\bar{\boldsymbol{J}}^{(\boldsymbol{O}^-)}$  by backward propagation.
- 5: Calculate the average perturbation Hessian matrix  $\bar{H}$  based on Eq. (8).

