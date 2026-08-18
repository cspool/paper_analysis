# <span id="page-4-0"></span>D. Estimating Performance Gradients

We now show how the Gradient Estimator computes  $\partial \text{perf}/\partial \text{power}$  at each time step t. To readily re-use the formulas at (1) and (2), we use instead the expression  $\partial BIPS/\partial P$  for this metric. Figure 5 shows that, at each time step t, the Gradient Estimator combines the current performance counters  $\mathbf{E}^{(t)}$ and the current frequency  $f^{(t)}$  with the regression coefficients  $(a_i \text{ from } (5) \text{ and } w_i \text{ from } (6)) \text{ and } \gamma \text{ from } (6), \text{ and generates the }$ power and performance models at this particular time, P(V)and CPI(f), shown in (4) and (3). These two equations take voltage V and frequency f as independent variables. They tell us what would be the power and performance if we set V and f to certain values. Note that the regression coefficients and  $\gamma$  are independent of the running workload, and are predetermined offline. By combining them with the workloaddependent hardware measurements at time step t, PowerGrad generates the power P(V) and performance CPI(f) models online at every time step t.

One of the major technical challenges of PowerGrad is differentiating the power and performance models to compute the performance gradients. This is because these models depend on variables that have intertwined dependencies: voltage, frequency, and performance counters are correlated with each other. As we cannot differentiate such a complex system directly, we need to approximate the relationships between the variables.

To this end, we make three assumptions. First, we approximate the core voltage as a second order polynomial of frequency [3]. Second, when the frequency changes, we assume that the values of the performance counters  $E_i^{(t)}$  are linearly proportional to the BIPS. In other words, we assume that  $E_i^{(t)} = e_i^{(t)} * BIPS$ , where  $e_i^{(t)}$  remain constant as frequency changes. This assumption is reasonable, since the absolute count of performance events like branch mispredictions or executed micro-ops are mostly proportional to the total executed instructions.

Finally, we assume that, as the frequency changes, the duration of the non-idle times is inversely proportional to the frequency, while the duration of the idle times remains constant. In this case, the Appendix shows that the utilization is a function of frequency f as follows:

$$util(f) = \frac{util^{(t)}}{util^{(t)} + (1 - util^{(t)}) * f/f^{(t)}}$$
(7)

where util(f) is the new utilization as a function of the changing frequency f, while  $util^{(t)}$  and  $f^{(t)}$  are the current utilization and frequency, respectively.

<span id="page-4-1"></span>We can now start to compute  $\partial BIPS/\partial P$  by breaking the power into idle and active power.

$$\frac{\partial BIPS}{\partial P} = (\frac{\partial P}{\partial BIPS})^{-1} = (\frac{\partial P_{active}}{\partial BIPS} + \frac{\partial P_{idle}}{\partial BIPS})^{-1} \quad (8)$$

Based on the second assumption,  $P_{active}$  from (6) can be expressed as  $\sum_i (w_i e_i * BIPS) * (V^{\gamma} + V)$ , whose gradient can be computed as follows:

<span id="page-4-4"></span>
$$\frac{\partial P_{active}}{\partial BIPS} = \sum_{i} (w_i e_i) * (V^{\gamma} + V) + \sum_{i} (w_i E_i) * \frac{\partial (V^{\gamma} + V)}{\partial V} \frac{\partial V}{\partial BIPS}$$
(9)

Since we assume that V is a second-order polynomial of f and it can be shown from (3) and (1) that BIPS is also a function of f, we can compute  $\partial V/\partial BIPS$  using the chain rule regarding f:

<span id="page-4-3"></span>
$$\frac{\partial V}{\partial BIPS} = \frac{\partial V}{\partial f} \left(\frac{\partial BIPS}{\partial f}\right)^{-1} \tag{10}$$

 $\partial V/\partial f$  can be obtained by differentiating the second-order polynomial.  $\partial BIPS/\partial f$  at the current frequency  $f^{(t)}$  can be derived from (1), (2), and (7) as shown in the Appendix:

$$\frac{\partial BIPS}{\partial f}(f^{(t)}) = \frac{util * CCPI}{CPI^2} - \frac{util(1 - util)}{CPI}$$
 (11)

With this, we have finished computing  $\partial P_{active}/\partial BIPS$ . Now, to compute  $\partial P_{idle}/\partial BIPS$ , we express it as:

<span id="page-4-6"></span>
$$\frac{\partial P_{idle}}{\partial BIPS} = \frac{\partial P_{idle}}{\partial V} \frac{\partial V}{\partial BIPS}$$
 (12)

The first term  $(\partial P_{idle}/\partial V)$  differentiates the idle power polynomial in (5). The second term  $(\partial V/\partial BIPS)$  reuses the value computed in (10). With this, we have computed the performance gradient in (8).

Since we collect hardware counter values for every core, the Gradient Estimator builds *per-core* performance and power models. Meanwhile, we want to optimize the efficiency of an entire cluster, which is hierarchically organized as multiple processors, and where each processor has multiple cores. To compute the performance gradient of a processor, we need to combine the performance gradients of all the cores in the processor. Because we assume that the power consumption of one core is independent of the power and performance of another core, we can use the chain rule of differentiation to compute the performance gradient of a processor  $\mathcal{G} = \partial BIPS/\partial P$ :

<span id="page-4-5"></span><span id="page-4-2"></span>
$$\mathcal{G} = \frac{\partial BIPS}{\partial P} = \sum_{i} (\frac{\partial BIPS}{\partial P_{i}} \frac{\partial P_{i}}{\partial P}) = \sum_{i} (\frac{\partial BIPS_{i}}{\partial P_{i}} \frac{P_{i}}{P})$$
(13)

