# <span id="page-9-2"></span>**Algorithm 1** Temperature Correction Based on Neighbor Influence on Air-Cooled multi-GPUs Servers

```
Output: Corrected temperature value

1: Select closest neighbor G_j

arg max Correlation (G_i, G_k)
```

- arg max Correlation $(G_i, G_k)$ 2. Compute neighbour temperature delta:  $\Delta T_i = T_i$
- 2: Compute neighbour temperature delta:  $\Delta T_j = T_j \min(T_j)$
- 3: Fit linear regression:  $T_i = \alpha \cdot \Delta T_i + \beta$

Input: Temperature reading for a GPU

- 4: **if**  $\alpha > 0$  (significant) **then**
- 5: Temperature corrected:  $T_i \leftarrow T_i \alpha \cdot \Delta T_j$
- 6: else
- 7: Temperature corrected:  $T_i \leftarrow T_i$
- 8: end if
- 9: **return** Temperature corrected

When using the density plot to define usage bounds (for each sub-degree, we select the most likely usage level), we achieve an accuracy of 76.7%, with a precision of 98.6% for idle, 82.0% for 30% usage, 59.5% for 60% usage, and 68.6% for fully loaded GPUs. When focusing solely on idle and non-idle states, we achieve a precision of 98.6% and 97.0%, respectively. This demonstrates the feasibility of inferring workload intensity from temperature variations. Moreover,

<span id="page-10-0"></span>![](_page_10_Figure_2.jpeg)

**Figure 10.** Temperature readings (initial and corrected) under 4xA100 GPUs with 4 stress levels

each intermediate compute level further confirms that the approach can be extended to lower TDP GPUs.

