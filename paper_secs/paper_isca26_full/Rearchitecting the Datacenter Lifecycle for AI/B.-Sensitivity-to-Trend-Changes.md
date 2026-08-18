# *B. Sensitivity to Trend Changes*

Our conclusions assume projected trends in model/demand growth, compute density, memory bandwidth/capacity scaling, interconnect scaling, accelerator TDP, and pricing trajectories. To evaluate robustness, we independently sweep the annual

<span id="page-12-0"></span>

| Trend Dimension      | BOP Optimal<br>(max ±%) | BOP in Top-5<br>(max ±%) |
|----------------------|-------------------------|--------------------------|
| Model Growth         | ±45%                    | ±80%                     |
| Demand Growth        | ±50%                    | ±85%                     |
| Compute Density      | ±35%                    | ±75%                     |
| Memory BW/Capacity   | ±40%                    | ±80%                     |
| Interconnect Scaling | ±55%                    | ±85%                     |
| Accelerator TDP      | ±45%                    | ±75%                     |
| Pricing Trajectory   | ±50%                    | ±85%                     |

TABLE X: Robustness of the Baseline-Optimal Policy (BOP) to trend deviations. Lower ranges indicate fragility.

scaling rate of each dimension by a relative percentage deviation from its baseline projection (*e.g.*, ±45% correspond to a 45% increase or decrease in the assumed annual improvement/growth rate) and re-run the simulations.

For each trend, we measure whether the baseline-optimal policy (BOP) remains (1) strictly optimal and (2) within the top-5 lowest-TCO policies. This quantifies *tolerance margins* of the BOP and distinguishes robust conclusions from scenario-dependent ones. [Table X](#page-12-0) summarizes the results. Across all dimensions, the BOP remains strictly optimal under substantial deviations: from ±35% (compute density) to ±55% (interconnect scaling). Even under larger deviations of ±75- 85%, it consistently remains within the top-5 policies. Hence, the optimal refresh strategy is robust to large uncertainty.

*Compute density* and *memory bandwidth/capacity* exhibit tight margins (±35% and ±40%, respectively), reflecting their direct influence on the dominant bottleneck in roofline models. In contrast, interconnect scaling, demand growth, and pricing trajectories primarily affect the magnitude of lifecycle TCO.

Workload dynamics influence refresh timing: reducing *model growth* by 40-50% favors longer lifecycles and additional generation skipping, whereas upward shifts in model size or demand accelerate memory and interconnect bottlenecks and trigger earlier refresh. *Memory bandwidth* scaling exhibits high leverage in memory-bound regimes: slowing it by ≈35% can advance refresh by one generation even if compute density follows baseline trends. *Pricing* largely affects the magnitude of TCO rather than policy structure: even under ±50% pricing deviations the BOP remains strictly optimal.

Importantly, our framework does not prescribe a single universally optimal strategy, as it is scenario-dependent. However, a unified conclusion remains robust across all tested regimes: *AI datacenters benefit from refresh strategies codesigned with workload demand, model properties, and crossstack efficiency*, rather than fixed generational upgrade rules.

