# D. Performance of PowerGrad over the Baselines

We compare ML inference performance with PowerGrad and with the DPS, SLURM, and Fair baselines in power-

![](_page_9_Figure_0.jpeg)

<span id="page-9-0"></span>Fig. 9. Per-application performance, power, and performance gradients in a node running Llama-high and Llama-low under Fair and PowerGrad.

limited settings (sometimes severely limited). As seen in Table I, on the Legacy platform, we run all three variants of PowerGrad, while on the Accelerated platform, we run PG-central and PG-multi. We compare the designs using the geometric means of average and P95 response times for all applications, and normalize these means to those with Fair.

Figure 10 and Figure 11 show the results for the Legacy and Accelerated platforms, respectively. They show the average response times (top) and the P95 response times (bottom) for the different schemes for various cluster power limits. They also show bars for the geometric mean over all the power limits in the charts.

1. Comparing PowerGrad to Other Schemes. We see that all the schemes typically show lower response times than Fair. Among the schemes, SLURM and DPS have higher response times than PowerGrad. Further, when the cluster power limits are low, SLURM and DPS gain little or no improvement over Fair. Consider DPS first. Under severely-limited system power budgets, DPS marks all nodes as *high-priority*, because a node's priority is determined by how often it consumes most of its power budget over a time period. The DPS Readjusting module splits the high-priority power budgets equally among all high-priority nodes, resulting in an equal distribution like Fair if all the nodes are marked high-priority [8].

SLURM is also not efficient because it only considers whether an application used all its allocated power in the last control period. It does not consider the performance impact of increasing or reducing that power. This limitation is amplified at tighter power budgets as seen in Figures 10 and 11, because all the applications are power-starved. This is precisely the limitation addressed by the PowerGrad designs.

All PowerGrad designs distribute the power allocation across the nodes using gradient estimations. The result is reduced response times, which become more apparent relative to the other schemes with tighter power limits. Consider *PG-multi*, which is the best PowerGrad design. In the Legacy cluster, *PG-multi* reduces the average and tail latencies of the applications by a geometric mean of 22.9% and 23.0%, respectively, relative to the best baseline scheme. In the Accel-

![](_page_9_Figure_7.jpeg)

<span id="page-9-1"></span>Fig. 10. Average and P95 latencies (using geomean over all applications, and normalized to Fair) on the Legacy cluster at various power limits.

erated cluster, *PG-multi* reduces the average and tail latencies by a geometric mean of 9.0% and 9.9%, respectively, relative to the best baseline scheme. In the setting with the lowest power budgets, the gains of *PG-multi* are highest: for 55W per Legacy node (880W total), the average latency reductions are 23.6% and 27.4%, while for 115W per Accelerated node (1840W total), the reductions are 18.3% and 20.2%.

Note that because an Accelerated node has only one CPU, PowerGrad cannot leverage the fast 100ms-period Local Controller. Hence, PowerGrad shows less response time reductions in Accelerated nodes than in Legacy nodes.

![](_page_9_Figure_11.jpeg)

<span id="page-9-2"></span>Fig. 11. Average and P95 latencies (using geomean over all applications, and normalized to Fair) on the Accelerated cluster at various power limits.

2. Comparing the Different Configurations of PowerGrad.

