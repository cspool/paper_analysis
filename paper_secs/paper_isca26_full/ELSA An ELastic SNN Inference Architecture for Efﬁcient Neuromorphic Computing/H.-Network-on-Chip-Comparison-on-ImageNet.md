# H. Network-on-Chip Comparison on ImageNet

We compare the network-on-chip (NoC) traffic and energy of ELSA with other multi-core SNN accelerators, including MorphIC [17] and TrueNorth [11], as summarized in Tab. VIII. For fairness, all NoCs are configured identically (6×6 2D mesh, same bandwidth and flit buffer size) and modified to support slayernorm, ssoftmax, and im2col for the benchmarks

TABLE VIII: NoC traffic-energy comparisons.

<span id="page-10-1"></span>

|                                               | Morp                                  | ohic                                 | TrueN                                 | orth                                 | ELS                                   | A                                    |
|-----------------------------------------------|---------------------------------------|--------------------------------------|---------------------------------------|--------------------------------------|---------------------------------------|--------------------------------------|
|                                               | Traffic                               | Energy                               | Traffic                               | Energy                               | Traffic                               | Energy                               |
| ResNet18<br>ResNet34<br>ResNet50<br>ViT Small | 15.5MB<br>29.9MB<br>92.8MB<br>994.8MB | 3.24µJ<br>6.81µJ<br>32.1µJ<br>0.33mJ | 12.9MB<br>26.7MB<br>72.3MB<br>701.6MB | 3.22µJ<br>5.69µJ<br>30.5µJ<br>0.26mJ | 11.2MB<br>21.0MB<br>52.6MB<br>560.3MB | 2.63µJ<br>5.05µJ<br>19.3µJ<br>0.18mJ |

<span id="page-10-2"></span>![](_page_10_Figure_2.jpeg)

Fig. 20: Elastic Inference in ELSA using three different pipeline schedulings in ResNet50 (Left) and ViT Small(Right). X-axis: latency (ms); Y-axis: top-1 accuracy (%).

in Tab. II. Benefiting from the bundled AER and multi-path routing, ELSA achieves the lowest NoC traffic and energy across all benchmarks, with 20.5% traffic and 24.3% energy reductions over TrueNorth on average.

#### I. Elastic Inference v.s. Spine/Token-wise Pipeline

Our spine/token-wise pipeline produces outputs at fine granularity. Each token and spine can exit independently once confidence is high. This design aligns perfectly with elastic inference. As shown in Fig. 20, compared to the other coarsegrained pipeline (no pipeline or layer-wise pipeline) used in prior SNN accelerators [9], [11], [14], the accuracy-latency curve of spine/token-wise pipeline shifts leftward, showing a faster response. As a result, spine/token-wise pipeline achieves an average  $2.0\times$  earlier on ViT-S and  $2.4\times$  speedup on ResNet50 compared to other pipelines at the same accuracy. This shows that spine/token-wise pipeline enables lower-latency elastic inference, critical for real-time applications.

#### J. Network Congestion Analysis

To analyze on-chip network congestion, we vary the number of flits by adjusting the number of effective spikes packed into each flit. We then measure the inference cycles under different data injection rates, as shown in Fig. 21. All flits have a fixed size of 512 bits. The gray dotted line in Fig. 21 (left) marks the baseline injection rate of 0.031 for ResNet50 in practical cases. As shown in Fig. 21 (left), when the injection rate exceeds 0.04, which is the 10× number of flits in the practical cases, the on-chip network becomes congested, and the inference cycles increase dramatically. Nevertheless, the cycle reduction achieved by elastic inference remains stable and is always larger than 19%. This indicates that the network congestion does not affect the benefit of elastic inference. Fig. 21 (right) explains the reason. Under different injection rates, the cycles of all time-steps increase proportionally, rather than being stalled at the first time-step.

