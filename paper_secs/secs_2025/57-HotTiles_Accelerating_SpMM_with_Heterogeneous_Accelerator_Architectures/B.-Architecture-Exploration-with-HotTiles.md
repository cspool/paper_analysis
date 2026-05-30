# B. Architecture Exploration with HotTiles

In this subsection, we use the performance prediction capabilities of *HotTiles* to explore different heterogeneous architecture alternatives. We focus mostly on the SPADE-Sextans architecture. We examine heterogeneous architectures that have more workers of one type at the expense of fewer workers of the other type. Specifically, we start with our scale 4 SPADE-Sextans architecture in Table IV, which we call 4-4. Then we consider "skewed" SPADE-Sextans architectures, such as 0-8, which includes no cold workers and hot workers with the same parameters as a scale 8 SPADE-Sextans architecture. We focus on skewed architectures where the sum of cold and hot scales equals to 8, namely 0-8, 1-7, 2-6, 3-5, 4-4, 5-3, 6-2, 7-1, and 8-0. We call these architectures *iso-scale*.

We consider two different scenarios. In the first one, we assume that the architecture is not reconfigurable, and want to find which of the 9 iso-scale architectures delivers the highest average performance for our benchmarks. In the second scenario, we assume that the architecture is reconfigurable, and want to find which of the 9 possible configurations delivers the highest performance for each individual benchmark. The first scenario corresponds to an ASIC accelerator, while the second one corresponds, e.g., to an FPGA-based one. In both cases, we use the execution time predictions made by *HotTiles*. For simplicity, we ignore any differences in area or power among the 9 iso-scale architectures. A future, more detailed analysis can compare performance per unit area or unit power.

![](_page_12_Figure_12.jpeg)

Fig. 16: Predicted and actual average performance of different iso-scale heterogeneous architectures.

We focus first on the first scenario. Figure 16 shows, for each iso-scale architecture, the performance predicted by *HotTiles* and the actual performance. The performance is given as speedup over the base SPADE-Sextans architecture (i.e., 4-4) and is the average across all the benchmarks of

![](_page_13_Figure_0.jpeg)

Fig. 17: Error in the predicted execution time of different homogeneous and heterogeneous executions.

Table V. We observe that, although the predicted performance is slightly higher than the actual one for the architectures that are dominated by hot workers, the trends of predicted and actual performance are the same. In the 7-1 and 8- 0 architectures, which are dominated by cold workers, the predicted performance is slightly lower than the actual one. This is because, as explained before, *HotTiles* ignores reuse in caches. Overall, the architecture predicted to perform the best (3-5) is also the one that performs the best.

We now focus on the second scenario, where the heterogeneous architecture is reconfigurable on a per-matrix basis. Table IX displays, for each matrix, the predicted best architecture, its speedup with respect to 4-4, the actual best architecture, its speedup with respect to 4-4, and whether the prediction is correct. We see that *HotTiles* selects the best architecture out of the nine iso-scale alternatives in 50% of the matrices. *HotTiles* predictions have some bias towards using more hot workers than needed. Again, this is due to ignoring caching effects and therefore under-predicting the true capabilities of cold workers. Despite this limitation, we observe that reconfiguring the architecture based on the predictions delivers a 1.23x speedup over the baseline 4-4 architecture. This is an encouraging result, considering that the oracle speedup is 1.33x.

TABLE IX: Predicted and actual best architecture per matrix.

| Matrix | Pred.<br>Best<br>Arch. | Speedup<br>of Pred.<br>Best | Actual<br>Best<br>Arch. | Speedup<br>of Actual<br>Best | Correct<br>Pred? |
|--------|------------------------|-----------------------------|-------------------------|------------------------------|------------------|
| ski    | 3-5                    | 0.95                        | 8-0                     | 1.11                         | N                |
| pap    | 1-7                    | 0.78                        | 4-4                     | 1.00                         | N                |
| del    | 3-5                    | 1.01                        | 8-0                     | 1.28                         | N                |
| dgr    | 1-7                    | 1.59                        | 1-7                     | 1.59                         | Y                |
| kro    | 0-8                    | 1.93                        | 0-8                     | 1.93                         | Y                |
| myc    | 0-8                    | 1.63                        | 0-8                     | 1.63                         | Y                |
| pac    | 2-6                    | 1.16                        | 2-6                     | 1.16                         | Y                |
| ser    | 0-8                    | 1.36                        | 0-8                     | 1.36                         | Y                |
| pok    | 3-5                    | 0.89                        | 8-0                     | 1.14                         | N                |
| wik    | 3-5                    | 0.96                        | 8-0                     | 1.08                         | N                |
| AVG    |                        | 1.23                        |                         | 1.33                         | 50%              |

Finally, Figure 17 shows the error in the execution time predicted by *HotTiles* compared to the actual execution time for SPADE-Sextans and PIUMA. The figure also includes the error in the predictions for the homogeneous executions (*HotOnly* and *ColdOnly*), although we do not directly utilize these predictions. For both SPADE-Sextans and PIUMA, the average error between the predicted and actual execution time is relatively low, at 4.8%, 19.6%, and 12.4% for *HotOnly*, *ColdOnly*, and *HotTiles*, respectively. The highest error occurs for certain matrices (pap, myc, pac, ser) in the *ColdOnly* execution. Upon further investigation, we found that these matrices display significant *Din* reuse through caches. Since our model ignores caching effects, the predicted *ColdOnly* execution time is higher than the actual one. This prediction error is more significant in SPADE-Sextans because the caches of the SPADE PEs are larger than the caches of the PIUMA MTPs. We believe that extending our modeling methodology to account for caching effects can further enhance the effectiveness of *HotTiles* predictions.

