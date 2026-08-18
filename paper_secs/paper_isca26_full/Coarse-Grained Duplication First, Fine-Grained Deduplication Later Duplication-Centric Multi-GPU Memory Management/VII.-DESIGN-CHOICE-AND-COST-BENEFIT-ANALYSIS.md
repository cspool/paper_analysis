# VII. DESIGN CHOICE AND COST-BENEFIT ANALYSIS

#### TABLE VI ADVANTAGES OF CDFD OVER COARSEDUP

| Method | Performance under<br>memory-sufficient<br>conditions | Performance under<br>oversubscription | Average Power<br>Increase | Area and<br>Space Overhead |  |
|--------|------------------------------------------------------|---------------------------------------|---------------------------|----------------------------|--|
|        | CoarseDup 1.55× over baseline                        | 1.28× over baseline                   | 5.74 Watt                 | None                       |  |
| CDFD   | 1.66× over baseline                                  | 1.55× over baseline                   | 5.58 Watt                 | 12800 B                    |  |

To leverage our insights about NVLink, including nonlinear latency–size scaling, negligible intra- and inter-link contention, and ample bandwidth headroom, we consider two designs: CoarseDup (coarse-grained duplication only) and CDFD (our full design). Table VI presents a cost–benefit comparison.

From a performance robustness perspective, CDFD consistently outperforms CoarseDup under both memorysufficient and oversubscribed settings. Under sufficient memory, CDFD achieves a 1.66× speedup over the baseline, compared to 1.55× for CoarseDup. The advantage becomes more pronounced under oversubscription (1.55× vs. 1.28×), demonstrating stronger robustness under memory pressure. This robustness stems from CDFD's adaptive feedback mechanism, which dynamically regulates the duplication ratio based on runtime conditions. From a memory utilization and power perspective, CDFD also compares favorably. By incorporating fine-grained deduplication, it avoids retaining low-benefit duplicated pages, improving effective GPU memory utilization and reducing redundant duplication and eviction. As a result, CDFD slightly lowers power overhead (5.58 W vs. 5.74 W), showing that its performance gains do not come at higher energy cost. From a hardware cost perspective, CDFD introduces only modest overhead. Unlike CoarseDup, which requires no additional storage, CDFD incurs an extra 12,800 B of storage. This cost is negligible relative to modern GPU resources, especially given its performance benefits.

Overall, CDFD provides a better tradeoff than CoarseDup, achieving higher performance, improved robustness and utilization, slightly lower power overhead, and minimal hardware cost.

# VII. DESIGN CHOICE AND COST-BENEFIT ANALYSIS

#### TABLE VI ADVANTAGES OF CDFD OVER COARSEDUP

| Method | Performance under<br>memory-sufficient<br>conditions | Performance under<br>oversubscription | Average Power<br>Increase | Area and<br>Space Overhead |  |
|--------|------------------------------------------------------|---------------------------------------|---------------------------|----------------------------|--|
|        | CoarseDup 1.55× over baseline                        | 1.28× over baseline                   | 5.74 Watt                 | None                       |  |
| CDFD   | 1.66× over baseline                                  | 1.55× over baseline                   | 5.58 Watt                 | 12800 B                    |  |

To leverage our insights about NVLink, including nonlinear latency–size scaling, negligible intra- and inter-link contention, and ample bandwidth headroom, we consider two designs: CoarseDup (coarse-grained duplication only) and CDFD (our full design). Table VI presents a cost–benefit comparison.

From a performance robustness perspective, CDFD consistently outperforms CoarseDup under both memorysufficient and oversubscribed settings. Under sufficient memory, CDFD achieves a 1.66× speedup over the baseline, compared to 1.55× for CoarseDup. The advantage becomes more pronounced under oversubscription (1.55× vs. 1.28×), demonstrating stronger robustness under memory pressure. This robustness stems from CDFD's adaptive feedback mechanism, which dynamically regulates the duplication ratio based on runtime conditions. From a memory utilization and power perspective, CDFD also compares favorably. By incorporating fine-grained deduplication, it avoids retaining low-benefit duplicated pages, improving effective GPU memory utilization and reducing redundant duplication and eviction. As a result, CDFD slightly lowers power overhead (5.58 W vs. 5.74 W), showing that its performance gains do not come at higher energy cost. From a hardware cost perspective, CDFD introduces only modest overhead. Unlike CoarseDup, which requires no additional storage, CDFD incurs an extra 12,800 B of storage. This cost is negligible relative to modern GPU resources, especially given its performance benefits.

Overall, CDFD provides a better tradeoff than CoarseDup, achieving higher performance, improved robustness and utilization, slightly lower power overhead, and minimal hardware cost.

