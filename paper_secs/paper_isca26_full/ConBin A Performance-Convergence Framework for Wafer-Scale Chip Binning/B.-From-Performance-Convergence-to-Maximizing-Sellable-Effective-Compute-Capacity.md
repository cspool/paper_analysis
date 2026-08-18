# *B. From Performance Convergence to Maximizing Sellable Effective Compute Capacity*

While prior works improve average performance, their inability to converge the performance distribution has profound implications for production yield and the aggregate guaranteed performance deliverable across bins. In wafer-scale manufacturing, each chip represents a substantial investment in silicon area and packaging cost, yet large inter-chip performance variance prevents consistent product qualification. When chip performance fluctuates excessively due to various fault distributions, vendors must adopt conservative binning thresholds and discard or down-bin many usable chips, lowering the fraction of premium-performance bins and the aggregate guaranteed performance deliverable across bins. Conversely, if performance variance across chips can be effectively reduced, the same functional yield can translate into a higher fraction of premium-performance chips and more stable product segmentation.

This effect can be formalized by defining the total sellable effective compute capacity (SECC) as the aggregate guaranteed performance deliverable across bins:

$$SECC = \sum P_k \cdot Y_k \tag{1}$$

where P<sup>k</sup> is the guaranteed deliverable performance level of bin k, determined by its performance threshold, and Y<sup>k</sup> is its yield fraction derived from the performance distribution f(p). For a given manufacturing yield, both P<sup>k</sup> and Y<sup>k</sup> are fundamentally constrained by the spread of performance across chips. When the performance variance σ p 2 is large, any reasonable threshold configuration either adopts conservative thresholds with lower P<sup>k</sup> to accommodate slow chips or sets higher thresholds at the cost of sharply reduced Y<sup>k</sup> for premium bins. In both cases, the total SECC is bounded by this variance. By contrast, when the performance distribution is well converged, thresholds can be set aggressively without sacrificing yield, allowing a larger portion of chips to qualify for high-priced bins. In this sense, performance convergence directly expands the feasible region of binning configurations that maximize SECC, transforming the same physical yield into higher aggregate guaranteed performance deliverable across bins.

Therefore, the ultimate objective of performance recovery in WSCs should extend beyond average performance. Rather than merely improving the mean performance of individual chips, the design goal must be to reduce inter-chip performance variance and achieve a globally converged performance distribution. By jointly optimizing redundancy and scheduling for variance reduction, we can systematically increase premiumbin yield and maximize the total SECC under manufacturing constraints. This motivates the design of ConBIN a performance-convergence framework for WSC binning that converges the performance distribution and bridges the gap among fault tolerance, performance uniformity, and commercial viability.

![](_page_4_Figure_6.jpeg)

Fig. 6. Overview of the ConBIN framework: from hardware-software optimization to performance binning.

# *B. From Performance Convergence to Maximizing Sellable Effective Compute Capacity*

While prior works improve average performance, their inability to converge the performance distribution has profound implications for production yield and the aggregate guaranteed performance deliverable across bins. In wafer-scale manufacturing, each chip represents a substantial investment in silicon area and packaging cost, yet large inter-chip performance variance prevents consistent product qualification. When chip performance fluctuates excessively due to various fault distributions, vendors must adopt conservative binning thresholds and discard or down-bin many usable chips, lowering the fraction of premium-performance bins and the aggregate guaranteed performance deliverable across bins. Conversely, if performance variance across chips can be effectively reduced, the same functional yield can translate into a higher fraction of premium-performance chips and more stable product segmentation.

This effect can be formalized by defining the total sellable effective compute capacity (SECC) as the aggregate guaranteed performance deliverable across bins:

$$SECC = \sum P_k \cdot Y_k \tag{1}$$

where P<sup>k</sup> is the guaranteed deliverable performance level of bin k, determined by its performance threshold, and Y<sup>k</sup> is its yield fraction derived from the performance distribution f(p). For a given manufacturing yield, both P<sup>k</sup> and Y<sup>k</sup> are fundamentally constrained by the spread of performance across chips. When the performance variance σ p 2 is large, any reasonable threshold configuration either adopts conservative thresholds with lower P<sup>k</sup> to accommodate slow chips or sets higher thresholds at the cost of sharply reduced Y<sup>k</sup> for premium bins. In both cases, the total SECC is bounded by this variance. By contrast, when the performance distribution is well converged, thresholds can be set aggressively without sacrificing yield, allowing a larger portion of chips to qualify for high-priced bins. In this sense, performance convergence directly expands the feasible region of binning configurations that maximize SECC, transforming the same physical yield into higher aggregate guaranteed performance deliverable across bins.

Therefore, the ultimate objective of performance recovery in WSCs should extend beyond average performance. Rather than merely improving the mean performance of individual chips, the design goal must be to reduce inter-chip performance variance and achieve a globally converged performance distribution. By jointly optimizing redundancy and scheduling for variance reduction, we can systematically increase premiumbin yield and maximize the total SECC under manufacturing constraints. This motivates the design of ConBIN a performance-convergence framework for WSC binning that converges the performance distribution and bridges the gap among fault tolerance, performance uniformity, and commercial viability.

![](_page_4_Figure_6.jpeg)

Fig. 6. Overview of the ConBIN framework: from hardware-software optimization to performance binning.

