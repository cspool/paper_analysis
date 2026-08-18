# VI. SOFTWARE-LEVEL OPTIMIZATION AND PERFORMANCE BINNING

#### A. Communication Performance Impact of Contention

Manufacturing faults reshape routing paths and reduce available communication resources, causing both longer paths and significantly higher contention. While both factors affect communication latency, our investigation shows that contention — not hop inflation — is the dominant source of communication latency under wafer-scale defects, which directly motivates ConBIN's contention-aware optimization for performance convergence and improvement.

Hop-based contention (i.e., no inter-task interference) in Fig.9(a) shows limited latency growth, which rapidly saturates beyond 5-6 hops due to router pipeline amortization. In contrast, Fig.9(b)-(c) illustrate inter-task contention, where destination, link, and hybrid contention cause severe and near-linear latency escalation as the number of competing

tasks increases. These results indicate that contention-induced slowdown far exceeds hop-induced latency, making contention the primary latency source under clustered faults.

We therefore use the metrics MDQC, MLCC, MCC, as defined in Fig.9(c), to guide ConBIN's mapping and scheduling in order to reduce fault-induced contention and converge chip performance toward premium bins.

#### B. Pre-Binning

Although the hardware-level design (Sec.V) restores a nearmesh topology, residual irregularities still cause non-trivial performance loss and inter-chip variation, motivating software-level optimization. To support this, ConBIN introduces a prebinning stage that partitions repaired chips into preliminary bins and establishes bin-specific optimization objectives for mapping and scheduling. Given a chip population and a vendor-defined bin count B, this stage derives representative targets with acceptable computational cost.

**Initial Partitioning.** Chips are sorted by hardware-level metric F (Sec.V-C) and divided into B+1 quantile-based groups, providing an initial estimate of target performance bins before optimization.

Target Generation via Lightweight Sampling. To assign optimization targets efficiently, ConBIN conducts lightweight mapping (Sec.VI-C) and scheduling (Sec.VI-D) on a small subset (top 5%–15%) of chips within each group, using reduced populations and limited generations. The average performance of these sampled chips defines each bin's representative objectives: the expected maximum link contention count  $MLCC_{exp}^{target}$  for mapping (Sec.VI-C) and the maximum contention cost  $\Phi^{target}$  for scheduling (Sec.VI-D). These targets ensure bin-wise alignment during subsequent optimization, enabling effective performance recovery and more profitable binning.

# VI. SOFTWARE-LEVEL OPTIMIZATION AND PERFORMANCE BINNING

#### A. Communication Performance Impact of Contention

Manufacturing faults reshape routing paths and reduce available communication resources, causing both longer paths and significantly higher contention. While both factors affect communication latency, our investigation shows that contention — not hop inflation — is the dominant source of communication latency under wafer-scale defects, which directly motivates ConBIN's contention-aware optimization for performance convergence and improvement.

Hop-based contention (i.e., no inter-task interference) in Fig.9(a) shows limited latency growth, which rapidly saturates beyond 5-6 hops due to router pipeline amortization. In contrast, Fig.9(b)-(c) illustrate inter-task contention, where destination, link, and hybrid contention cause severe and near-linear latency escalation as the number of competing

tasks increases. These results indicate that contention-induced slowdown far exceeds hop-induced latency, making contention the primary latency source under clustered faults.

We therefore use the metrics MDQC, MLCC, MCC, as defined in Fig.9(c), to guide ConBIN's mapping and scheduling in order to reduce fault-induced contention and converge chip performance toward premium bins.

#### B. Pre-Binning

Although the hardware-level design (Sec.V) restores a nearmesh topology, residual irregularities still cause non-trivial performance loss and inter-chip variation, motivating software-level optimization. To support this, ConBIN introduces a prebinning stage that partitions repaired chips into preliminary bins and establishes bin-specific optimization objectives for mapping and scheduling. Given a chip population and a vendor-defined bin count B, this stage derives representative targets with acceptable computational cost.

**Initial Partitioning.** Chips are sorted by hardware-level metric F (Sec.V-C) and divided into B+1 quantile-based groups, providing an initial estimate of target performance bins before optimization.

Target Generation via Lightweight Sampling. To assign optimization targets efficiently, ConBIN conducts lightweight mapping (Sec.VI-C) and scheduling (Sec.VI-D) on a small subset (top 5%–15%) of chips within each group, using reduced populations and limited generations. The average performance of these sampled chips defines each bin's representative objectives: the expected maximum link contention count  $MLCC_{exp}^{target}$  for mapping (Sec.VI-C) and the maximum contention cost  $\Phi^{target}$  for scheduling (Sec.VI-D). These targets ensure bin-wise alignment during subsequent optimization, enabling effective performance recovery and more profitable binning.

