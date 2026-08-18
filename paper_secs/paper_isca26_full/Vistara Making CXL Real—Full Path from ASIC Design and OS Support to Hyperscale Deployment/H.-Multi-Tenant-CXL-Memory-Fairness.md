# *H. Multi-Tenant CXL Memory Fairness*

The preceding evaluations focus on single-tenant servers, which represent the dominant production mode for our largest services. However, a fraction of our fleet stacks multiple containers on a single host to maximize utilization. In a tiered-memory system, na¨ıve page placement can cause one container to monopolize local DRAM while pushing a colocated neighbor disproportionately onto CXL, creating unfair latency imbalances and vulnerability to noisy-neighbor effects.

To evaluate this, we stack two CacheA containers on a single MemServer, each serving ≈2.5 M QPS with ≈430 GB of cache memory. With the baseline TPP policy, memory allocation across tiers is highly unbalanced: Container A retains 90% of its working set in local DRAM while Container B retains only 70%, despite identical loads. This imbalance leads to a 20% deviation in P99 latency between the two containers. Worse, when Container B experiences a sudden load spike (noisy neighbor), Container A's QPS drops by 65% and it is eventually OOM-killed due to excessive demotion pressure.

To address this, we introduce per-container memory-tier accounting (Fair Share). The algorithm enforces configurable per-container bounds (locallow, localhigh) on local DRAM usage; when a container exceeds localhigh, pages are demoted to CXL. As Table IX shows, Fair Share maintains a balanced ≈3:1 local:CXL split for both containers (74% vs. 71% local), improving P99 latency by 1.6× and 1.8× for Containers A and B, respectively. Under the same noisyneighbor scenario, Fair Share limits Container A's worst-case QPS dip to 12% (recovering within seconds) and eliminates the OOM risk entirely.

TABLE IX MULTI-TENANT CXL FAIRNESS: TWO CO-LOCATED CACHEA CONTAINERS UNDER BASELINE (TPP) AND FAIR-SHARE.

|                      | Baseline |       | Fair-Share |       |  |
|----------------------|----------|-------|------------|-------|--|
| Metric               | Ctr A    | Ctr B | Ctr A      | Ctr B |  |
| Local Memory (GB)    | 407      | 315   | 343        | 330   |  |
| CXL Memory (GB)      | 47       | 137   | 120        | 132   |  |
| Local Fraction (%)   | 90       | 70    | 74         | 71    |  |
| P99 Latency (µs)     | 283      | 327   | 176        | 185   |  |
| QPS Drop (noisy nbr) | 65%      | —     | 12%        | —     |  |

