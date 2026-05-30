# E Efficiency Evaluation in Offloading Scenarios

Notably, in memory-offloading scenarios where the per-token loading cost dominates, Twilight could achieve more significant gains. This is because Twilight reduces the number of loaded tokens with a fixed estimation cost. [Table 7](#page-20-2) shows Twilight could achieve up to 16× speedups compared to Quest.

<span id="page-20-2"></span>Table 7: Latency (in microseconds) of a single attention operator in offloading scenarios, where corresponding tokens in the KV cache are loaded from the CPU memory.

|           | 10k     | 20k     | 30k     |
|-----------|---------|---------|---------|
| Quest     | 3038.98 | 5990.75 | 8490.95 |
| Quest-Twi | 415.89  | 480.61  | 527.77  |

