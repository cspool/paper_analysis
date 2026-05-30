# <span id="page-16-1"></span>H DETAILED SCALING RESULTS

To further examine the scalability of MoM from the perspective of memory capacity, we evaluate the effect of enlarging the memory pool beyond the main settings. Specifically, we compare two activation ratios, where the number of active memories accounts for either 0.5 or 0.25 of the total memory states. In all cases, an additional shared memory is included. Starting from a single memory as the baseline, we expand the number of memories to 2, 4, 8, and 16, and report the averaged results on both recall-intensive and commonsense benchmarks. The results, shown in Fig. [8,](#page-16-3) indicate that under a fixed activation ratio, increasing the memory size consistently improves performance on both categories of tasks.

<span id="page-16-3"></span>![](_page_16_Figure_7.jpeg)

Figure 8: Detailed scaling results. Performance shows a general improvement on both recallintensive and commonsense tasks as the number of memories increases.

