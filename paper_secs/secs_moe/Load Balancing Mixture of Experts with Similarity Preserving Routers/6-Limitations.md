# 6 Limitations

While we train our models with relatively large data multipliers, prior work such as [Muennighoff](#page-11-1) [et al.](#page-11-1) [\[2025\]](#page-11-1) suggests that substantially more data (trillions of tokens) may be necessary to achieve strong performance on downstream benchmarks. Nevertheless, our training setup provides sufficient scale to meaningfully compare the relative effectiveness of different balancing methods, which we supplement with statistical significance comparisons.

Finally, although our architectural choices align with recent MoE literature, our study is limited to a single set of design decisions. We leave the exploration of alternative configurations to future work. For instance, we do not investigate how token dropping might affect the performance of our balancing mechanism (instead focusing on higher-quality dropless models [\[Gale et al., 2022\]](#page-10-11)), which could be a valuable direction for further analysis.

