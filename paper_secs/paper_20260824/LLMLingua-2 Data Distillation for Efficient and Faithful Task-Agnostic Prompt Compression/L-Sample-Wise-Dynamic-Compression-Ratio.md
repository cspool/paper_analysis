# L Sample-Wise Dynamic Compression Ratio

By default, *LLMLingua-2* applies fixed compression rate to all samples in the benchmark. However, this approach may not be optimal due to variations in the density of key information across different samples. To address this problem, we allow *LLMLingua-2* to dynamically adjust the compression rate for each sample under the overall compression rate constraint. Specifically, we employ the compressor to predict each token's preservation probability of all samples. We then set a probability threshold to achieve the overall compression rate constraint. For all samples, tokens with preservation probabilities higher than this threshold are retained.

Table [12](#page-17-1) presents the performance of *LLMLingua-2* using the sample-wise dynamic compression ratio, showcasing a 4.4% and 4.5% performance improvement under 7x and 5x compression ratios, respectively, compared to

*LLMLingua-2* with a fixed compression ratio.

