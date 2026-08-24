# <span id="page-4-0"></span>4.3 Bridging the Gap: Length-based Step Reward

As shown in Eq. 2 and visualized in Table 2, a key limitation of the truncation method is that it assigns the same penalties to overlong responses as it does to incorrect ones, which may over-penalize long but correct explorations. To address this issue, we extend it as a novel reward shaping approach called

Length-bAsed StEp Reward (LASER), which adopts a step reward function guided by a target length, rather than performing hard truncation.

Specifically, we design the length reward term S(y) as an indicator function based on a target length L<sup>T</sup> . This function assigns a length-based bonus to responses shorter than L<sup>T</sup> . We also set the context window significantly larger than the target length L<sup>T</sup> (e.g., 16,384 vs. 4,096) where truncation rarely happens. And the length reward term S(y) is only activated when responses are correct, thereby improving the efficacy-efficiency trade-off. As visualized in Table [2,](#page-4-1) LASER closely resembles the vanilla truncation approach; the only difference is that, instead of truncating long responses, it awards bonus rewards to correct responses that do not exceed the target length. To balance the correctness reward C(y) and length reward S(y), we follow a typical setting and set α as 0.5.

Empirical results are demonstrated in Figure [1](#page-1-0) and Table [3,](#page-7-1) training with the LASER reward achieves improved Pareto-optimality compared to all previous methods. Notably, it is the first approach to simultaneously deliver significant improvements in both accuracy and token efficiency on the challenging AIME24 benchmark. These results establish LASER as a promising reward design framework for enhancing the balance between efficacy and efficiency.

