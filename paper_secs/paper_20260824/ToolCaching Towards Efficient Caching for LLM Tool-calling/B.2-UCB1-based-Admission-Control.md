# B.2 UCB1-based Admission Control

We adopt UCB1 algorithm [\[12\]](#page-8-28) to address the multi-armed bandit problem in cache admission. At each decision round, we calculate the reward for each group according to [3,](#page-5-1) and select the group with the highest UCB value:

<span id="page-10-0"></span>
$$UCB_i = F_i + c\sqrt{\frac{\ln t}{N_i}} \tag{6}$$

where is the number of times group has been selected, is the current round, and is an exploration parameter.

The psudo-code of UCB1-based admission control is shown in [Algorithm 2.](#page-9-3)

### C Discussion on Limitations and Future Work

