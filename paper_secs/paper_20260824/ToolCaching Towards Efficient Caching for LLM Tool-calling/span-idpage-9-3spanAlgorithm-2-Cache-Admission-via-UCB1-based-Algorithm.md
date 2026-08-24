# <span id="page-9-3"></span>Algorithm 2 Cache Admission via UCB1-based Algorithm

**Input:** Leaf feature groups  $G = \{g_1, g_2, ..., g_n\}$  from hierarchical partitioning; current statistics for each  $g_i$ 

**Output:** Admission decisions for each decision round 1: **procedure:** UCB1-Admission(G)

Update  $C_{i^*} \leftarrow C_{i^*} + 1$ ,  $N_{i^*} \leftarrow N_{i^*} + 1$ 

10: end procedure

```
for each decision round t do
for all feature group g<sub>i</sub> ∈ G do
Update statistics for g<sub>i</sub>: hit ratio H<sub>i</sub>, node level L<sub>i</sub>, admission count C<sub>i</sub>, avg. caching value V<sub>i</sub>, selection count N<sub>i</sub>
Compute reward F<sub>i</sub> using Eq. (3)
Compute UCB<sub>i</sub> = F<sub>i</sub> + c√(\frac{\ln t}{N_i}\) using Eq. (6)
Select g<sub>i*</sub> = arg max<sub>gi</sub> UCB<sub>i</sub>
Admit requests from g<sub>i*</sub> into the cache
```

