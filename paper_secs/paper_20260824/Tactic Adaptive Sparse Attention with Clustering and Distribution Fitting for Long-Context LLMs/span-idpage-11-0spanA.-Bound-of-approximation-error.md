# <span id="page-11-0"></span>A. Bound of approximation error

In Sec. 3.2, the upper bound of  $\epsilon(I)$  can be derived as

$$\epsilon(I) = \|o - \tilde{o}(I)\| \tag{7}$$

$$= \|o - \frac{1}{p(I)} \sum_{i \in I} s_i v_i\|$$
 (8)

$$= \| \sum_{i \in I} s_i v_i + \sum_{i \notin I} s_i v_i - \frac{1}{p(I)} \sum_{i \in I} s_i v_i \|$$
 (9)

$$= \| \left( 1 - \frac{1}{p(I)} \right) \sum_{i \in I} s_i v_i + \sum_{i \notin I} s_i v_i \|$$
 (10)

$$\leq \| \left( 1 - \frac{1}{p(I)} \right) \sum_{i \in I} s_i v_i \| + \| \sum_{i \notin I} s_i v_i \|$$
 (11)

$$\leq \left| \left( 1 - \frac{1}{p(I)} \right) \right| \sum_{i \in I} s_i \|v_i\| + \sum_{i \notin I} s_i \|v_i\|$$
 (12)

$$\leq \left(\frac{1}{p(I)} - 1\right) p(I) \max_{i} \|v_i\| + (1 - p(I)) \max_{i} \|v_i\|.$$
(13)

Hence we can get

$$\epsilon(I) \le 2(1 - p(I)) \max_{i} ||v_i||.$$
 (14)

Both (10) to (11) and (11) to (12) are based on triangle inequality.

