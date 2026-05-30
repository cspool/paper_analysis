# <span id="page-17-3"></span>D VARYING EXPANSION RATE

<span id="page-17-5"></span>In this section, we provide results for E = 16. The training procedure is the same as described in App. [A.](#page-16-0) The models considered in this part are listed in Table [7.](#page-17-5)

Table 7: Architecture and training variants (MoE models).

| #parameters (nonemb) | dmodel | nblocks | nheads | D (in #tokens) | G              |
|----------------------|--------|---------|--------|----------------|----------------|
| 64x3M                | 256    | 4       | 4      | 8B, 16B, 33B   | 1, 2, 4, 8, 16 |
| 64x7M                | 256    | 8       | 4      | 8B, 16B, 33B   | 1, 2, 4, 8, 16 |
| 64x13M               | 512    | 4       | 8      | 8B, 16B, 33B   | 1, 2, 4, 8, 16 |
| 64x13M               | 512    | 4       | 8      | 66B            | 1, 2, 4        |
| 64x25M               | 512    | 8       | 8      | 8B, 16B, 33B   | 1, 2, 4, 8, 16 |
| 64x49M               | 640    | 10      | 10     | 8B             | 1, 2, 4, 8, 16 |

<span id="page-17-6"></span>We fit Eq. [9](#page-6-1) using the same procedure as described in Section [5.4.](#page-7-1) The results are detailed in Table [8.](#page-17-6)

Table 8: Values of the fitted coefficients.

| Model        | a     | α     | b     | β     | g    | γ     | c     |
|--------------|-------|-------|-------|-------|------|-------|-------|
| MoE (E = 16) | 19.64 | 0.124 | 57.07 | 0.169 | 1.18 | 0.986 | 0.472 |

<span id="page-18-1"></span>Using the coefficients and FLOPs calculation formulas, we can derive the compute optimal training parameters. The results are presented in Table [9.](#page-18-1)

Table 9: 10th and 90th percentiles estimated via bootstrapping data for E = 16.

| N         | D                  | G         |
|-----------|--------------------|-----------|
| 16 x 100M | (10.29B, 17.73B)   | (8 , 16)  |
| 16 x 1B   | (53.74B, 103.54B)  | (16, 32)  |
| 16 x 3B   | (106.22B, 261.04B) | (16, 32)  |
| 16 x 7B   | (177.65B, 511.43B) | (16, 32)  |
| 16 x 70B  | (721.60B, 3.22T)   | (32, 64)  |
| 16 x 300B | (1.73T, 10.69T)    | (32, 64)  |
| 16 x 1T   | (3.60T, 28.22T)    | (32, 128) |

We can observe that similarly to the case when E = 64, larger compute budgets imply larger optimal values of G. Note that the values for 10th and 90th percentiles form larger intervals in this case, as in this part we run a smaller number of experiments and keep shorter training durations. However, we believe that this preliminary study forms a valuable addition to the results in the main part.

