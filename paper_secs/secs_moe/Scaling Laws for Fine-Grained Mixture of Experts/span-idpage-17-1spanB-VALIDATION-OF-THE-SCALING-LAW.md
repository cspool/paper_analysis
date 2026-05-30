# <span id="page-17-1"></span>B VALIDATION OF THE SCALING LAW

<span id="page-17-0"></span>In this section, we provide coefficients of the scaling law fitted with 20% of datapoints with the lowest perplexity excluded for the purpose of validation.

Table 5: Values of the fitted coefficients.

| Model | a    | α     | b    | β     | g    | γ     | c     |
|-------|------|-------|------|-------|------|-------|-------|
| MoE   | 17.6 | 0.114 | 26.7 | 0.140 | 2.07 | 0.570 | 0.472 |

### <span id="page-17-2"></span>C RELIABILITY OF COMPUTE OPTIMAL FORMULA

<span id="page-17-4"></span>In this section, we assess the stability of our predictions presented in Section [6.1.](#page-8-1) Similarly to [Hoffmann et al.](#page-12-8) [\(2022\)](#page-12-8) we calculate the 10th and 90th percentiles estimated via bootstrapping data (80% of the data is sampled 100 times). See Table [6](#page-17-4) for the details.

Table 6: 10th and 90th percentiles estimated via bootstraping data.

| N         | D                  | G        |
|-----------|--------------------|----------|
| 64 x 100M | (2.97B, 5.98B)     | (8, 8)   |
| 64 x 1B   | (21.17B, 40.73B)   | (16, 16) |
| 64 x 3B   | (50.20B, 105.88B)  | (16, 32) |
| 64 x 7B   | (101.06B, 205.40B) | (32, 32) |
| 64 x 70B  | (638.49B, 1.59T)   | (32, 64) |
| 64 x 300B | (1.99T, 5.62T)     | (64, 64) |
| 64 x 1T   | (5.29T, 16.87T)    | (64, 64) |

