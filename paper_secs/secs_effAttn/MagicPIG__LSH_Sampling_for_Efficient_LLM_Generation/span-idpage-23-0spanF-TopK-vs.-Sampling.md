# <span id="page-23-0"></span>F TopK vs. Sampling

In this section, we provide an intuitive understanding of how sampling can work better than TopK. TopK only captures the ranking information when estimating attention output. In contrast, sampling considers the entire data distribution (i.e., the attention score after Softmax).

Here is an example. Imagine a zoo with 100 animals: 10 elephants, 10 pigs, 10 tigers, and 70 other unique animals. The daily food consumption for each group is as follows:

• Elephants: 50 lb/day each

• Pigs: 20 lb/day each

• Tigers: 10 lb/day each

• Other unique animals: 1 lb/day each

To compute the true average daily food consumption per animal in the zoo:

True Average = 
$$\frac{(10 \times 50) + (10 \times 20) + (10 \times 10) + (70 \times 1)}{100} = 8.7 \text{ lb.}$$

If we use a Top-K approach (e.g., selecting the top 10 animals based on the numbers of animals), we include elephants, pigs, tigers, and 7 randomly selected animals from the unique ones. The estimated average is:

TopK Average = 
$$\frac{(10 \times 50) + (10 \times 20) + (10 \times 10) + (7 \times 1)}{37} = 22 \text{ lb.}$$

This overestimates the average because it disproportionately weights high-consumption animals.

Instead, we perform sampling with replacement from the animal distribution, proportional to their numbers. The probabilities for each group are:

Sampling Probabilities = 
$$[0.1, 0.1, 0.1, 0.01 \times 70]$$
,

where 0.1 represents the probabilities for elephants, pigs, and tigers (10/100 each), and 0.01 corresponds to each unique animal (1/100).

Perform 10 random draws. A possible sampling outcome could be: [elephant, pig, tiger, other, other, other, other, other, other, other]. The corresponding daily food estimate is:

Sample Estimate = 
$$\frac{50 + 20 + 10 + (7 \times 1)}{10} = 8.7 \,\text{lb.}$$

This estimate is unbiased, meaning the expected value of the estimates equals the true average (8.7 lb). While there is variance across individual trials, the standard deviation (std) can be calculated as 4.7 lb for a 10-sample budget.

Increasing the sampling budget reduces variance. For example, with 20 samples, the std decreases to 3.4 lb. Meanwhile, Top-K with a budget of 20 adds 17 unique animals, yielding:

TopK Average (K=20) = 
$$\frac{(10 \times 50) + (10 \times 20) + (10 \times 10) + (17 \times 1)}{47} = 17 \text{ lb.}$$

Again, the Top-K estimate remains biased, significantly overestimating the average.

Note that this is intended as an intuitive example. For a detailed and formal derivation of the sampling methodology, please refer to [Kloek and Van Dijk](#page-14-10) [\(1978\)](#page-14-10); [Owen](#page-14-8) [\(2013\)](#page-14-8); [Lohr](#page-14-9) [\(2021\)](#page-14-9).

