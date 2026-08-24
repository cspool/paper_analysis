# <span id="page-14-0"></span>D Supplementary Details: Length-based Reward Shaping Formulations

In this section, we provide additional details regarding the various formulations of length-based reward shaping as presented in Table 2. These formulations can viewed as different variants of our unified framework in Eq. 2 which can be implemented by making specific design choices for three key components: C(y),  $\lambda(y)$ , and S(y) inside the framework. Here we review the formulation of Eq. 2 to better illustrate following approaches.

$$\hat{R}(x,y) = C(y) + \lambda(y) \cdot S(y)$$

#### **D.1** Truncation

**Vanilla Truncation** As aforementioned discussions (§4), truncation is a special case of the length reward with C(y) = 0, where the target length  $L_T$  is enforced by the context window.  $\rho$  is set as 0. It follows the design:

$$\begin{split} &C(y) = 0 \\ &\lambda(y) = 1 \\ &S(y) = \begin{cases} R(x,y) & \text{if } L(y) \leq L_T \\ \rho & \text{if } L(y) > L_T \end{cases} \end{split}$$

**ThinkPrune** ThinkPrune [10] is another truncation-based approach, which extends vanilla truncation by introducing an adaptive target lengths  $L_A$  to replace fixed target lengths  $L_T$ .  $\rho$  is set as 0. The design follows:

$$\begin{split} &C(y) = 0 \\ &\lambda(y) = 1 \\ &S(y) = \begin{cases} R(x,y) & \text{if } L(y) \leq L_A \\ \rho & \text{if } L(y) > L_A \end{cases} \end{split}$$

Their training methodology employs a progressive three-stage process with iterative refinement of  $L_A$ . Each subsequent stage initializes from the checkpoint of the previous stage while mannually

reducing the value of  $L_A$ . Specifically, they progressively decrease  $L_A$  through values of 4096, 3072, and 2048 across the three stages.

#### D.2 Group-based Rewards

In the context of group-based rewards, the length reward S(y) is specifically designed to promote brevity by assigning higher scores to shorter responses within a rollout group. This mechanism functions as a comparison-based reward system that inherently favors more concise responses. Most of them follow the design C(y) = R(x,y) to keep the accuracy performance of models.

**Efficient Reasoning** Efficient Reasoning [2] follows the principle of group-based reward by specifically encouraging conciseness within correct responses. The mean and variance scalars are computed exclusively from the subset of correct responses, ensuring appropriate statistical distributions. By selectively rewarding conciseness only when answers are correct, this approach maintains higher accuracy compared to Kimi-k1.5 [11], which encourages wrong responses to be shorter. Considering the similarity between the two approaches and the better efficacy-efficiency trade-off, we select Efficient Reasoning as the representative group-based reward in this paper. The corresponding design can be formulated as follows:

$$\begin{split} &C(y) = R(x,y) \\ &\lambda(y) = \mathbb{I}(R) \\ &S(y) = -\alpha \cdot \sigma \left( \frac{L(y) - Mean(y)}{STD(L)} \right) \end{split}$$

**Kimi-k1.5** The design of Kimi-k1.5 is similar to Efficient Reasoning [2], with two main differences. First, the scalar factors are computed using the minimum response length and the difference between maximum response length and maximum length within a rollout group. Second, Kimi-k1.5 encourages all responses to be shorter, rather than focusing solely on shortening correct responses. Such a design has the potential to intensify reward hacking, as models may exploit the reward function by favoring shorter responses to maximize their scores. The designs follows:

$$\begin{split} C(y) &= R(x,y) \\ \lambda(y) &= 1 \\ S(y) &= \begin{cases} 0.5 - \frac{L(y) - L_{\min}}{L_{\max} - L_{\min}} & \text{if } \mathbb{I}(R) = 1 \\ \min\left(0, \ 0.5 - \frac{L(y) - L_{\min}}{L_{\max} - L_{\min}}\right) & \text{if } \mathbb{I}(R) = 0 \end{cases} \end{split}$$

#### D.3 Budget-based Reward

Budget-based rewards use query-specific target lengths (budgets) and penalize responses that deviate from these instructions. And the coefficient  $\alpha$  controls the trade-off between length reward and correctness reward. They come in two flavors: exact mode and max mode. We follow same settings as L1 [1] and set  $\alpha = 0.0003$  for exact mode,  $\alpha = 0.01$  for max mode.

**Exact Mode** In exact mode, the model must hit the specified target length  $L_T$  exactly, and any deviation (even shorter outputs) is penalized. The design can be formulated as:

$$C(y) = R(x, y)$$

$$\lambda(y) = 1$$

$$S(y) = -\alpha \cdot |L(y) - L_T|$$

**Max Mode** In max mode, only outputs that exceed  $L_T$  incur a penalty. The designs follow:

$$C(y) = 0$$

$$\lambda(y) = \mathbb{I}(R)$$

$$S(y) = \text{clip}(\alpha \cdot (L(y) - L_T) + \delta, 0, 1)$$

