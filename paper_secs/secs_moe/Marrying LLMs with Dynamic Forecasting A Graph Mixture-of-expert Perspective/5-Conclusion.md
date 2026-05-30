# 5 Conclusion

In this paper, we study the problem of dynamical system modeling under environmental changes and propose a new approach LEGO which connects LLM-as-a-judge with the mixture-of-expert framework. Our LEGO first extracts hierarchical prompts from three views to infer environmental information. Then, they are fed into LLMs as a routing function to determine which experts are most relevant to different environments. The framework is optimized by alternating the updates of the routing weights and expert parameters to achieve robust performance. Extensive experiments across various benchmark datasets demonstrate the effectiveness of LEGO compared to numerous baseline methods.

