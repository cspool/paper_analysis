# 5 Conclusions

We introduce compressed expert, a lightweight and efficient approach to reducing the number of activated experts in MoE models while maintaining strong performance. By replacing auxiliary experts with compact expert representations, our method significantly reduces computational overhead while preserving model capacity. Extensive experiments on Phi-MoE and OLMoE across various tasks demonstrate that compressed experts reduce active parameters by over 30% and cut inference costs by 20% while retaining over 90% of the performance of full-expert configurations. Beyond reducing inference costs, our findings suggest broader implications for scaling MoE models efficiently. Compressed experts offer a promising direction for optimizing sparse activation in MoE architectures.

