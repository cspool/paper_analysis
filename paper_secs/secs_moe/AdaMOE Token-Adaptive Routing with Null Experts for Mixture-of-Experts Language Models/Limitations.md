# Limitations

One potential drawback of this work is that we did not pre-train a MoE-LLM using our A*da*MO<sup>E</sup> method. Pre-training an MoE-LLM would have allowed us to thoroughly evaluate the full capabilities and performance improvements of our method, but the significant resources required made it impractical for our current study. Additionally, we did not explore the scenario of null experts as identity mappings, where null experts would also need zero FLOPs to process input tokens. We hypothesize that this approach might accelerate training convergence because null experts as identity mappings would potentially update their corresponding router parameters more frequently.

We acknowledge these limitations and leave these aspects for future work. Addressing these issues could provide a more comprehensive evaluation of the A*da*MO<sup>E</sup> method and potentially uncover additional benefits or areas for improvement.

