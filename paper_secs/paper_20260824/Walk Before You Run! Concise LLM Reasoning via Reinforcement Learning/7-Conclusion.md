# 7 Conclusion

In this paper, we propose ConciseR, which introduces a simple yet effective two-stage reinforcement learning framework. First, it incentivizes the model's reasoning capabilities via GRPO++, and then it reduces the model's response length to improve the quality of the CoT response implicitly via L-GRPO. *Importantly, we innovatively propose that during training, response length optimization is only triggered when all rollouts for a given training sample are correct. This embodies the "walk before you run" principle.* Experiments demonstrate that ConciseR consistently achieves the best efficiency-accuracy synergistic improvement, significantly outperforming existing efficient reasoning methods across five benchmarks.

