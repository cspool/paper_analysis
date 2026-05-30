# **D** Training Curve

To better understand the behavior of VideoAuto-R1, we visualize the training curves of the task rewards for both the initial and reviewed answers during training, as shown in Figure 6. We highlight three key observations below.

Reviewed Answer vs. Initial Answer. For both Qwen2.5-VL-7B and Qwen3-VL-8B, the reviewed answer consistently achieves a higher task reward than the initial answer during training. This performance gap remains stable after convergence, indicating that the answer-think-answer paradigm effectively leverages intermediate reasoning to refine predictions. Moreover, this confirms that the dual-answer reward design (with  $w_2 > w_1$ ) can encourage the model to treat the second answer as a meaningful revision rather than a naive re-sampling of the first.

**Training Dynamics.** As training progresses, the task rewards for both answers increase. In the early stages, we observe a rapid improvement, followed by a slower but steady rise until convergence. This pattern suggests that GRPO quickly captures coarse task structure and gradually optimizes finer-grained reasoning capabilities over time.

Impact of Backbone Capacity. Throughout training, Qwen3-VL-8B consistently outperforms Qwen2.5-VL-7B in both answers. The stronger backbone benefits from better initialization and sustains a higher reward margin after convergence. These results demonstrate that VideoAuto-R1 scales effectively with model capacity: larger base models can more fully exploit dual-answer supervision and confidence-based reasoning, resulting in higher final results.

