# 6 Conclusion

In this paper, we introduced MoE Jetpack, a novel framework for fine-tuning pre-trained dense checkpoints into Mixture of Experts. Our approach leverages checkpoint recycling, which inherits the knowledge of open-source dense checkpoints and the hyperspherical adaptive MoE (SpheroMoE) layer to enhance fine-tuning performance. These innovations contribute to improved convergence speed and model accuracy. The MoE Jetpack significantly improved various visual tasks while maintaining computational efficiency.

The limitation of our approach is its dependency on the quality of pre-trained dense checkpoints; poorly trained or inadequately generalized dense models could limit the performance enhancements. Additionally, while our experiments focused on visual tasks, further research is needed to validate the generalizability of MoE Jetpack across other domains, such as natural language processing and reinforcement learning. We believe future work will address these limitations, enhance the scalability and robustness of the framework, and extend MoE applicability to a broader range of tasks.

