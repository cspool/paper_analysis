# 6 Conclusion

This paper addresses the challenges of training large Mixture-of-Experts models, under the constraints of limited GPU memory. By offloading expert parameters and optimizer states to the host, ES-MoE supports scaling MoE models without additional GPUs. Its dynamic expert placement ensures that the load is spread uniformly across GPUs without introducing zero-padding, solving the straggler problem and further saving the memory usage.

Our extensive evaluation demonstrates ES-MoE's superior scalability and throughput. It successfully accommodates up to 67× more experts than conventional methods and achieves remarkable throughput improvements. ES-MoE outperforms existing offloading frameworks by up to 17.5× and shows up to 2.13× gains over Tutel.

### Acknowledgements

We thank the anonymous reviewers for providing helpful feedback and suggestions to improve our work. This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00398157) and Samsung Electronics.

### Impact Statement

This work prioritizes batch-level parallelism over expertlevel parallelism and leverages CPU offload to achieve scalable training of large MoE-based models. We believe this work will empower researchers from academia and small organizations with the ability to train MoE-based LLMs with a larger number of experts.

