# VII. RELATED WORK

Due to the surge in AI workloads, the trend of developing in-house AI chips has gained significant momentum among major IT companies, including Google [17], Amazon [11], Huawei [22], IBM [23], Microsoft [1], Alibaba [15], Baidu [28], OpenAI [27], Tesla [25], Tencent [19], and Meta [7], [10]. Additionally, many startups, such as Cerebras [24], Groq [3], SambaNova [30], Cambricon [4], and Rivos [31], offer AI chips as alternatives to GPUs from established vendors [6], [18], [33].

Compared with Meta's prior inference chips [7], [10], MTIA 300—our first training chip—introduces three distinguishing features: (1) built-in NIC chiplets, (2) dedicated message engines for collective offloading, and (3) nearmemory compute to accelerate reduction-based collectives. To our knowledge, these features are unique among existing AI accelerators.

Although Google TPU's sparse core [16] can offload communication, it is specialized for a non-RDMA, non-switched torus network and lacks a general collective library interface. This limits the applicability of the sparse core's technology to other industry accelerators, which are typically built around RDMA and similar collective library interfaces.

Most custom AI ASICs focus on inference in graph mode and lack first-class support for eager mode, which is important for PyTorch's usability by model developers. In contrast, MTIA provides a native, PyTorch-first software ecosystem supporting TorchDynamo, TorchInductor, Triton, and both eager and graph modes, with a CUDA-like runtime API that simplifies model porting. Notably, while both MTIA and other chips [30] adopt a dataflow architecture in which computations occur as their dependencies are resolved, MTIA is the only one to provide a native PyTorch experience. Finally, whereas modern AI ASICs are primarily optimized for GenAI models, MTIA 300 is optimized for DLRMs.

## VIII. CONCLUSION

We presented the design and evaluation of MTIA 300, our first in-house AI chip for DLRM training. With its first-ofits-kind integrated NICs and collective offloading engines, we demonstrate improved efficiency on a recommendation model training workload compared with GPUs. For future work, we are accelerating the development of Meta's next-generation AI chips [34], which will address the growing demands of LLM training and inference workloads alongside recommendation tasks. We expect that the shift toward in-house AI chips will drive new model co-design opportunities and the next wave of model innovations.

