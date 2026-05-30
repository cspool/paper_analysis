# 5 *FineMoE*'s Implementation

We prototype *FineMoE* on top of Huggingface Transformers framework [\[55\]](#page-15-10) using MoE-Infinity codebase [\[59\]](#page-15-23). The implementation of *FineMoE* is described as follows.

Expert Map Store is implemented in Python using Py-Torch [\[42\]](#page-15-25) and NumPy [\[19\]](#page-14-22) libraries. We store all semantic embeddings and expert maps using ndarrays data structure for efficient array operations. The arrays are converted to tensors to compute similarity for expert map searching.

Expert Map Searcher is implemented in Python using PyTorch [\[42\]](#page-15-25). We implement the pairwise computations, including similarity ([§4.2\)](#page-6-0) and redundancy (§ [4.4\)](#page-8-1) scores, using PyTorch native operations.

Expert Cache is implemented in C++ based on MoE-Infinity codebase [\[59\]](#page-15-23). The expert management in GPUs is implemented with the CUDA Runtime APIs [\[40\]](#page-15-26). We implement prefetching and caching logic of *FineMoE* in the MoE-Infinity codebase to enable expert offloading. Same with MoE-Infinity, *FineMoE* supports multi-GPU inference with expert parallelism (EP), where the experts are mapped to different GPU devices for loading and offloading. We use a hash map to assign expert IDs to different GPUs and retrieve them during inference. The expert assignment follows a round-robin manner to balance the overall GPU load. Additionally, we use a task pool in the GPU space with asynchronous threads to schedule and execute expert prefetching and on-demand loading tasks.

