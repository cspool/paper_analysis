# *D. GPU Memory Consistency*

As outlined in previous studies [30], the NVIDIA GPU memory model defines rules governing the perceived ordering of GPU memory operations and the possible values returned by read operations. The most pertinent concepts are the distinction between weak and strong accesses, as well as the scope associated with strong accesses. In essence, sysscoped memory operations or fences are employed to explicitly signal synchronization across GPUs. In contrast, other types of accesses are not required to be visible to, or ordered with, memory operations from different GPUs unless synchronization is enforced through sys-scoped operations.

# *D. GPU Memory Consistency*

As outlined in previous studies [30], the NVIDIA GPU memory model defines rules governing the perceived ordering of GPU memory operations and the possible values returned by read operations. The most pertinent concepts are the distinction between weak and strong accesses, as well as the scope associated with strong accesses. In essence, sysscoped memory operations or fences are employed to explicitly signal synchronization across GPUs. In contrast, other types of accesses are not required to be visible to, or ordered with, memory operations from different GPUs unless synchronization is enforced through sys-scoped operations.

