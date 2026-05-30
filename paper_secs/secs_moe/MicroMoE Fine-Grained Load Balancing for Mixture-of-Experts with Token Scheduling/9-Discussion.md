# 9 Discussion

Scalability. Scaling FineMoE to large clusters and increased expert counts introduces new challenges to the system. When scaling up, FineMoE benefits from an expanded scheduling space, which enhances the load balancing capability of FineEP. However, this larger scheduling space increases computational overhead in scheduling. To compromise between load balancing capability and system efficiency, we can organize GPUs and experts into groups and perform scheduling at the group level, similar to prior works [\[9,](#page-13-1) [25,](#page-14-4) [63\]](#page-16-1).

FSDP. We currently implement FineMoE based on Megatron-LM's DDP [\[50\]](#page-15-13). Megatron-LM also supports Fully Sharded Data Parallel (FSDP) [\[65\]](#page-16-3), which resembles DeepSpeed's ZeRO-3 [\[45\]](#page-15-14), sharding model parameters and gradients for memory saving. We plan to integrate FineMoE with FSDP in future work.

