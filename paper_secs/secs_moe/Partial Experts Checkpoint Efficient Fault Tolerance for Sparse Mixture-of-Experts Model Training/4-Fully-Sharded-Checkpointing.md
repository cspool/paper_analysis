# 4 Fully Sharded Checkpointing

As discussed in Section 2.3.2, existing work lacks an efficient data-parallel sharding strategy for checkpointing MoE models in distributed training. Figure 7(a) demonstrates that the baseline method provided by the Megatron-DeepSpeed framework [40] only utilizes "Rank0" to save non-expert states and "EP-Group-0" to save expert states.

