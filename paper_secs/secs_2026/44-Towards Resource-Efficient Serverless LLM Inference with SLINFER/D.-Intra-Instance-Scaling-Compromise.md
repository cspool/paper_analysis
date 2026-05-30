# *D. Intra-Instance Scaling Compromise*

The orchestration mechanism may reject a scale-up demand if there is not enough memory (recall Figure 19). When trying to add a new request to a instance, SLINFER also performs a shadow check on whether the potential scale-up demand can be approved. To fully utilize all available memory, if the shadow check fails, SLINFER will attempt to compromise the scale-up demand, allowing the request to be accepted as long as it can scale up to Mrequire rather than Mrecommend.

Additionally, although we have strengthened the robustness of the estimates for the KV-cache, there is still a possibility of underestimation. In this rare case, SLINFER will attempt to scale up the cache again. If the attempt fails due to the node memory shortage, SLINFER will evict and re-schedule the request with the longest headroom.

