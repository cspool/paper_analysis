# 4 Resource Pool Management Methods

E"ective pool size management is critical in serverless platforms as it directly a"ects performance and cost. Oversized pools lead to idle resources and higher costs, while undersized pools result in allocation failures, violating the SLO.

The resource bu"ering problem is not unique to serverless platforms. For example, operating systems and networked environments typically maintain memory bu"ers to improve performance. This section discusses three main approaches that are widely used to determine a bu"er size: static, reactive, and prediction-based approaches. We evaluate the performance of these approaches in Section 7.

Static approach. In this approach, the system pre-allocates a !xed bu"er size in advance, which remains !xed regardless of the system load. Azure Functions maintains a !xed number of pre-warmed containers for each pool to avoid allocation failures. A high watermark and a low watermark can be used to lazily replenish the bu"er; instead of replenishing the bu"er immediately after an item is consumed, the bu"er is replenished only if it drops below the low watermark. The main advantage of this approach is its simplicity, as it eliminates the overhead associated with dynamic resizing and runtime monitoring. However, in large-scale serverless platforms, this approach requires manual management of many pools, which is cumbersome and prone to miscon!guration.

Reactive approach. A reactive approach can be used to dynamically manage the bu"er size based on its usage. The bu"er has a target size, and whenever a resource is consumed from the bu"er, the system starts replenishing the bu"er to its target size. Low and high watermarks are used to trigger changes to the target bu"er size. When many resources are consumed and the low watermark is reached, the target bu"er size is increased. Similarly, if many resources are idle and the high watermark is crossed, the bu"er size is decreased. The reactive approach adapts to the workload, which can yield better cost e#ciency. However, a reactive approach uses a lagging indicator, which does not !t bursty workloads: after a usage spike is observed, the target bu"er size is increased. This can lead to potential allocation failures in serverless platforms. Predictive approach. To overcome the lagging nature of the reactive approach, load forecasting can be used to predict the upcoming workload based on historical data and set the bu"er

size in advance. Similar to other approaches, the buffer is replenished eagerly. Forecasting models have limited effectiveness for bursty workloads, as described in Section 3. Another challenge with this approach lies in translating the predicted load into a pool size. We discuss this challenge in Section 7.

#### 5 DROPS

This section introduces DROPS, a new statistical, data-driven resource optimization method to manage resource pools in infinite-capacity systems, such as serverless platforms. In such systems, there are no constraints on the system's ability to create new resources (i.e., there is no upper bound on the number of resources, such as VMs and containers, that can be created in parallel).

DROPS utilizes historical traces to generate an accurate pool-size-to-success-rate mapping that can be used to determine the minimum pool size needed to meet a target SLO. DROPS uses an efficient sliding window analysis to capture burstiness relative to resource creation latencies. In this work, we use the target success rate of allocation requests as an SLO. That is, the SLO specifies the minimum percentage of allocation requests that must be fulfilled using pre-warmed containers. Section 5.1 presents the proposed method. Section 5.2 and Section 5.3 discuss how we apply the proposed method to optimize the container and VM pools.

#### 5.1 Resource Optimization Algorithm

DROPS optimizes resource pools in an infinite-capacity system that maintains a set of pools. Each pool has a target size, which the system strives to maintain at all times. Three main actions can occur to a resource pool: resource consumption, resource recycling, and resource replenishment. A resource consumption means that the system consumed an item from the pool. The system can keep using the consumed item indefinitely, or it can release the item after some time. Released resource items may be added back to the pool (i.e., resource recycling) and can be consumed again. When a resource item is consumed from the pool, the system triggers a resource replenishment to create a new resource item to refill the pool. The system can create an unlimited number of resource items in parallel. However, resource creation is not instantaneous and incurs some delay.

DROPS is based on a statistical demand-supply analysis. The demand represents the consumption of items from the pool, while the supply represents the time required to create resource items. That is, the duration from the time point at which a request to create a resource item is issued to the time point at which the resource item becomes ready and part of the pool.

DROPS utilizes the following property of infinite-capacity systems: the latency to fulfill a resource consumption request is bounded by the maximum creation latency of a resource. When a consumption request is received, it is either fulfilled immediately if the pool is not empty or must wait until a

### Algorithm 1 DROPS

```
\textbf{Input:} \ \mathsf{Demand} \ \mathsf{trace:} \ R, \mathsf{Recycling} \ \mathsf{trace:} \ Q, \mathsf{Resource} \ \mathsf{creation} \ \mathsf{latency}
     distribution: C, Number of samples: N
    Output: Pool-size-to-success-rate mapping: P
 3: Initialize an empty list L
 4: for i = 1 to N do
 5:
         Set WS to P_{100}(C)
                                            ▶ Window size = max creation latency
 6:
         for each consumption request r in R do
 7:
                                                      \triangleright Start with a pool size of zero
                                            \triangleright Set window w using timestamp of r
              Set w \leftarrow [t_r - WS, t_r]
 8:
 9:
              for each request x in w do
                                                             ▶ Analyze requests in w
                                                            ▶ Latency drawn from C
10:
                  Rep_{lat} \leftarrow sample from C
                                                        ▶ Is replenishment time > t_r
                  if t_x + Rep<sub>lat</sub> > t_r then
11:
12:
                      PS = PS + 1
                                                  ▶ Increment the needed pool size
13:
                  end if
14:
              end for
15
              q \leftarrow count of recycling events with timestamp \in w
16:
              PS = PS - q > Resource recycling reduces the needed pool size
17.
                                                         \triangleright Append pool size PS to L
              L \leftarrow L \cup \{PS\}
18:
         end for
19: end for
20: Compute P as the empirical CDF of L
21: return P
```

resource item becomes ready. In the worst case, the consumption request waits for the maximum creation latency. The pool size needed to fulfill a resource consumption request at time  $t_r$  depends only on the volume of the consumption and recycling within the interval  $[t_r - t_c, t_r]$ , where  $t_c$  is the maximum resource creation latency. In other words, it depends only on the consumption requests and recycling events occurring within the creation latency window preceding the request r.

Algorithm 1 shows the pseudo-code of DROPS, which requires the following inputs: 1) a demand trace, a sequence of resource consumption requests, 2) a recycling trace, a sequence of resource recycling events, and 3) a supply-latency distribution, a distribution of resource creation latency. DROPS outputs a mapping that can precisely set the pool size to meet any target SLO.

DROPS uses a sliding window analysis to construct a statistical distribution of the consumption demand over time. To determine the minimum pool size required to fulfill a request r from the pool, DROPS examines all prior consumption requests within the interval  $[t_r - t_c, t_r]$  (Algorithm 1, Line 9–14). Recall that every consumption request triggers a replenishment to refill the pool. Hence, for each request within the interval, DROPS samples a replenishment latency from the creation latency distribution (Line 10). If the sampled replenishment completes before  $t_r$ , then its demand does not affect the pool size required at  $t_r$ . In contrast, if the replenishment arrives after  $t_r$ , the request still affects the pool at time  $t_r$ , and the pool size is incremented accordingly (Lines 11–12). To determine the final pool size required to fulfill request r, we need to account for recycling events. DROPS counts the number of recycling events that occur within the interval, and uses it to adjust the pool size (Line 15–16). A resource recycling event reduces

the required pool size, as a recycled resource is added back to the pool and can be used to ful!ll future consumption requests. The adjusted pool size is added to the distribution (Line 17).

For each consumption request, the distribution contains one value representing the pool size required to ful!ll that request. The distribution is transformed to a pool-size-tosuccess-rate mapping by computing the empirical cumulative distribution function (CDF) of *L* (Line 20), which DROPS generates as an output. This mapping can be used to set the pool size to meet any SLO. For instance, if the target SLO is 99%, setting the pool size to the 99 percentile of the mapping guarantees meeting that SLO as long as the future workload has the same properties as the input workload.

DROPS can handle both stationary and bursty workloads. DROPS captures workload burstiness while accounting for the creation latency of resources. DROPS enables accurate sizing for any target SLO, a capability that other methods lack. The accuracy of DROPS depends on the length of the input traces, which production systems typically collect and store for long durations. DROPS is e#cient and does not involve any complex training algorithms. Furthermore, a system must re-run DROPS only if new pools are introduced or if there are changes in the workload that lead to a higher failure rate or resource underutilization than expected.

## 5.2 Container Pools Optimization

The serverless platform maintains multiple container pools. Each pool has its unique workload and container-creation latency distribution (Figure 4). Hence, we use Algorithm 1 to independently optimize each container pool. For each pool, we use the container-allocation trace as the demand trace and the container-creation latency distribution as the supplylatency distribution. In our platform, containers cannot be reused; when a container is deleted, its resources are released to the VM hosting the container. Hence, we set the recycling trace to an empty trace. The output pool-size-to-success-rate is then used to set the size of the container pool.

## 5.3 VM Pool Optimization

In this section, we discuss how DROPS is applied to optimize the VM pool. We use the empirical distribution of the VM-creation latency. We use the input container-allocation trace and the container-lifecycle distributions to generate the core-allocation and the core-recycling traces.

Core-allocation trace. Recall that the platform maintains a target number of idle VMs in the VM pool. As a result, whenever a VM is consumed, the platform creates a new VM to re!ll the pool. To generate a core-allocation trace, we map each container-allocation request in the container-allocation trace to a core-allocation event, with the number of allocated cores determined based on the target container size (e.g., 1 core or 2 cores). The generated core-allocation trace re\$ects the VM demand of all container pools.

The above assumes that the core-allocation event occurs at the same time as the container-allocation request, which is inaccurate. The core-allocation event must happen earlier than the container-allocation request. Speci!cally, the core-allocation event must precede the container-allocation request by at least one unit of container-creation latency. So, we adjust the core-allocation trace by shifting each coreallocation event backward in time by a duration sampled from the container-creation latency distribution.

Core-recycling trace. In our serverless platform, a container is deleted onceit becomesidle. Resources of deleted containers are released back to the VM pool (i.e., core recycling). Recycled cores signi!cantly impact the VM pool size as these cores become available to create new containers. Hence, neglecting the core-recycling events results in resource overprovisioning and higher costs.

The core-recycling events depend on the user workload characteristics, which vary from one tenant to another and from time to time. To account for that, we generate a corerecycling trace, a sequence of core-recycling events. For each container-allocation request, we determine the time at which the container will be deleted by sampling the duration of every stage in the container lifecycle from the container-lifecycle distributions.

Now, we have all the inputs required to apply Algorithm 1 to optimize the VM pool.We use the generated core-allocation trace as the demand trace, the generated core-recycling trace as the resource-recycling trace, and the VM-creation latency distribution as the supply distribution. The generated poolsize-to-success-rate mapping can be used to set the size of the VM pool to meet any target SLO.

