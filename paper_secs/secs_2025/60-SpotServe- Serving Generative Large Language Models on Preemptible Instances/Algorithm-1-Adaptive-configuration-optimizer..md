# Algorithm 1 Adaptive configuration optimizer.

```
1: function ConfigOptimizer(N_t, C_t, \alpha_t)
         if \exists C.\phi(C) \geq \alpha_t and cloud has enough instances for C
    then
 3:
              C_{t+1} \leftarrow \arg\min_{C \mid \phi(C) > \alpha_t} l_{req}(C)
 4.
         else
             C_{t+1} \leftarrow \arg\max_{C|N_t} \phi(C)
 5:
         \Delta \leftarrow \#Instances(C_{t+1}) - N_t
 6:
         if \Lambda > 0 then
 7.
 8:
              InstanceManager.alloc(\Delta, ondemand_and_spot)
 9:
10:
              InstanceManager.free(-\Delta, ondemand_first)
         ConfigUpdate(C_t, C_{t+1})
11:
```

instances. The new configurations are proposed by the *parallelization controller* and materialized by the *device mapper* and *migration planner*. Each inference engine also launches an *interruption arranger* to support stateful inference recovery to further reduce inference latency.

For the rest of this paper, we first introduce SpotServe's design, including the parallelization controller in §3.2, device mapper in §3.3, migration planner in §3.4, and interruption arranger in §4. Finally, we introduce SpotServe's implementation in §5 and evaluate its performance in §6.

#### 3.2 Parallelization Controller

SpotServe uses parallel configurations to identify a strategy to parallelize LLM serving across multiple GPU instances. A parallel configuration is represented as a tuple C = (D, P, M, B), where D, P, and M indicate the data, pipeline-model and tensor-model parallelism degrees, and *B* is the maximum mini-batch size. A key difference between SpotServe and existing spot-instance serving systems is that SpotServe can proactively adjust its parallel configuration by leveraging the ahead-of-time notifications provided by the cloud to handle instance preemptions and acquisitions. For each preemption and acquisition notification, SpotServe's parallelization controller opportunistically adjusts the parallelization configuration to improve LLM serving performance. Such reparallelization mechanism is also adaptive for fluctuating inference workload, which has been extensively studied in prior work [57].

Grace period of spot instance. Modern clouds generally offer a grace period (e.g., 30 seconds on Azure [3]) to allow a spot instance to complete running tasks before preempting the instance. Allocating new instance doesn't have a grace period, but initializing inference engine also takes a short period of time (e.g., 2 minutes for launching and initializing in our evaluations), which can be measured in advance and treated as the acquisition grace period in SpotServe.

*Adaptive configuration optimizer.* SpotServe uses an adaptive optimization algorithm to balance the trade-off

![](_page_5_Figure_2.jpeg)

**Figure 4.** Figure 4a shows an example of SpotServe changes the parallel configuration from (1,2,8) to (1,3,4) through context migration within the grace period and continues previous decoding progress of request  $r_3$ . Figure 4b shows an example bipartite graph between six available instances (i.e.,  $u_0 \sim u_5$ ) and topology positions in the new configuration (2,3,1). Here we only draw the weighted edges starting from  $u_1$ .

among throughput, latency, and cost. We use two time-varying variables  $C_t$  and  $N_t$  to denote the parallel configuration and the number of available instances at time step t. Note that  $N_t$  considers instances in the grace period, which includes newly allocated instances and excludes instances to be preempted. Let  $\phi(C)$  denote the serving throughput of Spot-Serve with a parallel configuration C and  $\alpha_t$  be the request arrival rate at time step  $t^1$ . Algorithm 1 shows the workflow of the optimizer, which mainly works when the current serving capability is not compatible with  $\alpha_t$  due to changes in instances' availability or serving workload.

Overall, the optimizer minimizes the end-to-end inference latency  $l_{rea}(C)$  while maintaining a throughput higher than  $\alpha_t$  (line 3). Specially, if there are multiple configurations that can achieve similar minimum inference latency, SpotServe selects the configuration with the lowest monetary cost (i.e., using fewest instances). In addition to minimizing  $l_{reg}(C)$ , other targets are also feasible in practice. For example, some SLO-sensitive scenarios (e.g., interactive applications) require a strict latency guarantee, rather than throughput. In that case, we can prioritize meeting the SLO requirement (i.e.,  $l_{reg}(C) \leq l_{SLO}$ ) and then minimize monetary cost if possible. When SpotServe's peak serving throughput can not exceed the request arrival rate  $\alpha_t$  (i.e.,  $\not\equiv C.\phi(C) \geq \alpha_t$ ), SpotServe updates its parallel configuration to maximize the overall serving throughput (line 5). The suggested configuration  $C_{t+1}$  may require more or less instances than before (line 6). Since the allocation of spot instance might not always success, SpotServe also supports optionally allocating ondemand instances to further improve serving throughput. Specifically, the instance manager allocates on-demand and spot instances at the same time (line 8) to avoid the waiting overhead when spot-instance allocation fails. The instance manager is also in charge of releasing the allocated instances (line 10) to alleviate over-provision, where on-demand instances have higher priority due to their costs. To alleviate the impacts of frequent disturbance of instance availability, SpotServe often maintains a few addition instances (e.g., two in our evaluation) as a candidate pool for smoother instance substitution. Finally, SpotServe updates the parallel configuration (line 11), and the interruption arranger (§4) decides when to complete reparallelization, especially for the cases triggered by instance availability changes. This step is still necessary even when  $C_{t+1} = C_t$ , since instance preemptions and acquisitions update instances' memberships.

The optimizer runs online and has negligible overhead (i.e., less than 1s) since the latency estimation of different configurations is done offline in advance. SpotServe's configuration exploration space is much larger than that considered by prior work such as Varuna [12], which only considers data and pipeline parallelism. It is also possible to extend SpotServe to more complicated model-parallel strategies [50, 61], which we leave as future work.

