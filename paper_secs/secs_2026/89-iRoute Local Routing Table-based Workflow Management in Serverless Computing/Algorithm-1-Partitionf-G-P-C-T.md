# **Algorithm 1:** Partition(f, G, P, C, T)

```
Input:
```

```
f \triangleright The newly scaled function;
```

 $G \triangleright \text{The DAG of workflow};$ 

 $P \triangleright$  The partition results of existing instances;

 $C \triangleright$  The capacity of each function in a partition;

 $T \triangleright$  The request duration of functions;

#### Output

*index* ▷ The partition index of the newly scaled instance;

```
1 if C == {} then
         P_{1,f} \leftarrow P_{1,f} + 1, index \leftarrow 1;
         if P_{1,f_i} > 1 \ \forall f_i \in G then
              GetCapacity(G, P, C, T); // Calculate the
                capacity when all functions have scaled
 5 else
         index \leftarrow arg \min_{k} P_{k,f} < C_f, P_{k,f} \leftarrow P_{k,f} + 1;
          // Select the unfilled partition
         if index == -1 then
              index \leftarrow |P| + 1, P_{index} = \{..., f_i : 1, ...\}; // Add
                new partition when no available capacity
 9 return index;
10 Function GetCapacity(G, P, C, T):
         for stage_i \in G do
11
              if i == 1 then
12
                   for f_j \in stage_1 do
13
                       C_{f_i} \leftarrow P_{1,f_i}; // Use current number of
                          instances as the capacity
15
                   \lambda_{max} \leftarrow \min_{f_j \in stage_{i-1}} \frac{C_{f_j}}{T_{f_i}}; // \text{ Calculate the max}
                     workload of upstream stage
                   for f_j \in stage_i do
17
                        C_{f_i} \leftarrow \left[\lambda_{max} T_{f_i}\right]; // \text{ Calculate the}
                          required capacity to meet \lambda_{max}
```

same partition (i.e., *state* = *active*), thus delaying the synchronization of routing information for new instances located in other partitions does not affect the correctness of routing decisions.

Re-partitioning all instances for each scaling event would result in high coordination overhead under highly dynamic workloads. Therefore, the *syncing engine* opts to assign each newly scaled instance to an available partition based on the capacity, i.e., the maximum number of instances of each function that a partition can accommodate. For example in Figure 7(c), a partition can only accommodate 3A and 2B, thus  $A_4$ - $A_5$  and  $B_3$  must be allocated to  $Partition_2$ .

The principle behind calculating partition capacity is to avoid overloading instances within a filled partition as much as possible, i.e., the capacity for each function must be sufficient to handle the workload from upstream functions. For

example, the 3 instances of  $\{A_1, A_2, A_3\}$  can generate a maximum workload of  $\lambda = 15$ , which is less than the capacity accommodated by the 2 instances of  $\{B_1, B_2\}$  (i.e., the combined capacity of  $B_1$  and  $B_2$  is 20, Figure 7(b)). Algorithm 1 details the process for selecting a partition for each newly scaled instance. By default, the algorithm allocates all instances to a single partition until there are multiple instances for each function in the workflow (Lines 1-2). At this point, it determines the need to create multiple partitions and begins calculating the capacity of instances (Lines 3-4). Based on the current number of instances for functions in the first stage, the algorithm iteratively calculates the capacity required for each function to support the maximum upstream workload (Lines 10-18). Subsequently, it can assign newly scaled functions to partitions that are not yet filled (Lines 5-6), and creates new partitions when no available partitions exist (Lines 7-8). Note that, to ensure that each LRT has available routing destinations, at least one instance for each function is initiated upon the creation of a new partition.

**3.3.3 Fault tolerance.** Failures of the *CC* may lead to inconsistencies across multiple *LRTs*, potentially resulting in partial request blocking. For example, requests in *fan-in* scenarios may fail to be routed to the same downstream instance due to unsynchronized *LRTs*. However, the *GRT* is maintained in an external service (e.g., *ZooKeeper* [40]). In the event of a *CC* failure, the cluster manager (e.g., Kubernetes) automatically restarts the *CC*, which then reload the *GRT* from the external storage and synchronizes it with *LCs*, subsequently resolving any potential routing faults. Therefore, a *CC* failure does not disrupt workflow execution.

## 4 Implementation

We implement **iRoute** with Python3 atop *OpenFaaS* [41], one of the most popular open-source serverless platforms. The *CC* and *LC* modules are integrated with *OpenFaaS* as a web service and a runtime dependency library, respectively, to facilitate workflow execution. We replace *Open-FaaS*'s *alert-manager* with *scaling engine* to manage scaling while reusing other modules, such as *gateway* and *faas-netes*, to minimize development costs.

The *CC* is implemented using *socket* library [42] and runs as a daemon process. It receives JSON-formatted DAG definitions, and generates a *config.json* file for each function, which specifies the dependencies at *function-level*. Then, it places the *config.json* in corresponding function's code directory, and deploys workflows by invoking *OpenFaaS*'s *faas-cli* module through *subprocess*. After Kubernetes launches function instances and assigns IP addresses, the *CC* queries instance metadata through Kubernetes APIs and generates the required information for routing table, which are subsequently synchronized with *LCs* through TCP-based direct connections. The *CC* stores the *GRT* in *ZooKeeper* [40] and

notifies LC upon updates to provide synchronized consistency. For each function, CC creates a dedicated ZooKeeper node to store instance information and enforces read-only access for LCs through access control lists. To enhance security, the CC periodically re-generates node paths and authentication credentials, and subsequently re-notifies LCs.

The LC is packaged together with the user code and takes over the execution of the instance's main process (i.e., index.py). To support IPC-based 1-hop transfer, function instances are mounted with a shared tmpfs directory to set up named pipes and shared memory buffers. During workflow execution, the LC uses the epoll system call to listen for I/O events from all transmission channels' file descriptors, including pipes, sockets, and completion event channel of RDMA. It then utilizes the os [43], socket [42], and third-party rdma-core libraries [44] to handle message reception and transmission. Upon receiving complete intermediate data, the LC can directly invoke the user's code (i.e., handle function) and obtain the output. To avoid frequent network and deserialization overhead, LCs retrieve the latest routing table from ZooKeeper only when it is explicitly notified of an update. As for exactly-once semantics, we utilize Redis [45] as the third-party storage, and employ setnx (set if not exists) operation to ensure consistent buffer.

#### 5 Evaluation

