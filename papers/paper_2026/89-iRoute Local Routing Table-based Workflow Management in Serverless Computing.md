# **iRoute: Local Routing Table-based Workflow Management in Serverless Computing** 

Yiming Li[†] , Laiping Zhao[†∗] , Zhiyuan Su[‡] , Guowei Liu[†] , Wenhao Huang[†] , Kang Chen[§] , Zhaolin Duan[†] , Jingjie Zong[†] , Wenxin Li[†] , Deze Zeng[¶] , Dong Zhang[‖] , Wenyu Qu[†] 

- College of Intelligence & Computing, Tianjin University, Tianjin Key Lab. of Advanced Networking 

‡IEIT SYSTEMS Co., Ltd, §Tsinghua University, ¶China University of Geosciences 

- ‖Jinan Inspur Data Technology Co., Ltd 

## **Abstract** 

Serverless computing typically relies on the centralized orchestrator and gateway for function-level and instance-level workflow management. Their intermediary intervention approaches fail to meet the strict microsecond-scale latency requirements of web services. To accelerate workflow execution, prior works have proposed offloading function dependencies to local functions and maintaining connections between frequently invoked instances. However, these methods still suffer from high routing lookup overhead and poor resource efficiency 

To address these issues, we propose offloading both orchestrating and routing capabilities from global to local to enable universal 1-hop transfers without compromising resource efficiency. We introduce **iRoute** , a local routing tablebased solution for workflow management. It adopts a duallayer architecture, where local routing controllers make correct routing decisions while concurrently cooperating with a centralized coordinator to ensure consistency across multiple local routing tables. **iRoute** can achieve sub-millisecond data transmission latency while maintaining high resource efficiency. Our experimental results demonstrate that **iRoute** outperforms state-of-the-art systems by up to 27.3× on latency, and improve the throughput by up to 6.7×. 

## _**CCS Concepts:**_ • **Computer systems organization** → **Cloud computing** . 

_**Keywords:**_ Serverless Computing, Local Routing Table, Data Transfer 

## **ACM Reference Format:** 

Yiming Li, Laiping Zhao, Zhiyuan Su, Guowei Liu, Wenhao Huang, Kang Chen, Zhaolin Duan, Jingjie Zong, Wenxin Li, Deze Zeng, 

- ∗Corresponding author: laiping@tju.edu.cn 

This work is licensed under a Creative Commons Attribution 4.0 International License. 

_EUROSYS ’26, Edinburgh, Scotland Uk_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 https://doi.org/10.1145/3767295.3769318 

Dong Zhang and Wenyu Qu. 2026. iRoute: Local Routing Tablebased Workflow Management in Serverless Computing. In _European Conference on Computer Systems (EUROSYS ’26), April 27–30, 2026, Edinburgh, Scotland Uk._ ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3767295.3769318 

## **1 Introduction** 

Workflow refers to a sequence of tasks executed in a specific order to solve complex problems. In fields such as scientific computing, big data, and artificial intelligence, workflows like Pegasus [1], MapReduce [2], and Pytorch graph [3] have been widely adopted. In recent years, _serverless workflows_ , which orchestrate stateless functions without the need for server management, have also gained significant attention in production [4–7]. Studies indicate that > 31% of serverless applications currently utilize _serverless workflows_ [4]. 

However, unlike traditional workflows, which follow a single-layer abstraction, _serverless workflows_ feature a twolayer abstraction model due to the auto-scaling nature: (1) _Function-level_ defines workflows **offline** as directed acyclic graphs (DAGs) and ensures that the execution order of functions aligns with the predefined logic. (2) _Instance-level_ identifies the specific instances of functions involved _during runtime_ and ensures that data transfer is completed. 

Therefore, data transmission between functions relies on a storage service and a two-layer control mechanism, where the orchestrator coordinates _inter-function_ invocation and the gateway handles _inter-instance_ communication. This process involves a total of six steps (Figure 2(a)), resulting in communication latency that far exceeds computation time, reaching >20× in the case of _Social Network_ [8]. 

The inefficiency of data transmission process primarily arises from two sources: (1) _Slow inter-function routing_ , i.e., the significant overhead involved in resolving downstream functions at the orchestrator and locating their instances through the gateway; (2) _Slow inter-instance transmission_ , i.e., the significant time associated with transferring data between dependent functions. To accelerate data transmission, existing work generally falls into two categories: (1) Offloading the resolution of downstream functions from the orchestrator to local instances (Figure 1(b), Unum [9]), thereby accelerating the routing process. However, this method does 

1 

144 

**==> picture [241 x 157] intentionally omitted <==**

**----- Start of picture text -----**<br>
Function Instance Dependencies Routing Table<br>Function-level:<br>Fn A Fn B Fn C<br>(b) Offload Dependencies LRT<br>     Gateway GRT<br>  Orchestrator Keep-alive<br>(a) Centralized (d) Local Routing Table<br>Instance-level:<br>(c) Keep-alive Connection<br>**----- End of picture text -----**<br>


**Figure 1.** Schematic overview of serverless workflow management. 

not improve the efficiency of looking up instance at the gateway. (2) Maintaining connections between instances with frequent communication to accelerate data transmission (Figure 1(c), FUYAO [10]). However, _keep-alive connection_ is only effective for frequent invocations, whereas, in the real world, most workflows (e.g., 80% in Azure durable functions [6]) are sparsely invoked. Therefore, no existing system can efficiently support both _inter-function routing_ and _inter-instance communication_ for serverless workflows. 

For _inter-function routing_ , the inefficiency mainly stems from the centralized orchestrator and gateway. This “intermediary intervention” introduces performance bottlenecks and incurs additional network communication overhead. To improve routing efficiency, both function resolution and instance lookup should be offloaded to local instances. 

For _inter-instance transmission_ , low efficiency is primarily due to the stateless nature of functions, which prevents them from directly locating each other. Therefore, intermediate data is transferred via _third-party forwarding_ . Specifically, function 𝐴 stores intermediate data in a third-party storage service, and function 𝐵 subsequently retrieves the data from that service. This “ _instance-level_ intermediary intervention” potentially accounts for up to 95% of overall latency [7, 11–14]. To improve transmission efficiency, a natural solution is to publish the address of each instance and allow instances to proactively establish direct connections. 

Establishing direct connections through the **Global Routing Table** (GRT) incurs 10-100 milliseconds overhead [10]. Moreover, it may lead to low resource efficiency due to the “ _binding scaling_ ” problem, i.e., downstream instances must synchronize scaling with upstream instances to prevent potential overload [15]. To address these issues, we argue for splitting _GRT_ into multiple **Local Routing Tables** (LRTs), thereby enabling local instances to perform autonomous routing and achieve 1-hop transmission. 

**==> picture [234 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
Global Routing Table (GRT) Dependencies<br>Trigger A Trigger B Trigger A Establish Conn.<br>Instance A1 3 6<br>Transfer<br>Instance B1 5<br>Storage 3 6<br>Put Get<br>Gateway 2 5 2 4<br>Lookup GRT<br>Orchestrator 1 4 1<br>(a) Third-party Forwarding (b) Keep-alive Connection<br>**----- End of picture text -----**<br>


**Figure 2.** The data transmission process of _third-party forwarding_ and _keep-alive connection_ . 

However, _LRT_ -based routing introduces new challenges: (1) _Correctness of routing decisions_ : In _fan-in_ scenarios, parallel upstream functions must independently route results to the same downstream instance without global coordination. To address this, we employ consistent hashing routing algorithm, which enables globally consistent routing decisions based on local views. Additionally, an _exploration mode_ is introduced to handle routing faults caused by inconsistencies between _LRTs_ . (2) _Performance overhead of LRTs synchronization_ : Workflows may exhibit high _fan-in_ degrees (e.g., several hundreds [6]) and experience burst workloads (e.g., 33,000× within one minute [16, 17]). Hence, the overhead of synchronizing _LRTs_ can be non-negligible. We adopt a dual-layer architecture comprising _local routing controllers_ and a _centralized coordinator_ to efficiently synchronize _LRTs_ . Moreover, a _partition policy_ is employed to enable rapid updates of _LRTs_ with minimal overhead. 

We propose **iRoute**[1] , a novel routing solution for _serverless workflows_ that addresses the inefficiency of inter-function communication. **iRoute** leverages _LRTs_ to enable local instances to autonomously locate dependent instances and establish direct connections, achieving sub-millisecond data transfer latency while preserving high resource efficiency. 

Our Contributions can be summarized as follows: 

- We analyze the performance and scalability issues of intermediate data transfer in _GRT_ -based methods, including _third-party forwarding_ and _keep-alive connection_ . 

- We propose **iRoute** , a comprehensive _LRT_ -based solution that enables 1-hop transfer across all scenarios (even for workflows with sparse invocations) without compromising on scalability. It achieves both low transmission latency and high resource efficiency. 

- Experimental results on diverse serverless workflows demonstrate that **iRoute** can outperform the state-of-theart by up to 27.3 × in latency and offer 6.7 × throughput. 

## **2 Background & Motivation** 

Workflows orchestrate sequences of functions, typically modeled as DAGs, to manage complex processes and data flows. 

> 1https://github.com/tanksys/iRoute 

2 

145 

With the advancement of serverless technology, _serverless workflows_ have attracted significant attention from cloud providers for building latency-sensitive web services. For example, AWS has rearchitected web services like e-commerce [18] and airline booking [19] using serverless. Microsoft also supports web applications based on Azure Functions [20]. However, due to the high overhead of inter-function data transmission (e.g., > 70% of the overall latency [10]), _serverless workflow_ -based web services often struggle to meet strict latency requirements. As the adoption of _serverless workflows_ continues to grow (e.g., > 31% [4]), reducing workflow communication latency has become a critical challenge for _serverless workflow_ -based web services. 

In _serverless workflows_ , communication between functions primarily consists of two steps: (1) _inter-function routing_ and (2) _inter-instance communication_ . Although Unum [9] has reduced routing overhead by offloading the resolution process from the orchestrator to local instances, there is still significant overhead in locating instances and transmitting data between instances. We analyze the _posting_ feature in the benchmark of _Social Network_ [8], and find that the computation times of its 10 functions range from 28 𝜇𝑠 to 1.6 𝑚𝑠, with even 6 functions having latencies below 150 𝜇𝑠. However, the data transfer overhead between two functions can surpass 3.3 𝑚𝑠, more than 20× the computation time. Consequently, the substantial overhead incurred by data transfer poses a major challenge in adopting serverless computing for latency-sensitive services. 

## **2.1 Intermediate Data Transfer Costs** 

Due to the lack of location awareness between stateless functions, existing intermediate data transfer relies on the _gateway_ and external storage, i.e., _third-party forwarding_ . As illustrated in Figure 2(a), the orchestrator triggers Function A via the gateway , which uses the GRT to locate an available instance (A1) . A1 completes execution and stores the intermediate data in external storage . The orchestrator then triggers Function B based on the dependencies, and B’s instance retrieves the data from storage. The entire process consumes > 3.3 𝑚𝑠. 

By maintaining a direct connection between function instances (i.e., _keep-alive connection_ ), we can bypass the overhead of request forwarding and _GRT_ queries, achieving submillisecond latency via 1-hop data transfer [10]. _Keep-alive connection_ requires routing lookups before the first 1-hop transfer: one for the orchestrator to specify function dependencies , and another for exchanging function instances’ addresses (e.g., IP address and port, named pipe) , both via the gateway (Figure 2(b)). After exchanging addresses, they can establish a persistent direct connection to achieve 1-hop transfer . This connection will be kept alive for a period of time for reuse by subsequent requests. 

We evaluate the data transfer efficiency by deploying _Social Network_ [8] on three existing platforms: (1) _FMI_ (implementing _keep-alive connection_ utilizing TCP); (2) _FUYAO_ (implementing _keep-alive connection_ using IPC (inter-process communication) and RDMA; and (3) _OpenFaaS_ (implementing _third-party forwarding_ ). Experimental configurations are detailed in §5.1, and all function instances are pre-warmed to eliminate the impact of cold starts. Prior studies [21–23] have reduced cold start overheads to the millisecond or even sub-millisecond level. Moreover, as the frequency of workflow invocations increases, the reuse rate of warm instances also rises. For example, Durable Functions reports a median cold start rate of only 0.35% for workflows invoked ≥ 100 times per day [6]. With cold start overhead minimized, routing and connection establishment become the dominant performance concern. 

The Azure trace [10, 16, 24] offers a diverse range of invocation patterns, including periodic, sparse, and bursty workloads, effectively covering the typical access patterns of web services [25–27]. Thus, we choose it to evaluate the performance of _keep-alive connection_ using two traces: a _stable_ trace with few bursts, and a _burst_ trace featuring frequent invocation spikes. (Figure 3(a)). It can be observed that _FMI_ can reduce the 99th percentile latency by 2.1× compared to _OpenFaaS_ under stable trace with frequent invocations. However, _FMI_ demonstrates significant performance fluctuations under burst trace, with latency even averaging 1.2× over _OpenFaaS_ . This is primarily due to the frequent bursts of invocation leading to scale up and down of function instances, which involves the establishment and release of direct connections. As shown in Figure 3(b), _FMI_ and _FUYAO_ can experience 8.7× and 2.2× higher latency compared to _OpenFaaS_ in the first request due to the substantial routing lookup overhead. Therefore, _keep-alive connection_ in scenarios with frequent bursts struggles to amortize the routing lookup overhead through reusing the connection for subsequent requests. Unfortunately, workloads with frequent bursts are common in both public and private clouds [16, 28, 29]. 

Besides burst traffic, many _serverless workflows_ are typically infrequently invoked. For example, those with an invocation frequency of < 100 times per day account for 80% in Durable Functions [6]. Under sparse invocations, maintaining connections results in resource wastage, while establishing a new connection for each request incurs high overhead (Figure 3(a)). Therefore, _keep-alive connection_ is also not suitable in such scenarios. 

**Observation I:** _Limited applicability: keep-alive connection can significantly reduce the overall latency under frequent invocations, yet it is unsuitable for scenarios with frequent bursts due to high routing lookup overhead._ 

3 

146 

**==> picture [484 x 105] intentionally omitted <==**

**----- Start of picture text -----**<br>
OpenFaaS FMI Workload Trace Routing lookup Overhead FUYAO FMI<br>10 [3]<br>450 150<br>Stable Burst<br>150<br>0 10 [2] 100<br>200 50<br>10 [1]<br>50<br>0 0<br>0 20 40 60 80 100 120 0 20 40 60 80 100 120 OpenFaaS    FMI  FUYAO 0 1000 2000<br>Timeline (hour)                      Timeline (hour) Queries Per Second<br>23.5 204.3 51.9<br>Latency (ms)<br>P99 latency (ms)<br>Queries Per Second Over-provisionin (%)<br>**----- End of picture text -----**<br>


**Figure 3.** Performance analysis of _Social Network_ . (a) The figure shows a dual-axis representation: the P99 latency of _Social Network_ on the left and QPS across two workload traces on the right. (b) A breakdown of the first end-to-end latency, with shading indicating the routing lookup overhead. (c) The over-provisioning percentage quantifies the additional function instances allocated beyond the minimum required to meet demand. We present the over-provisioning percentages for both _FMI_ and _FUYAO_ across different queries per second (QPS). 

## **2.2 Resource Provisioning Costs** 

Current serverless platforms make scaling decisions through a global engine based on the concurrency of functions. However, 1-hop transmission bypasses the global engine, preventing it from promptly detecting the concurrency level of each function, which renders the existing scaling mechanism ineffective. Therefore, _keep-alive connections_ typically adopt _binding scaling_ , wherein downstream functions scale concurrently with upstream functions, to mitigate potential high tail latency issues. Obviously, this scaling mechanism, which relies on resource over-provisioning, can result in inefficient resource utilization. 

We measure the over-provisioning percentage of function instances of _FMI_ and _FUYAO_ under various QPSs, and illustrate the results in Figure 3(c). Due to usage of a static policy that pre-defines which function pairs should establish connections, _FMI_ necessitates scaling the entire workflow when addressing high workloads. This approach completely compromises the fine-grained scalability of serverless computing, leading to a resource wastage of up to 177.8% under QPS = 900. Despite _FUYAO_ maintaining partial scalability by enabling upstream functions to keep alive direct connections with multiple instances of the same downstream function concurrently, the binding scaling problem still leads to 50% resource over-provisioning under QPS = 1200. 

**Observation II:** _Resource over-provisioning: Keep-alive connection needs to maintain direct connections between upstream and downstream functions, which results in binding scaling of dependent functions when scaling, leading to low resource efficiency._ 

## **2.3 Implications** 

A _serverless workflow_ system must possess the dual management capabilities at both the function and instance levels. The system should: 1) support the definition and orchestration of complex inter-function dependencies, and 2) reduce the overhead of intermediate data transmission. 

**Table 1.** Comparison of Existing Systems: FN and INS denote Function-level and Instance-level; and denote whether the DAG structure or functionality is supported; ‘C’ and ‘D’ denote Centralized and Decentralized. 

||FN|System<br>Fan-out patern<br>Fan-in patern<br>Architecture<br>Overhead<br>Data transfer|OpenFaaS<br><br><br>C<br>High<br>Forward|Unum<br><br><br>D<br>Low<br>Forward|FMI<br><br><br>D<br>Low<br>Direct|Fuyao<br><br><br>D<br>Low<br>Direct|**iRoute**<br><br><br>D<br>Low<br>Direct|
|---|---|---|---|---|---|---|---|
||INS|Fault tolerance||||||
|||Overhead|High|High|Medium|Medium|Low|



Although existing solutions [9, 10, 15] have made improvements in workflow management at both the function and instance levels (as shown in Table 1), they still fail to fully meet the needs of _serverless workflows_ . At the function-level, _decentralized design_ [9] offloads the process of resolving function dependencies to the local instance, avoiding high overhead. However, there remains significant latency in locating instances and inter-instance transmission. While _keepalive connection_ methods [10, 15] can reduce transmission latency, they still have problems such as limited applicability **(Observation I)** , resource over-provisioning **(Observation II)** . Therefore, a _serverless workflow_ system that can comprehensively address these challenges is still needed. 

## **2.4 Design Goals** 

Our design aims to accelerate inter-function data transmission in _serverless workflows_ by eliminating intermediary intervention overhead at both the _function-level_ and _instancelevel_ . First, dependency resolution process should be offloaded to local function instances to enable efficient coordination of function execution. Next, local instances should be able to independently select available downstream instances to support invocation routing . Finally, local instances should be able to directly locate each other and establish direct connections, thereby enabling 1-hop inter-instance communication at all times. 

4 

147 

**==> picture [241 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
Function Code Dependencies Local Routing Table Routing Algorithm<br>DAG<br>Parser<br>Developer DAG Functions Function-level<br>Instance-level<br>Centralized  Syncing Scaling<br>Coordinator (CC) Engine Engine<br>LRT Sync Scaling instances<br>User Requests Ins B1 LRT Ins D1<br>Ins A1 Ins E1<br> Local Routing  Ins C1<br> Controller (LC) Partition #1<br>**----- End of picture text -----**<br>


**Figure 4.** The overall architecture of **iRoute** . 

## **3 Design** 

## **3.1 Overview** 

Figure 4 shows an overview of the architecture of **iRoute** , featuring a dual-layer structure composed of multiple _Local Routing Controllers_ ( _LCs_ ) and a _Centralized Coordinator_ ( _CC_ ). Unlike traditional centralized controllers, _CC_ is only responsible for handling lightweight coordination tasks. It includes three main components: the _DAG parser_ , the _scaling engine_ , and the _syncing engine_ . At the function level, the _DAG parser_ is responsible for parsing the dependencies between functions in a workflow and distributing these dependencies to each function. At the instance level, each function instance is equipped with an _LC_ acting as a sidecar. The _LC_ maintains a **Local Routing Table (LRT)** that stores routing information for dependent instances and uses a _routing algorithm_ to make routing decisions, enabling efficient 1-hop data transmission. During the execution of the workflow, the _scaling engine_ makes scaling decisions based on the workloads, and the _syncing engine_ then updates the _LRTs_ at the _LCs_ . 

To ensure the correctness of routing decisions ( **Challenge 1** ), _LC_ operates in two modes: _stable mode_ and _exploration mode_ . In the _stable mode_ , _LC_ employs routing algorithms based on calling patterns to determine the correct routing destination. In the _exploration mode_ , _LC_ further detects and resolves routing faults. When function scaling occurs, the _CC_ instructs _LCs_ to switch from _stable_ to _exploration mode_ . Upon completion of _LRT_ synchronization, _LCs_ will be informed to revert to the _stable mode_ . 

To reduce the synchronization overhead of _LRTs_ ( **Challenge 2** ), **iRoute** divides the instances into multiple _partitions_ , where intermediate data transfer is only allowed within the same partition. For example, the instances {𝐴1, 𝐵1, ...} in Figure 4 belong to the same 𝑝𝑎𝑟𝑡𝑖𝑡𝑖𝑜𝑛#1. When 𝐴1 completes, the 𝐿𝐶𝐴1 takes over the responsibility of routing intermediate data. It first queries _LRT_ to identify available instances of the function _B_ (e.g., 𝐵1 in 𝑝𝑎𝑟𝑡𝑖𝑡𝑖𝑜𝑛#1), then employs the routing algorithm to select a destination and finally initiates 1-hop data transmission based on the destination’s address 

in _LRT_ . When scaling out, the _scaling engine_ creates a new instance and assigns it to a partition. The _syncing engine_ then updates the relevant LRTs accordingly. 

Note that, we only consider the direct invocation pattern where upstream functions can invoke downstream functions themselves for immediate processing. In the indirect invocation pattern, where downstream functions are triggered by external events such as storage, timers, or queues, it is generally unnecessary to establish direct connections between functions, since upstream functions cannot determine when or which downstream functions will be triggered. 

## **3.2 Local Routing Controller** 

**3.2.1 LRT Design.** _LRT_ stores the address information of downstream functions, allowing the LC to make routing decisions based on this information. Hence, the number of entries in the _LRT_ equals the sum of downstream function instances. Each entry in _LRT_ contains four fields: (1) _Instance ID_ denotes the identifier of a downstream function instance. It is used for instance selection in routing algorithms; (2) _Address_ denotes the IP address and port number used for accessing the instance; (3) _State_ denotes the reachable state of a destination instance, which can be either _active_ or _inactive_ . It is set to _active_ when the local instance and the destination instance belong to the same partition; (4) _Tunnel_ denotes the transmission channel for accessing the instance. It could be _IPC_ , _Socket_ , or _RDMA_ . _IPC_ denotes the intra-node direct communication utilizing Linux FIFO. _Socket_ denotes the inter-node direct communication based on socket objects. _RDMA_ denotes the inter-node direct communication utilizing network acceleration devices. 

Note that, for the same downstream function instance, the _LRTs_ of upstream functions may store different _state_ and _tunnel_ information. For example in Figure 4, _LRTs_ of function instances 𝐴1-𝐴2 all store the _instance ID_ and _address_ of 𝐵1-𝐵3. Assuming that 𝐴1 and 𝐵1 are located on the same node, the _tunnel_ in the _LRT_ of 𝐴1 is set to _IPC_ ; otherwise, it is set to _socket_ . Additionally, _state_ is configured based on the partitions; thus, instances of _B_ , which belong to the same partition as 𝐴1, have their _state_ set to _active_ . 

**3.2.2 Routing Algorithm.** Following the dependencies defined in a workflow, after an upstream function completes execution, its result is passed to the local _LC_ . The _LC_ then identifies downstream functions for this intermediate data. It filters out available downstream instances (i.e., _state_ = _active_ ) from the _LRT_ . Then, it employs a routing algorithm to choose a routing destination for each downstream function. After selecting the downstream instance, it routes intermediate data to the destination via the transmission channel indicated by the _tunnel_ field in _LRT_ . 

For workflows with a static DAG structure, the calling pattern between dependent functions typically involves three types: _chain_ (1 →1), _fan-out_ (1 →𝑛), and _fan-in_ (𝑛→1). 

5 

148 

**==> picture [217 x 62] intentionally omitted <==**

**----- Start of picture text -----**<br>
Function Scaling LRT Sync<br>CC<br>2.Enter 4.Sync 6.Exit<br>1.Submit 3.ACK 5.ACK<br>LC<br>Stable Mode Exploration Mode Stable Mode<br>**----- End of picture text -----**<br>


**Figure 5.** The interaction process between _CC_ and _LC_ . 

The operational mechanisms of the routing algorithm under different patterns are as follows: 

**Chain (** 1 →1 **)** : there is a one-to-one dependency relationship between upstream and downstream functions. Since there is only one downstream function, the routing algorithm primarily addresses the load balancing of multiple instances of the downstream function. We evaluate several commonly used load balancing algorithms, including _Random_ , _Consistent Hashing_ and _Round Robin_ . As shown in Figure 16 in Section 5.5, _Round Robin_ achieves the best load balancing performance. Therefore, we adopt the _Round Robin_ algorithm in our design. 

**Fan-out (** 1 →𝑛 **)** : the intermediate data produced by an upstream function is simultaneously distributed to multiple downstream functions. In this case, the local _LC_ considers it as a variation of chain, and sequentially invokes _Round Robin_ algorithm for each downstream function. 

**Fan-in (** 𝑛→1 **)** : the intermediate data produced by multiple upstream functions converges to a single downstream function. In this case, the _LC_ employs a _Consistent Hashing with Exploration Mode (CH-EM)_ algorithm to ensure correct routing decisions. This algorithm maps downstream instances (i.e., _instance IDs_ ) in the _LRT_ onto a hash ring, then maps the request’s ID onto the hash ring and selects the nearest instance as the destination. Thus, although each function makes independent decisions, it still ensures that the intermediate data of the same request is routed to the same downstream instance. 

When a workflow includes conditional logic (e.g., _Choice_ in AWS Step Functions [30]), its DAG evolves at runtime based on intermediate data or execution state. In such scenarios, _LRTs_ must be retrieved from _CC_ at runtime, incurring connection establishment overhead similar to that of _centralized orchestrator_ . One possible solution is to pre-distribute _LRTs_ containing the address information of all branch functions, enabling _LC_ to select the appropriate routing target based on the conditional logic. We leave the exploration of this approach for future work. 

**3.2.3 Fault-tolerant Routing.** In the _fan-in_ communication pattern, when all _LRTs_ of upstream functions are consistent, the _CH-EM_ algorithm can ensure that the intermediate data from multiple upstream functions converges to the same destination. However, during the synchronization process (Figure 5), temporary inconsistencies in the _LRTs_ of upstream functions may cause routing decision conflicts. 

**==> picture [231 x 101] intentionally omitted <==**

**----- Start of picture text -----**<br>
C2: Active C2: Active<br>CC<br>A<br>Req Exec C<br>Ins A1 B<br>Ins B1<br>Clean<br>Ins C1<br>Fault  Re-routing<br>Decision<br>Ins C2<br>Feedback Feedback<br>**----- End of picture text -----**<br>


**Figure 6.** Exploration mode initiates feedback after detecting fault routing decisions. 

In this case, the _CH-EM_ algorithm employs the _exploration mode_ to detect and fix fault routing decisions (Figure 6). 

The basic idea of _exploration mode_ is the _check_ and _rerouting_ mechanism. Downstream _LCs_ check the integrity of the received data and request the upstream instances to re-route any missing intermediate data. The _check_ occurs upon receiving data from the critical path, typically the result from the upstream function that arrives the latest due to the longest execution time. The content to be checked is whether all upstream results have been received based on the dependency. If so, the function execution proceeds; otherwise, the re-routing mechanism is triggered for each missing function. As illustrated in Figure 6, _A->C_ represents the critical path, thus the _LC_ of 𝐶2 starts checking when receiving data from 𝐴1. It deduces that a fault routing decision may have occurred due to the lack of data from _B_ . Note that, we denote the path with the longest execution time as the critical path [31]. Upon workflow submission, the _DAG parser_ statically profiles parallel branches to determine the initial critical path, which is subsequently refined by the _CC_ using collected function latencies (§3.3.1). Inaccurate critical path analysis can lead to unnecessary re-routing, incurring latency overhead of 1.9-11.1% as shown in Figure 13. 

Due to the uncertainty of the required data’s location, the _LC_ identifies all instances of each missing function based on transmission channels, and sends re-routing request to them. If the upstream _LC_ holds the required data, it will reroute it to the instance that sent the feedback (e.g., 𝐶2) and instruct the original destination (e.g., 𝐶1) to remove the previously routing data. Otherwise, the _LC_ continues to monitor subsequent execution requests until it discovers the required data or receives confirmation from downstream that the data has been re-routed by another instance. To address all potential routing faults, as shown in Figure 5, the _CC_ ensures that all _LCs_ switch to _exploration mode_ before initiating _LRTs_ synchronization. Moreover, the upstream _LCs_ need to temporarily buffer intermediate data for re-routing. To minimize the storage overhead, the _LC_ utilizes both proactive release and expire mechanism to manage the lifetime of each buffer entry, which is detailed in §3.2.4. 

To ensure routing correctness during down scaling, an instance is removed only after all involved requests have 

6 

149 

been fully processed. First, the _CC_ updates the status of the instance to be released as _inactive_ in the _GRT_ and synchronizes with the _LRTs_ of upstream dependent function instances. Then, upstream functions check the _inactive_ status in _LRTs_ and stop sending requests to the instance. Next, the _CC_ notifies _LCs_ of the instance to check and clear its local buffer. Once all local buffers are actively cleared, _LCs_ inform the _CC_ , which can then safely remove the instance from the _GRT_ and release it. 

**3.2.4 Fault-Tolerant Execution.** Current serverless platforms provide three execution semantics: _at-most-once_ , _atleast-once_ and _exactly-once_ [32–34]. _At-most-once_ indicates that each invocation is attempted no more than once, without automatic retries upon failure. _At-least-once_ achieves reliability by retrying failed invocations until completion, potentially leading to duplicate executions. _Exactly-once_ ensures that each invocation produces a single definitive result that is delivered to downstream functions only once, even in the presence of failures or retries. **iRoute** supports all three semantics and adopts _at-most-once_ as the default execution mode. Stronger guarantees ( _at-least-once_ and _exactly-once_ ) require explicit user configuration to enable retries upon execution failure. 

However, offloading both _function-level_ and _instance-level_ management capabilities to local instances presents a key challenge for **iRoute** in maintaining execution semantics. First, function interactions in our system bypass the centralized gateway, which typically tracks execution status and retries failed invocations. Consequently, detecting failed function invocations becomes more challenging in our distributed architecture. Second, while 1-hop data transfer eliminates the interaction overhead with third-party storage, it also increases the risk of data loss. As a result, failed invocations may require re-executing the entire workflow to reproduce the lost data, leading to considerable overhead. 

**iRoute** supports _at-most-once_ semantics by executing each invocation no more than once and immediately aborting failed requests without retries. Specifically, when _LC_ detects an execution anomaly or _CC_ identifies an instance failure, the corresponding invocation is aborted and an error response is returned to the upstream caller. 

**iRoute** leverages _data buffer_ and _re-scheduling_ mechanisms to provide _at-least-once_ execution semantics. First, each _LC_ temporarily buffers execution results and routing destinations, enabling downstream functions to be re-executed when necessary. Second, when an instance failure is detected, the _CC_ notifies all upstream _LCs_ within the same partition to check for potential failed invocations, i.e., buffer entries were routed to the crashed instance. The _re-scheduling_ mechanism is then activated for these affected invocations. _LCs_ employ _Round Robin_ and _CH-EM_ routing algorithms, as described in §3.2.2 and §3.2.3, to choose new destinations for 

each entry. Finally, downstream _LCs_ retrieve buffered data and re-execute corresponding invocations. 

To prevent redundant re-executions, **iRoute** implements a proactive buffer release mechanism. This mechanism is based on the following key insight: once all downstream _LCs_ have buffered and routed their execution results, the previous buffer entries are no longer necessary to enforce execution semantics. Additionally, each buffer entry is assigned a Time To Live (TTL), e.g., twice the duration of the downstream function, and expires once timeout occurs. However, even after expiration, the identity of the expired invocation is retained until proactive release, allowing for the recovery of lost data if needed. 

For applications that require stricter execution semantics (e.g., payments), **iRoute** introduces a centralized buffer mechanism. While this approach incurs additional performance overhead, it guarantees _exactly-once_ execution. In this model, _LCs_ perform atomic operations on a third-party storage to buffer exactly one execution result for each function invocation [9]. If the buffer is successfully written, the execution result is routed to downstream instances via 1-hop transfer. Otherwise, only the location of the existing buffer is transmitted. This ensures that downstream functions consistently receive the same input across multiple executions. Additionally, _LCs_ check for the presence of buffer before execution. If it exists, the execution is skipped, and the buffer location is transferred directly to downstream functions. 

Note that, the _exactly-once_ guarantee may not apply to workflows with external side effects, e.g., transactions. This well-known issue are independent of the routing architecture. Therefore, orthogonal techniques, such as log-based approaches like Boki [35] and Halfmoon [36], can be integrated with **iRoute** to enhance the execution semantics. 

**3.2.5 Trust model.** After offloading the routing functionality from the global to local instances, _LC_ is currently part of the trusted computing base (TCB). To mitigate security risks, **iRoute** restricts _LC_ to read-only access to _GRT_ and relies on periodically refreshed access credentials. If _LC_ is to be removed from TCB, authentication mechanisms must be introduced between _LC_ and _CC_ , as well as among _LCs_ themselves. Additionally, Byzantine fault tolerant protocols [37, 38] can be integrated to further enhance security. 

## **3.3 Centralized Coordinator** 

In this section, we discuss how the _CC_ collaborates with _LCs_ to achieve function scaling and synchronization of _LRTs_ , while addressing highly dynamic serverless workloads without compromising on scalability. 

**3.3.1 Function Scaling.** Existing serverless platforms utilize a centralized controller to route each request, which records the concurrency of each function in real-time to make scaling decisions. For example, AWS Lambda provisions a separate instance for each concurrent request [39]. 

7 

150 

However, delegating the routing functionality to local sidecar bypasses centralized components for intermediate data transfer, rendering existing scaling policy ineffective. Therefore, **iRoute** redesigns the scaling mechanism through the collaboration of _CC_ and _LCs_ . 

**==> picture [241 x 105] intentionally omitted <==**

**----- Start of picture text -----**<br>
Local Routing Table ID State … ID State …<br>(LRT) ID State … B1 Active … B1 Inactive …<br>ID State … B1 Active … B2 Active … B2 Inactive …<br>B1 Active … B2 Active … B3 Inactive … B3 Active …<br>A1 A1 A2 A3 A1 A2 A3 A4 A5<br>tA  = 0.2s<br>tB  = 0.1s<br>B1 B1 B2 B1 B2 B3<br>(a)  λ  = 5 req/s (b)  λ  = 15 req/s (c)  λ  = 25 req/s<br>Partition1 Partition1 Partition1 Partition2<br>**----- End of picture text -----**<br>


**Figure 7.** The function scaling decisions are determined based on the average requests per second 𝜆 and request duration 𝑡. And the instances are divided into multiple partitions to reduce the overhead of _LRTs_ synchronization. 

The _scaling engine_ of _CC_ makes scaling decisions for each function based on the workload metrics collected from each _LC_ , including the average requests per second (𝜆) and request duration (𝑡). Function scaling can be triggered in two scenarios: (1) _LC_ detects that the local instance is overloaded; (2) the _scaling engine_ periodically collects metrics and identifies instance redundancy. Specifically, a function instance is considered overloaded if the request per second exceed its processing capacity (𝜆>[1] 𝑡[) or a given threshold (][𝜆>][𝜆][𝑡ℎ][),] which is utilized to avoid transmission faults caused by excessive workloads. Similar to AWS Lambda [39], the _scaling engine_ calculates the concurrency for each function as the expected number of instances for the current workload: 

## **Algorithm 1:** Partition(𝑓 , 𝐺, 𝑃, 𝐶, 𝑇) 

**Input:** 𝑓▷ The newly scaled function; 𝐺▷ The DAG of workflow; 𝑃▷ The partition results of existing instances; 𝐶▷ The capacity of each function in a partition; 𝑇▷ The request duration of functions ; **Output:** 𝑖𝑛𝑑𝑒𝑥▷ The partition index of the newly scaled instance; **1 if** _C == {}_ **then 2** 𝑃1,𝑓 ←𝑃1,𝑓 + 1, 𝑖𝑛𝑑𝑒𝑥←1; **3 if** 𝑃1,𝑓𝑗 > 1 ∀𝑓𝑗 ∈𝐺 **then 4** 𝐺𝑒𝑡𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦(𝐺, 𝑃, 𝐶, 𝑇); // Calculate the capacity when all functions have scaled **5 else 6** 𝑖𝑛𝑑𝑒𝑥← arg min𝑘 𝑃𝑘,𝑓 < 𝐶𝑓 , 𝑃𝑘,𝑓 ←𝑃𝑘,𝑓 + 1; // Select the unfilled partition **7 if** _index == -1_ **then 8** 𝑖𝑛𝑑𝑒𝑥←|𝑃| + 1, 𝑃𝑖𝑛𝑑𝑒𝑥 = {..., 𝑓𝑗 ∶1, ...}; // Add new partition when no available capacity **9 return** 𝑖𝑛𝑑𝑒𝑥; **10 Function** GetCapacity( _G, P, C, T_ ) **: 11 for** 𝑠𝑡𝑎𝑔𝑒𝑖 ∈𝐺 **do 12 if** _i == 1_ **then 13 for** 𝑓𝑗 ∈𝑠𝑡𝑎𝑔𝑒1 **do 14** 𝐶𝑓𝑗 ←𝑃1,𝑓𝑗 ; // Use current number of instances as the capacity **15 else 16** 𝜆𝑚𝑎𝑥 ← 𝑓𝑗 ∈𝑠𝑡𝑎𝑔𝑒min𝑖−1 𝐶𝑇𝑓𝑗𝑓𝑗 ; // Calculate the max workload of upstream stage **17 for** 𝑓𝑗 ∈𝑠𝑡𝑎𝑔𝑒𝑖 **do 18** 𝐶𝑓𝑗 ←⌈𝜆𝑚𝑎𝑥 𝑇𝑓𝑗 ⌉; // Calculate the required capacity to meet 𝜆𝑚𝑎𝑥 

**==> picture [161 x 31] intentionally omitted <==**

where 𝜆𝑖 and 𝑡𝑖 denotes the average arrival requests per second and request duration of the 𝑖𝑡ℎ instance of the function, respectively. And 𝛼[′] presents the number of existing instances. For example in Figure 7(b), the _LC_ of 𝐴1 reports 𝜆𝐴1 = 15 and 𝑡𝐴1 = 0.2, and then the _scaling engine_ calculates 𝛼𝐴 = 3. Thus, it scales up instance 𝐴2 and 𝐴3. 

**3.3.2 LRT Synchronization.** The _syncing engine_ of _CC_ is responsible for distributing the routing information of newly scaled instances to all _LCs_ of its dependent functions. However, simultaneously communicating with all _LCs_ , which may refer to thousands of function instances, can incur significant synchronization overhead. Therefore, the _syncing engine_ further partitions function instances, and reduces the synchronization overhead by firstly synchronizing only the _LRTs_ of the partition containing the newly scaled instance. _LCs_ can only select routing target from instances within the 

same partition (i.e., 𝑠𝑡𝑎𝑡𝑒= 𝑎𝑐𝑡𝑖𝑣𝑒), thus delaying the synchronization of routing information for new instances located in other partitions does not affect the correctness of routing decisions. 

Re-partitioning all instances for each scaling event would result in high coordination overhead under highly dynamic workloads. Therefore, the _syncing engine_ opts to assign each newly scaled instance to an available partition based on the capacity, i.e., the maximum number of instances of each function that a partition can accommodate. For example in Figure 7(c), a partition can only accommodate 3𝐴 and 2𝐵, thus 𝐴4-𝐴5 and 𝐵3 must be allocated to 𝑃𝑎𝑟𝑡𝑖𝑡𝑖𝑜𝑛2. 

The principle behind calculating partition capacity is to avoid overloading instances within a filled partition as much as possible, i.e., the capacity for each function must be sufficient to handle the workload from upstream functions. For 

8 

151 

example, the 3 instances of {𝐴1, 𝐴2, 𝐴3} can generate a maximum workload of 𝜆= 15, which is less than the capacity accommodated by the 2 instances of {𝐵1, 𝐵2} (i.e., the combined capacity of 𝐵1 and 𝐵2 is 20, Figure 7(b)). Algorithm 1 details the process for selecting a partition for each newly scaled instance. By default, the algorithm allocates all instances to a single partition until there are multiple instances for each function in the workflow (Lines 1-2). At this point, it determines the need to create multiple partitions and begins calculating the capacity of instances (Lines 3-4). Based on the current number of instances for functions in the first stage, the algorithm iteratively calculates the capacity required for each function to support the maximum upstream workload (Lines 10-18). Subsequently, it can assign newly scaled functions to partitions that are not yet filled (Lines 5-6), and creates new partitions when no available partitions exist (Lines 7-8). Note that, to ensure that each _LRT_ has available routing destinations, at least one instance for each function is initiated upon the creation of a new partition. 

**3.3.3 Fault tolerance.** Failures of the _CC_ may lead to inconsistencies across multiple _LRTs_ , potentially resulting in partial request blocking. For example, requests in _fan-in_ scenarios may fail to be routed to the same downstream instance due to unsynchronized _LRTs_ . However, the _GRT_ is maintained in an external service (e.g., _ZooKeeper_ [40]). In the event of a _CC_ failure, the cluster manager (e.g., Kubernetes) automatically restarts the _CC_ , which then reload the _GRT_ from the external storage and synchronizes it with _LCs_ , subsequently resolving any potential routing faults. Therefore, a _CC_ failure does not disrupt workflow execution. 

## **4 Implementation** 

We implement **iRoute** with Python3 atop _OpenFaaS_ [41], one of the most popular open-source serverless platforms. The _CC_ and _LC_ modules are integrated with _OpenFaaS_ as a web service and a runtime dependency library, respectively, to facilitate workflow execution. We replace _OpenFaaS_ ’s _alert-manager_ with _scaling engine_ to manage scaling while reusing other modules, such as _gateway_ and _faasnetes_ , to minimize development costs. 

The _CC_ is implemented using _socket_ library [42] and runs as a daemon process. It receives JSON-formatted DAG definitions, and generates a _config.json_ file for each function, which specifies the dependencies at _function-level_ . Then, it places the _config.json_ in corresponding function’s code directory, and deploys workflows by invoking _OpenFaaS_ ’s _faascli_ module through _subprocess_ . After Kubernetes launches function instances and assigns IP addresses, the _CC_ queries instance metadata through Kubernetes APIs and generates the required information for routing table, which are subsequently synchronized with _LCs_ through TCP-based direct connections. The _CC_ stores the _GRT_ in _ZooKeeper_ [40] and 

notifies _LC_ upon updates to provide synchronized consistency. For each function, _CC_ creates a dedicated _ZooKeeper_ node to store instance information and enforces read-only access for _LCs_ through _access control lists_ . To enhance security, the _CC_ periodically re-generates node paths and authentication credentials, and subsequently re-notifies _LCs_ . 

The _LC_ is packaged together with the user code and takes over the execution of the instance’s main process (i.e., _index.py_ ). To support IPC-based 1-hop transfer, function instances are mounted with a shared _tmpfs_ directory to set up named pipes and shared memory buffers. During workflow execution, the _LC_ uses the _epoll_ system call to listen for I/O events from all transmission channels’ file descriptors, including pipes, sockets, and completion event channel of RDMA. It then utilizes the _os_ [43], _socket_ [42], and third-party _rdma-core libraries_ [44] to handle message reception and transmission. Upon receiving complete intermediate data, the _LC_ can directly invoke the user’s code (i.e., _handle_ function) and obtain the output. To avoid frequent network and deserialization overhead, _LCs_ retrieve the latest routing table from _ZooKeeper_ only when it is explicitly notified of an update. As for _exactly-once_ semantics, we utilize _Redis_ [45] as the third-party storage, and employ _setnx_ (set if not exists) operation to ensure consistent buffer. 

## **5 Evaluation** 

## **5.1 Methodology** 

**Experimental setup.** We evaluate **iRoute** on a 12-node cluster. Each node is configured as shown in Table 2. And these nodes are connected via 25 Gbps, full-bisection bandwidth Ethernet. 

**Table 2.** Experimental cluster configurations 

||**Component**<br>CPU Device<br>CPU Treads<br>Storage<br>Operating System|**Confguration**<br>Intel Xeon Gold 6338 @ 2.00GHz<br>128 cores (64 physical cores)<br>256GB Memory with 2TB SSD<br>Ubuntu 22.04|
|---|---|---|



**Benchmarks.** We evaluate **iRoute** and the comparison systems using three typical benchmarks. _Social Network_ is a latency-sensitive application from _DeathstarBench_ [8] that creates post embedded with text, media, links and user tags. _Excamera_ [46] is a video-processing application that encodes and processes chunks in parallel. And the function execution dominates the overall latency. _Financial Industry Regulatory Authority (FINRA)_ is a financial application that validates trades based on trade and market data. It can be configured with varying _widths_ (i.e., _fan-in_ degree). Specifically, we implement four distinct variants of _FINRA_ with widths of 5, 10, 20 and 40. We further use production trace from Azure Function [16] for evaluation, including the _stable_ trace, _burst_ trace and _sporadic_ trace. 

9 

152 

**==> picture [236 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
Connection Lookup OpenFaaS iRoute-Socket Nightcore<br>8 100.7 Unum FUYAO-RDMA FUYAO-IPC<br>6 6 FMI27.7 15.7iRoute-RDMA iRoute-IPC<br>4 4<br>2 2<br>1<br>0 0<br>100B 1KB 1MB 512MB<br>FMI FUYAO iRoute Data Size<br>SocketIPCRDMAIPCSocketRDMA<br>Latency (ms) 0.13ms 0.23ms 3.04ms 670.01ms<br>Norm. Transfer Latency<br>**----- End of picture text -----**<br>


**Figure 8.** Latency **Figure 9.** Comparison of data transof building direct mission latency under various data connection. volumes. 

**Comparison systems.** We compare **iRoute** with state-ofthe-art systems, including _OpenFaaS_ [41], _Unum_ [9], _Nightcore_ [47], _FMI_ [15] and _FUAYO_ [10]. Specifically, _OpenFaaS_ uses a standalone orchestrator and basic _third-party forwarding_ for workflow execution; _Unum_ offloads _function-level_ dependencies to local instance to avoid the interaction overhead with a standalone orchestrator; _Nightcore_ employs IPC to speed up data transfer between functions within the same node; _FMI_ implements _keep-alive connection_ based on Socket; _FUYAO_ also supports _keep-alive connection_ with IPC and RDMA-based direct connections. To ensure fairness, each system is compared only with others that adopts the same communication channel, i.e., Socket, RDMA, or IPC. Moreover, all function instances are pre-warmed to eliminate the impact of cold starts. 

**Metrics.** We evaluate the performance of **iRoute** using three metrics: (1) Latency (e.g., data transfer latency, end-to-end latency and overhead); (2) Throughput (i.e., QPS); (3) Resource provisioning (i.e., required number of instances to support various QPS). We conduct each test five times and report the average results. 

## **5.2 Intermediate Data Transfer Latency** 

We first evaluate the intermediate data transfer latency between two functions. 

**Connection overhead: iRoute can establish direct connections within** ≤2.3 𝑚𝑠 **.** As shown in Figure 8, _FMI_ and _FUYAO_ require the exchange of instance address information through controller, leading to connection overhead of 5.4-100.7 𝑚𝑠. In contrast, **iRoute** only needs to retrieve destination addresses from _LRT_ and avoids any interaction with a global controller, which reduces the connection time to 0.2-2.3 𝑚𝑠. In fact, **iRoute** can further eliminate the overhead of connection establishment when using UDP [48, 49]. For RDMA-based transmission, existing acceleration techniques such as KECORE [50] can be integrated into **iRoute** to further reduce overhead. 

**Data transfer latency: iRoute can reduce the transfer latency to 288** 𝑚𝑠 **when transmitting 512MB of data.** After the direct connection is established, **iRoute** , _FMI_ and _FUYAO_ all use the same 1-hop data-transfer mechanism, so 

their communication latency differences are mainly determined by the transfer mode. Because _FMI_ provides only a socket-based transmission while _FUYAO_ supports both IPC and RDMA, we group _FUYAO-RDMA_ with **iRoute-RDMA** , _FUYAO-IPC_ with **iRoute-IPC** , and _FMI_ with **iRoute-Socket** for comparison. Figure 9 shows the latency for transmitting various data sizes. Compared to _OpenFaaS_ and _Nightcore_ which both rely on _third-party forwarding_ , **iRoute** can transfer 512MB in 288 𝑚𝑠, representing a speedup of 8.6× and 2.3 × over _OpenFaaS_ and _Nightcore_ , respectively. Among IPC, socket and RDMA, IPC offers the best performance when the data size ≤ 1KB, while RDMA performs better when the data size ≥ 1MB. This is primarily because IPC requires additional copying of large data into the shared memory [10, 51]. 

## **5.3 Benchmark Analysis** 

Next, we evaluate **iRoute** using three latency-sensitive workflows: _Social Network_ , _Excamera_ and _FINRA-5_ (i.e., _fan-in_ =5). **P99 latency: iRoute can reduce the P99 latency of applications by 1.1-81.8** × **.** We use Azure traces [10, 16] to generate three production-like 120-hour workloads, with 10 minutes keep-alive duration for function instance. The P99 response latency of three benchmark applications under these workloads is recorded. As shown in Figure 10(a) and 10(c), in _Social Network_ and _FINRA-5_ , **iRoute** consistently outperforms all other systems in P99 latency. In the most extreme case ( _FINRA-5_ in _stable_ ), **iRoute-Socket** outperforms _FMI_ by up to 81.8×. The poor performance of _FMI_ is primarily due to its need to synchronously scale all other functions and establish direct connections during scaling each function. _OpenFaaS_ and _Unum_ exhibit up to 7.8× and 6.3× higher latency than **iRoute-Socket** , respectively, mainly because they rely on _third-party forwarding_ . Compared to _Nightcore_ , _FUYAO-IPC_ and _FUYAO-RDMA_ , **iRoute** still reduces the P99 latency by up to 2.3×, 27.3× and 10.4×, respectively. Note that, **iRoute-RDMA** exhibits higher latency compared to **iRoute-Socket** . This is mainly due to the frequent re-establishment of direct connections, with RDMA incurring a connection overhead 6.5× greater than that of socket. Furthermore, these applications transmit relatively small data (e.g., <1KB in _FINRA_ ), preventing performance benefits typically offered by RDMA. In Figure 10(b), _Excamera_ in which function execution dominates the overall latency, shows less sensitivity to communication and connection overhead compared to other applications. Nevertheless, **iRoute** still achieves lower latency than _OpenFaaS_ , _Unum_ , _FMI_ and _FUYAO_ , with reductions of 1.4×, 1.3×, 1.6× and 1.1×, respectively. **Resource efciency: iRoute can reduce the number of function instances by up to 6.5** × **while supporting the same throughput.** As shown in Figure 11, across all experimental cases, **iRoute** requires fewer instances than _OpenFaaS_ , _Unum_ , _Nightcore_ , _FMI_ , _FUYAO-RDMA_ and _FUYAO-IPC_ , with average reductions of 1.6×, 2.6×, 1.1×, 2.3×, 1.8× and 

10 

153 

**==> picture [505 x 442] intentionally omitted <==**

**----- Start of picture text -----**<br>
OpenFaaS Unum FMI iRoute-Socket FUYAO-RDMA iRoute-RDMA Nightcore FUYAO-IPC iRoute-IPC<br>102 400 102<br>1 200 101<br>10<br>0<br>Stable Burst Sporadic Stable Burst Sporadic Stable Burst Sporadic<br>(a) SocailNetwork (b) Excamera (c) FINRA-5<br>Figure 10.  The P99 latency under different production traces, including  Stable ,  Burst and  Sporadic .<br>OpenFaaS Unum FMI iRoute-Socket FUYAO-RDMA iRoute-RDMA Nightcore FUYAO-IPC iRoute-IPC<br>100 Bottleneck 150 No more instances 40<br>60<br>30<br>40 1 00 Bottleneck<br>20<br>20<br>50<br>10<br>10<br>100 300 500 700 1500 2000 5 10 25 50 75 100 150 100 300 500 700 900 1500 2500<br>(a) SocialNetwork (b) Excamera (c) FINRA-5<br>100 2 00 2 00 No more instances<br>60<br>100<br>40<br>100<br>20 40 Bottleneck<br>60<br>10 20 40<br>100 300 500 700 900 1500 2500 100 300 500 700 900 1500 2500 100 300 500 700 900 1500 2500<br>(d) FINRA-10 (e) FINRA-20 (f) FINRA-40<br>Figure 11.  Comparison of the number of required instances for supporting different QPS.<br>OpenFaaSUnum FMIiRoute-Socket FUYAO-RDMAiRoute-RDMA NightcoreFUYAO-IPC iRoute-IPC the number of instances required is consistent across all sys-<br>tems except FMI and FUYAO . This is due to their binding<br>2<br>scaling problem, i.e., downstream instances must scale syn-<br>chronously with upstream instances to avoid potential over-<br>1 load. The comparison between FINRA-5 (Figure 11(c)) and<br>FINRA-40  (Figure 11(f)) reveals that increasing the  fan-in  de-<br>0 gree leads to a higher computational load and larger num-<br>SN EC F-5 F-10 F-20 F-40  Unum<br>P99 Latency (ms)<br># of Instances<br># of Instances<br>Norm. Throughput<br>**----- End of picture text -----**<br>


the number of instances required is consistent across all systems except _FMI_ and _FUYAO_ . This is due to their _binding scaling_ problem, i.e., downstream instances must scale synchronously with upstream instances to avoid potential overload. The comparison between _FINRA-5_ (Figure 11(c)) and _FINRA-40_ (Figure 11(f)) reveals that increasing the _fan-in_ degree leads to a higher computational load and larger number of required instances for the same QPS. _OpenFaaS_ , _Unum_ and _Nightcore_ also experience central bottlenecks earlier under higher _fan-in_ . In _FINRA-40_ , _FUYAO-RDMA_ requires more frequent scaling to meet its memory allocation demands and encounters the _binding scaling_ problem earlier, resulting in the highest instance requirements, even surpassing _Unum_ . We further measure the maximum throughput each system can sustain on our local cluster. As shown in Figure 12, **iRoute** can support up to 100×, 80×, 10×, 5×, 11.8× and 2.3× throughput over _OpenFaaS_ , _Unum_ , _Nightcore_ , _FMI_ , _FUYAO-RDMA_ and _FUYAO-IPC_ , respectively. 

**Figure 12.** The maximum throughput of _Social Network_ (SN), _Excamera_ (EC) and _FINRA_ (F) in local cluster. 

1.4×, respectively. In _Social Network_ (Figure 11(a)), _Unum_ requires the most number of instances, primarily due to the highest additional overhead for function execution, including the slowest socket-based _third-party forwarding_ and connection overhead of directly invoking downstream functions. _FMI_ necessitates the simultaneous scaling of all functions, leading to higher instance demand than _OpenFaaS_ and _FUYAO_ . Since RDMA performs less efficiently than IPC for transmitting small data, _FUYAO-RDMA_ requires more instances than _FUYAO-IPC_ . As for **iRoute** , **iRoute-IPC** requires the fewest instances among its variants, owing to its minimal transmission overhead. In _Excamera_ (Figure 11(b)), where the overall latency is dominated by function execution time, 

## **5.4 Fault Tolerance** 

During _LRT_ updates, **iRoute** enables fault-tolerant routing mechanism to re-route intermediate data and ensure execution correctness. When function execution fails, **iRoute** activates fault-tolerant execution to re-schedule and re-execute 

11 

154 

**==> picture [241 x 244] intentionally omitted <==**

**----- Start of picture text -----**<br>
MB<br>Stable Mode Excamera<br>6 5<br>Exploration Mode<br>4 0 10 25 50 75 100<br>KB Social Network<br>2 25 FINRA-5<br>0 0 100 500 1000 1500 2000<br>F-5 F-10 F-20 F-40 Queries Per Second<br>Figure 13. Average and Figure 14.  Maximum mem-<br>P99 latencies of FINRA (F) ory usage of buffered data<br>workflow under stable and in each  LC for  at-least-once<br>exploration mode. execution semantics.<br>No-FT 0% failure 10% 25% 50% BGL<br>20<br>35 500<br>20 10<br>30<br>1 0 450<br>25 5<br>400 10<br>20<br>EO EO ALO EO EO ALO EO EO ALO<br>Unum iRoute Unum iRoute Unum iRoute<br>(a) Social Network (b) Excamera (c) FINRA-5<br>Latency (ms)<br>     Maximum Buffer Memory<br>340<br>Latency (ms) 320<br>**----- End of picture text -----**<br>


**Figure 15.** Latency distribution of fault-tolerant execution in Unum (left axis) and iRoute (right axis). 

the failed request. In this section, we first evaluate the overhead of fault-tolerant routing by comparing latencies under _stable_ and _exploration_ modes. Next, we simulate different failure rates by injecting failure intervals of varying durations during application execution and assess the resulting overhead of fault-tolerant execution. 

## **Fault-tolerant routing: Exploration mode leads to an** 

**increase in P99 latency by 1.9-11.1%.** As shown in Figure 13, _exploration_ mode results in a 7.7% increase in average latency compared to _stable mode_ in _FINRA_ . Moreover, as the _fan-in_ degree increases, the P99 latency overhead rises from 1.9% to 11.1%, primarily due to the need for downstream functions to send re-routing requests to more upstream functions. 

**Fault-tolerant execution: Semantics guarantee results in an increase of 0.5-18** 𝑚𝑠 **in average latency.** As **iRoute** supports both _exactly-once_ (EO) and _at-least-once_ (ALO) semantics, and only _Unum_ among the compared systems offers EO semantics, our comparison focuses on **iRoute** , _Unum_ and a non-fault-tolerant baseline ( _No-FT_ ). As illustrated in Figure 15, the fault injection rate ranges from 0% (i.e., no failures injected, but with data buffer overhead) to 50%. The results show that **iRoute-EO** and _Unum-EO_ exhibit similar latency patterns under fault injection. Moreover, as the injection rate increases, response latency rises significantly for both systems. In contrast, **iRoute-ALO** which leverages local buffer to provide relaxed ALO semantics, experiences significantly lower latency overhead compared to EO. To further evaluate the impact of real-world failure patterns, we also inject failure following the trace from the BlueGene/L 

**==> picture [242 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
Random Hash Round Robin iRoute<br>QPS=2K QPS=25 QPS=2K QPS=2K QPS=25 QPS=2K<br>2<br>0.4<br>0.2 1<br>0.0 0<br>SN EX F-5 SN EX F-5<br>IPC RDMA Socket IPC RDMA Socket IPC RDMA Socket<br>Norm. Latency<br>Instance Load CoV<br>**----- End of picture text -----**<br>


**Figure 16.** Load balancing of routing algorithms. (a) The coefficient of variation (CoV). (b) The normalized latency. 

supercomputer system ( _BGL_ ) [52, 53], where each millisecond is mapped to whether a failure occurred in a 3-minute window of _BGL_ . Under this workload, **iRoute** incurs minor fault-tolerance overhead, primarily because the failure rate in _BGL_ is relatively low and approximately corresponds to fault injection of 7%. 

## **5.5 Load Balancing** 

Delegating routing decisions to _LC_ may compromise loadbalancing performance, since _LC_ routes requests based on a local rather than a global view. Therefore, in this section we evaluate **iRoute** ’s load-balancing performance. **iRoute** employs a hybrid routing strategy, which uses _Round Robin_ ( _RR_ ) for non _fan-in_ functions and _Consistent Hashing_ ( _Hash_ ) for _fan-in_ functions. We compare **iRoute** against several classic global routing algorithms: _Random_ , _Hash_ , and _RR_ . **Load balancing: iRoute’s routing can achieve load balancing performance comparable to that of global routing.** As shown in Figure 16(a), we evaluate the load-balancing performance of three benchmarks under QPS of 2,000, 25 and 2,000, respectively. _RR_ demonstrates the best performance, achieving a coefficient of variation (CoV) ranging from 0 to 0.0067. In contrast, _Hash_ performs the worst, with a CoV reaching 0.144, primarily due to the non-uniform distribution of hash values on the hash ring. Thus, **iRoute** uses _RR_ as the preferred algorithm. However, in _fan-in_ patterns, _RR_ becomes unsuitable because it may route results from multiple upstream functions to different downstream instances, compromising routing correctness. Therefore, **iRoute** employs _Hash_ under _fan-in_ to ensure that results from the same request are routed to a single downstream instance. Although **iRoute** exhibits higher CoV than _RR_ and _Random_ , its impact on end-to-end latency remains limited in workflows where _fan-in_ functions have short execution times. As illustrated in Figure 16(b), the end-to-end latency of **iRoute** is only 0.3-6.8% higher than that of _RR_ , This is mainly because _fan-in_ functions in the three applications account for < 20% of the overall latency. 

## **5.6 Overheads** 

Then, we evaluate the overhead of **iRoute** . 

12 

155 

**==> picture [233 x 86] intentionally omitted <==**

**----- Start of picture text -----**<br>
Min<br>0.2 20 Partition<br>Max<br>0.1 10<br>0.0 0<br>Baseline+Scaling+Fault+Routing F-5 F-10 F-20 F-40<br>Latency (ms) Latency (ms)<br>**----- End of picture text -----**<br>


**Figure 17.** Average and **Figure 18.** LRTs synchroP99 latencies of 2 no-op nization overhead of _FINRA_ functions w/o and with dif(F) w/o ( _Min_ and _Max_ ) and ferent components in _LC_ . with instance _Partition_ . 

**Overhead breakdown: iRoute introduces only an average overhead of 35** 𝜇𝑠 **for** _**LC**_ **.** To illustrate the overhead of our design to support _LC_ , we further break down **iRoute** ’s performance, and present the results of 2 no-op functions in Figure 17. “Baseline” only implements 1-hop transfer between function, similar to _FUYAO_ . “Scaling” adds metrics collection of requests per second and request duration for collaborative scaling. “Fault” further supports _at-least-once_ execution with local buffer. “Routing” increases the number of available downstream instances from 1 to 20 and utilizes _CH-EM_ algorithm to select a destination for each request. Experimental results indicate that the three components result in an average additional overhead of 7, 12 and 16 𝜇𝑠, respectively. 

**Synchronization overhead: Instance partition can reduce the overhead of LRTs synchronization by 1.9-5.6** ×. We measure the _LRTs_ synchronization overhead in different scenarios using the _FINRA_ application and present the results in Figure 18. As the number of function instances continues to increase, the cost of _LRTs_ synchronization also significantly increases. For example, _FINRA-5_ can have up to 184 upstream function instances in our local cluster (i.e., _Max_ ), which increases the synchronization overhead of _LRTs_ by 5.5× compared to the minimum of 5 upstream instances, i.e., _Min_ . With instance partition, the _CC_ only needs to synchronize _LRTs_ between a maximum of 11 instances each time, with a cost increase of only 5.3% compared to _Min_ . 

## **6 Related Work** 

**Function level:** Standalone orchestrators, such as AWS Step Functions [54] and Durable Functions [55], are a primary method for composing serverless workflows. Unum [9] seeks to eliminate the intermediary overhead introduced by orchestrators through offloading function dependencies to local instances. Similarly, Pheromone [51] and DataFlower [56] propose data-centric function orchestration, where data storage is used to trigger target functions, reducing frequent interactions with a standalone orchestrator. Additionally, ORION [6] introduces several optimization techniques for static DAGs, such as resource rightsizing and co-location of parallel invocations. **iRoute** can also benefit from these 

techniques. For example, **iRoute** can pre-calculate all the routing decisions to enhance load balancing and further minimize end-to-end latency. Moreover, optimizing instances resources and parallelism can improve overall resource efficiency. 

**Instance level:** _Third-party forwarding_ is currently the most commonly used method for data transmission in serverless computing [7, 13, 54]. One optimization approach is to accelerate data forwarding using distributed caching [57–59]. Moreover, prior studies leverage data locality to reduce transmission overhead [13, 47, 51, 56]. However, these methods still rely on indirect data transfer between function instances, which prevents optimal performance. Boxer [60] and FMI [15] explore stateful connections to enable 1-hop direct transfer using TCP hole punching. FUYAO [10] achieves 1-hop transfer between functions both within and across nodes, leveraging DPU to further accelerate inter-node communication. Despite these advances, they still rely on _GRT_ for routing lookup, introducing additional latency of hundreds of microseconds. Furthermore, _keep-alive connection_ employs _binding scaling_ to avoid high latency, but suffer from low resource efficiency. Additionally, systems like SAND [61], Faasm [62] and SPRIGHT [63] also implement similar local routing mechanisms, but are limited within a node or even a single instance. They cannot entirely eliminate the routing lookup or data overhead caused by the orchestrator when scaled to larger cases. 

## **7 Conclusion** 

This paper presents **iRoute** , a serverless workflow system that offloads both _function-level_ and _instance-level_ management capabilities to local instances. **iRoute** utilizes a local sidecar to effectively resolve function-level dependencies and route requests based on LRT. It has two key attributes. (1) Low transmission latency for all scenarios: **iRoute** enables efficient 1-hop data transfer for both frequent and sparse invocations without additional constraints. (2) High resource efficiency: **iRoute** can dynamically select routing destination and update routing tables in a timely manner to maintain high scalability in 1-hop transfer. 

## **8 Acknowledgments** 

We thank the anonymous reviewers and our shepherd, Rodrigo Bruno, for their insightful comments and suggestions that greatly improved this paper. This work is supported by the National Key Research and Development Program of China (No.2022YFB4500702); project ZR2022LZH018 supported by the Shandong Provincial Natural Science Foundation; the National Natural Science Foundation of China under grant 62372322, 62432015; and Tianjin Science and Technology Plan Project (24ZXKJGX00060). 

13 

156 

## **References** 

- [1] Ewa Deelman, Gurmeet Singh, Mei-Hui Su, James Blythe, Yolanda Gil, Carl Kesselman, Gaurang Mehta, Karan Vahi, G. Bruce Berriman, John Good, Anastasia C. Laity, Joseph C. Jacob, and Daniel S. Katz. Pegasus: A framework for mapping complex scientific workflows onto distributed systems. _Sci. Program._ , 13(3):219–237, 2005. 

- [2] Jeffrey Dean and Sanjay Ghemawat. Mapreduce: simplified data processing on large clusters. _Commun. ACM_ , 51(1):107–113, January 2008. 

- [3] The PyTorch Foundation. PyTorch GraphModule. https: //pytorch.org/docs/main/fx.html?spm=5176.28103460.0.0. 57dc5d27SL2uI1#torch.fx.GraphModule, 2023. 

- [4] Simon Eismann, Joel Scheuner, Erwin van Eyk, Maximilian Schwinger, Johannes Grohmann, Nikolas Herbst, Cristina L. Abad, and Alexandru Iosup. The state of serverless applications: Collection, characterization, and community consensus. _IEEE Transactions on Software Engineering_ , 48(10):4152–4166, 2022. 

- [5] Simon Eismann, Joel Scheuner, Erwin van Eyk, Maximilian Schwinger, Johannes Grohmann, Nikolas Herbst, Cristina L. Abad, and Alexandru Iosup. A review of serverless use cases and their characteristics, 2021. 

- [6] Ashraf Mahgoub, Edgardo Barsallo Yi, Karthick Shankar, Sameh Elnikety, Somali Chaterji, and Saurabh Bagchi. ORION and the three rights: Sizing, bundling, and prewarming for serverless DAGs. In _16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)_ , pages 303–320, Carlsbad, CA, July 2022. USENIX Association. 

- [7] Ashraf Mahgoub, Karthick Shankar, Subrata Mitra, Ana Klimovic, Somali Chaterji, and Saurabh Bagchi. SONIC: Application-aware data passing for chained serverless applications. In _2021 USENIX Annual Technical Conference (USENIX ATC 21)_ , pages 285–301. USENIX Association, July 2021. 

- [8] Yu Gan, Yanqi Zhang, Dailun Cheng, Ankitha Shetty, Priyal Rathi, Nayan Katarki, Ariana Bruno, Justin Hu, Brian Ritchken, Brendon Jackson, Kelvin Hu, Meghna Pancholi, Yuan He, Brett Clancy, Chris Colen, Fukang Wen, Catherine Leung, Siyuan Wang, Leon Zaruvinsky, Mateo Espinosa, Rick Lin, Zhongling Liu, Jake Padilla, and Christina Delimitrou. An open-source benchmark suite for microservices and their hardware-software implications for cloud & edge systems. In _Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems_ , ASPLOS ’19, page 3–18, New York, NY, USA, 2019. Association for Computing Machinery. 

- [9] David H Liu, Amit Levy, Shadi Noghabi, and Sebastian Burckhardt. Doing more with less: Orchestrating serverless applications without an orchestrator. In _20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23)_ , pages 1505–1519, 2023. 

- [10] Guowei Liu, Laiping Zhao, Yiming Li, Zhaolin Duan, Sheng Chen, Yitao Hu, Zhiyuan Su, and Wenyu Qu. Fuyao: Dpu-enabled direct data transfer for serverless computing. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ , ASPLOS ’24, page 431–447, New York, NY, USA, 2024. Association for Computing Machinery. 

- [11] Joseph M. Hellerstein, Jose M. Faleiro, Joseph Gonzalez, Johann Schleier-Smith, Vikram Sreekanti, Alexey Tumanov, and Chenggang Wu. Serverless computing: One step forward, two steps back. In _9th Biennial Conference on Innovative Data Systems Research, CIDR 2019, Asilomar, CA, USA, January 13-16, 2019, Online Proceedings_ . www.cidrdb.org, 2019. 

- [12] Swaroop Kotni, Ajay Nayak, Vinod Ganapathy, and Arkaprava Basu. Faastlane: Accelerating Function-as-a-Service workflows. In _2021 USENIX Annual Technical Conference (USENIX ATC 21)_ , pages 805– 820. USENIX Association, July 2021. 

- [13] Zijun Li, Yushi Liu, Linsong Guo, Quan Chen, Jiagan Cheng, Wenli Zheng, and Minyi Guo. Faasflow: enable efficient workflow execution for function-as-a-service. In _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , ASPLOS ’22, page 782–796, New York, NY, USA, 2022. Association for Computing Machinery. 

- [14] Yiming Li, Laiping Zhao, Yanan Yang, and Wenyu Qu. Rethinking deployment for serverless functions: A performance-first perspective. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ , SC ’23, New York, NY, USA, 2023. Association for Computing Machinery. 

- [15] Marcin Copik, Roman Böhringer, Alexandru Calotoiu, and Torsten Hoefler. Fmi: Fast and cheap message passing for serverless functions. In _Proceedings of the 37th International Conference on Supercomputing_ , ICS ’23, page 373–385, New York, NY, USA, 2023. Association for Computing Machinery. 

- [16] Mohammad Shahrad, Rodrigo Fonseca, Inigo Goiri, Gohar Chaudhry, Paul Batum, Jason Cooke, Eduardo Laureano, Colby Tresness, Mark Russinovich, and Ricardo Bianchini. Serverless in the wild: Characterizing and optimizing the serverless workload at a large cloud provider. In _2020 USENIX annual technical conference (USENIX ATC 20)_ , pages 205–218, 2020. 

- [17] Xingda Wei, Fangming Lu, Tianxia Wang, Jinyu Gu, Yuhan Yang, Rong Chen, and Haibo Chen. No provisioned concurrency: Fast RDMA-codesigned remote fork for serverless computing. In _17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)_ , pages 497–517, Boston, MA, July 2023. USENIX Association. 

- [18] Amazon Web Services. Serverless Ecommerce Platform. https: //github.com/aws-samples/aws-serverless-ecommerce-platform, 2025. 

- [19] Amazon Web Services. AWS Serverless Airline Booking. https: //github.com/aws-samples/aws-serverless-airline-booking, 2025. 

- [20] Microsoft. Serverless web application. https://learn.microsoft.com/ en-us/azure/architecture/web-apps/serverless/architectures/webapp, 2025. 

- [21] Dong Du, Tianyi Yu, Yubin Xia, Binyu Zang, Guanglu Yan, Chenggang Qin, Qixuan Wu, and Haibo Chen. Catalyzer: Sub-millisecond startup for serverless computing with initialization-less booting. In _Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems_ , pages 467– 481, 2020. 

- [22] Jialiang Huang, MingXing Zhang, Teng Ma, Zheng Liu, Sixing Lin, Kang Chen, Jinlei Jiang, Xia Liao, Yingdi Shan, Ning Zhang, Mengting Lu, Tao Ma, Haifeng Gong, and YongWei Wu. Trenv: Transparently share serverless execution environments across different functions and nodes. In _Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles_ , SOSP ’24, page 421–437, New York, NY, USA, 2024. Association for Computing Machinery. 

- [23] Xiaohu Chai, Tianyu Zhou, Keyang Hu, Jianfeng Tan, Tiwei Bie, Anqi Shen, Dawei Shen, Qi Xing, Shun Song, Tongkai Yang, Le Gao, Feng Yu, Zhengyu He, Dong Du, Yubin Xia, Kang Chen, and Yu Chen. Fork in the road: Reflections and optimizations for cold start latency in production serverless systems. In Lidong Zhou and Yuanyuan Zhou, editors, _19th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2025, Boston, MA, USA, July 7-9, 2025_ , pages 199–218. USENIX Association, 2025. 

- [24] Yanan Yang, Laiping Zhao, Yiming Li, Huanyu Zhang, Jie Li, Mingyang Zhao, Xingzhen Chen, and Keqiu Li. Infless: a native serverless system for low-latency, high-throughput inference. In _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , pages 768–781, 2022. 

14 

157 

- [25] Martin F. Arlitt and Carey L. Williamson. Web server workload characterization: the search for invariants. In _Proceedings of the 1996 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems_ , SIGMETRICS ’96, page 126–137, New York, NY, USA, 1996. Association for Computing Machinery. 

- [26] National Institute of Standards and Technology. Tweets2011 Twitter Collection. https://trec.nist.gov/data/tweets/, 2011. 

- [27] Guido Urdaneta, Guillaume Pierre, and Maarten van Steen. Wikipedia workload analysis for decentralized hosting. _Elsevier Computer Networks_ , 53(11):1830–1845, July 2009. http://www.globule.org/publi/ WWADH_comnet2009.html. 

- [28] Ao Wang, Shuai Chang, Huangshi Tian, Hongqi Wang, Haoran Yang, Huiba Li, Rui Du, and Yue Cheng. FaaSNet: Scalable and fast provisioning of custom serverless container runtimes at alibaba cloud function compute. In _2021 USENIX Annual Technical Conference (USENIX ATC 21)_ , pages 443–457. USENIX Association, July 2021. 

- [29] Alireza Sahraei, Soteris Demetriou, Amirali Sobhgol, Haoran Zhang, Abhigna Nagaraja, Neeraj Pathak, Girish Joshi, Carla Souza, Bo Huang, Wyatt Cook, Andrii Golovei, Pradeep Venkat, Andrew Mcfague, Dimitrios Skarlatos, Vipul Patel, Ravinder Thind, Ernesto Gonzalez, Yun Jin, and Chunqiang Tang. Xfaas: Hyperscale and low cost serverless functions at meta. In _Proceedings of the 29th Symposium on Operating Systems Principles_ , SOSP ’23, page 231–246, New York, NY, USA, 2023. Association for Computing Machinery. 

- [30] Amazon Web Services. Choice workflow state. https://docs.aws. amazon.com/step-functions/latest/dg/state-choice.html, 2025. 

- [31] Yu-Kwong Kwok and I. Ahmad. Dynamic critical-path scheduling: an effective technique for allocating task graphs to multiprocessors. _IEEE Transactions on Parallel and Distributed Systems_ , 7(5):506–521, 1996. 

- [32] Amazon Web Services. Understanding retry behavior in Lambda. https://docs.aws.amazon.com/lambda/latest/dg/invocationretries.html, 2024. 

- [33] Amazon Web Services. Choosing workflow type in Step Functions. https://docs.aws.amazon.com/step-functions/latest/dg/ choosing-workflow-type.html, 2024. 

- [34] Google Cloud. Enable event-driven function retries. https://cloud. google.com/functions/docs/bestpractices/retries, 2024. 

- [35] Zhipeng Jia and Emmett Witchel. Boki: Stateful serverless computing with shared logs. In _Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles_ , SOSP ’21, page 691–707, New York, NY, USA, 2021. Association for Computing Machinery. 

- [36] Sheng Qi, Xuanzhe Liu, and Xin Jin. Halfmoon: Log-optimal faulttolerant stateful serverless computing. In _Proceedings of the 29th Symposium on Operating Systems Principles_ , SOSP ’23, page 314–330, New York, NY, USA, 2023. Association for Computing Machinery. 

- [37] Diogo S. Antunes, Afonso N. Oliveira, André Breda, Matheus Guilherme Franco, Henrique Moniz, and Rodrigo Rodrigues. Alea-BFT: Practical asynchronous byzantine fault tolerance. In _21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)_ , pages 313–328, Santa Clara, CA, April 2024. USENIX Association. 

- [38] Diogo Avelas, Hasan Heydari, Eduardo Alchieri, Tobias Distler, and Alysson Bessani. Probabilistic byzantine fault tolerance. In _Proceedings of the 43rd ACM Symposium on Principles of Distributed Computing_ , PODC ’24, page 170–181, New York, NY, USA, 2024. Association for Computing Machinery. 

- [39] Amazon Web Services. AWS Understanding Lambda function scaling. https://docs.aws.amazon.com/lambda/latest/dg/lambdaconcurrency.html, 2024. 

- [40] The Apache Software Foundation. Apache ZooKeeper. https:// zookeeper.apache.org/, 2025. 

- [41] OpenFaaS Ltd. OpenFaaS - Serverless Functions, Made Simple. https: //www.openfaas.com/, 2024. 

- [42] Python Software Foundation. socket — Low-level networking interface. https://docs.python.org/3/library/socket.html, 2024. 

- [43] Python Software Foundation. Miscellaneous operating system interfaces. https://docs.python.org/3/library/os.html#module-os, 2024. 

- [44] RDMA Core Userspace Libraries and Daemons. https://github.com/ linux-rdma/rdma-core, 2024. 

- [45] Redis Ltd. Redis: The open-source, in-memory data store. https:// redis.io/, 2024. 

- [46] Sadjad Fouladi, Riad S. Wahby, Brennan Shacklett, Karthikeyan Balasubramaniam, William Zeng, Rahul Bhalerao, Anirudh Sivaraman, George Porter, and Keith Winstein. Encoding, fast and slow: Lowlatency video processing using thousands of tiny threads. In Aditya Akella and Jon Howell, editors, _14th USENIX Symposium on Networked Systems Design and Implementation, NSDI 2017, Boston, MA, USA, March 27-29, 2017_ , pages 363–376. USENIX Association, 2017. 

- [47] Zhipeng Jia and Emmett Witchel. Nightcore: efficient and scalable serverless computing for latency-sensitive, interactive microservices. In _Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , ASPLOS ’21, page 152–166, New York, NY, USA, 2021. Association for Computing Machinery. 

- [48] Jana Iyengar, Martin Thomson, et al. Quic: A udp-based multiplexed and secure transport. In _RFC 9000_ . Internet Engineering Task Force (IETF) Fremont, CA, USA, 2021. 

- [49] Kaiyu Hou, Sen Lin, Yan Chen, and Vinod Yegneswaran. Qfaas: accelerating and securing serverless cloud networks with quic. In _Proceedings of the 13th Symposium on Cloud Computing_ , SoCC ’22, page 240–256, New York, NY, USA, 2022. Association for Computing Machinery. 

- [50] Xingda Wei, Fangming Lu, Rong Chen, and Haibo Chen. KRCORE: A microsecond-scale RDMA control plane for elastic computing. In _2022 USENIX Annual Technical Conference (USENIX ATC 22)_ , pages 121– 136, Carlsbad, CA, July 2022. USENIX Association. 

- [51] Minchen Yu, Tingjia Cao, Wei Wang, and Ruichuan Chen. Following the data, not the function: Rethinking function orchestration in serverless computing. In _20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23)_ , pages 1489–1504, Boston, MA, April 2023. USENIX Association. 

- [52] Adam Oliner and Jon Stearley. What supercomputers say: A study of five system logs. In _37th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN’07)_ , pages 575–584, 2007. 

- [53] Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu. Loghub: A large collection of system log datasets for ai-driven log analytics. In _2023 IEEE 34th International Symposium on Software Reliability Engineering (ISSRE)_ , pages 355–366, 2023. 

- [54] Amazon Web Services. AWS Step Functions: Visual workflows for distributed applications. https://aws.amazon.com/step-functions/, 2025. 

- [55] Microsoft. Durable Functions is an extension of Azure Functions that lets you write stateful functions in a serverless compute environment. https://docs.microsoft.com/en-us/azure/azure-functions/ durable/, 2025. 

- [56] Zijun Li, Chuhao Xu, Quan Chen, Jieru Zhao, Chen Chen, and Minyi Guo. Dataflower: Exploiting the data-flow paradigm for serverless workflow orchestration. In _Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 4_ , ASPLOS ’23, page 57–72, New York, NY, USA, 2024. Association for Computing Machinery. 

- [57] Ao Wang, Jingyuan Zhang, Xiaolong Ma, Ali Anwar, Lukas Rupprecht, Dimitrios Skourtis, Vasily Tarasov, Feng Yan, and Yue Cheng. InfiniCache: Exploiting ephemeral serverless functions to build a Cost-Effective memory cache. In _18th USENIX Conference on File and Storage Technologies (FAST 20)_ , pages 267–281, Santa Clara, CA, February 2020. USENIX Association. 

- [58] Francisco Romero, Gohar Irfan Chaudhry, Íñigo Goiri, Pragna Gopa, Paul Batum, Neeraja J. Yadwadkar, Rodrigo Fonseca, Christos Kozyrakis, and Ricardo Bianchini. Faa$t: A transparent auto-scaling 

15 

158 

cache for serverless applications. In _Proceedings of the ACM Symposium on Cloud Computing_ , SoCC ’21, page 122–137, New York, NY, USA, 2021. Association for Computing Machinery. 

- [59] Djob Mvondo, Mathieu Bacou, Kevin Nguetchouang, Lucien Ngale, Stéphane Pouget, Josiane Kouam, Renaud Lachaize, Jinho Hwang, Tim Wood, Daniel Hagimont, Noël De Palma, Bernabé Batchakui, and Alain Tchana. Ofc: an opportunistic caching system for faas platforms. In _Proceedings of the Sixteenth European Conference on Computer Systems_ , EuroSys ’21, page 228–244, New York, NY, USA, 2021. Association for Computing Machinery. 

- [60] Michal Wawrzoniak, Ingo Müller, Gustavo Alonso, and Rodrigo Bruno. Boxer: Data analytics on network-enabled serverless platforms. In _11th Conference on Innovative Data Systems Research, CIDR 2021, Virtual Event, January 11-15, 2021, Online Proceedings_ . www.cidrdb.org, 2021. 

- [61] Istemi Ekin Akkus, Ruichuan Chen, Ivica Rimac, Manuel Stein, Klaus Satzke, Andre Beck, Paarijaat Aditya, and Volker Hilt. SAND: Towards High-Performance serverless computing. In _2018 USENIX Annual Technical Conference (USENIX ATC 18)_ , pages 923–935, Boston, MA, July 2018. USENIX Association. 

- [62] Simon Shillaker and Peter Pietzuch. Faasm: Lightweight isolation for efficient stateful serverless computing. In _2020 USENIX Annual Technical Conference (USENIX ATC 20)_ , pages 419–433. USENIX Association, July 2020. 

- [63] Shixiong Qi, Leslie Monis, Ziteng Zeng, Ian-chin Wang, and K. K. Ramakrishnan. Spright: extracting the server from serverless computing! high-performance ebpf-based event-driven, shared-memory processing. In _Proceedings of the ACM SIGCOMM 2022 Conference_ , SIGCOMM ’22, page 780–794, New York, NY, USA, 2022. Association for Computing Machinery. 

16 

159 

