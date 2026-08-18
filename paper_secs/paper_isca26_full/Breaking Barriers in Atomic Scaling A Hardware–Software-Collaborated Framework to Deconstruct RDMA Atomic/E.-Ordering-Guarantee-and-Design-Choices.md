# E. Ordering Guarantee and Design Choices

We further discuss the ordering guarantees of Fusa and elaborate on the corresponding design choices. The InfiniBand specification [3] mandates that all requests posted to a single QP must be executed by the RNIC strictly in the order in which they are issued by upper-layer applications.

While RNICs typically enforce request ordering by assigning each QP to a PU, Fusa breaks the original ordering semantics by replacing specific RDMA Atomic operations with RPCs, whose execution is shifted to the server CPU and can be performed asynchronously and independently of the requests executed by the PU of the RNIC. Figure 13 shows an example, for a send queue (SQ) with three requests {X, ATOMIC, Y} in ordered, Fusa-RPC converts the atomic operation to a pair of WRITE and RECV operations (Steps ① and ②) and generates an SQ' (with three requests {X, WRITE, Y}) and RQ' (with the request {RECV}) at the client side. After the conversion, the three requests in SQ' are order guaranteed,

where the request Y must be executed after the completion of WRITE; however, as the completion of WRITE only indicates that the the request has been delivered to the server's memory (not the completion of the atomic operation), the execution of Y might be started prior the completion of the atomic operation (Step ③), violating the original execution orders in SQ.

While Fusa might break the semantics of ordering guarantee, we also show that it is applicable for a wide spectrum of applications (including tree-based indexes [63], [65], hash-based indexes [50], [82], transaction systems [12], [69], and databases [9], [60]) without altering the correctness of their operation flows. The rationale is that in these applications, the next operation right after the atomic operation generally depends on the result after completing the atomic operation to decide the subsequent action (e.g., RACE [82] and Sherman [65] need the results of the atomic operation to decide whether to retry or perform a write) or generate the subsequent operations (e.g., PolarDB [9] needs to obtain the CTS timestamp using RDMA\_CAS for ordering subsequent log writes).

For the applications [44], [80] that need strict requirement of the execution ordering, we can also make slight modifications to Fusa and generate a variant of Fusa (named OrderedFusa) that provides ordering guarantees. Different from Fusa that allows the SQ' to continue executing subsequent requests while the RPC request for atomic operation is still pending, OrderedFusa blocks the execution of requests in QP until the RPC request completes. Figure 13 shows an example of OrderedFusa. When an RDMA Atomic operation is converted to an RPC request with a pair of WRITE and RECV operations (Steps 1) and 2), OrderedFusa will append an RDMA WAIT verb [34], [55], [58] right after the WRITE (Step 3) and configure it to be released only upon the completion of the client's pre-posted RECV (i.e., indicating that the atomic operation is completed by the server). After the RECV operation in RQ' is completed, OrderedFusa starts the subsequent requests of the SQ' afterwards (Steps 5). This design ensures that the atomic request replaced by Fusa-RPC is always executed before any subsequent requests on the same QP, thereby preserving per-QP linearizability. We also evaluate the performance of OrderedFusa and show that OrderedFusa still significantly outperforms the baseline RNIC-Only by achieving up to 2.5× higher throughput (see Exp#1 and Exp#2 in §VI-B).

## F. Generality of Fusa When PCIe Atomic is Enabled

While Fusa is primarily designed for environments where PCIe Atomic is disabled, we also evaluate its generality under PCIe Atomic-enabled configurations. A key constraint is the limited processing capacity of the *PCIe Atomic Completer Engine*, which achieves lower atomic throughput than the RNIC (Figure 3). In view of this, Fusa can proactively redistribute more atomic requests to the server-side CPU, thus achieving a balanced workload that fully utilizes both PCIe Atomic and CPU-executed atomic operations. This hybrid handling allows Fusa to effectively break the performance limits of PCIe Atomic and improve overall RDMA performance. We also

evaluate the performance of Fusa in PCIe Atomic-enabled scenarios in Exp#7 (§VI-D).

#### V. IMPLEMENTATION

We implement Fusa on the Mellanox RNIC by modifying the user-space driver to develop Fusa-Driver and Fusa-RPC. We also implement Fusa-Agent and Fusa-Server to identify contention and update the dispatch strategy.

Fusa-Driver: To implement Fusa-Driver, we modify two functions in the user-space driver: (i) mlx5\_create\_qp(): During QP creation, we allocate a dedicated memory region for each QP to log metadata related to the issued atomic requests. (ii) mlx5\_post\_send(): Before sending RDMA Atomic to the RNIC, we intercept and rewrite the requests based on the proactive dispatch strategy. Fusa-Driver maintains transparency to applications by leveraging the LD\_PRELOAD.

Fusa-Agent and Fusa-SHM: Fusa-Agent runs as an

independent control thread that periodically refines the dispatch strategy. The two components communicate through a shared memory region (Fusa-SHM), where Fusa-Driver records perrequest statistics and Fusa-Agent updates strategy bits. We configure the group number as 8,192 by default, since larger values introduce higher metadata overhead without providing measurable performance gains. We reserve 65 KB in Fusa-SHM to manage 8,192 groups, each consisting of a 64-bit counter (split into a 32-bit field for per-second request counts and a 32-bit field for in-flight request counts) and a one-bit flag indicating whether the group requests are dispatched to serverside RNIC ('0') or CPU ('1'). Each QP maintains a 64-bit state (1-bit running state and 63-bit epoch) for client consensus. Each client uses at most 32 QPs (32 threads), incurring a maximum QP metadata overhead of 256 B. We embed the 13-bit group\_id  $(log_2(8192) = 13)$  of each request into the Work Request ID (WR\_ID) [42] of the atomic operation. When Fusa-Driver polls the CQE, it extracts the group\_id from the WR\_ID and decrements the in-flight counter, thereby enabling accurate tracking of in-flight RNIC Atomic requests. Fusa-Server: To provide a global view for contention identification and dispatch strategy selection, we propose Fusa-Server. Fusa-Server aggregates contention statistics from all clients and analyzes them to determine an appropriate dispatch strategy for the next stage. It then broadcasts the updated strategy to all clients, after which each Fusa-Agent updates its local dispatch strategy recorded in Fusa-SHM. Shorter stage improve timeliness but increase consensus overhead; Fusa chooses 1 second as a balanced setting. To prevent concurrent access to the same memory region by both the RNIC and the CPU, we employ a reject mechanism that enforces strategy consensus across clients (e.g., 1) in Figure 10). The server attaches a reject flag to the returned message to indicate that

the request has been denied. On the client side, the user-

space library detects this flag and automatically retransmit the

message, without requiring any awareness from the application.

Table I: General Workloads.

| Workload                  | Update | Read |
|---------------------------|--------|------|
| update-intensive (YCSB-A) | 50%    | 50%  |
| update-heavy (U40R60)     | 40%    | 60%  |
| read-heavy (U30R70)       | 30%    | 70%  |
| read-intensive (U20R80)   | 20%    | 80%  |
| read-dominant (YCSB-B)    | 5%     | 95%  |

## VI. EVALUATION

We evaluate Fusa using microbenchmarks and representative RDMA-based systems. Our objective is to seek the answers for the following questions:

- How do the design techniques impact the end-to-end performance of Fusa? (Exp#1–#2 in §VI-B)
- How much performance penalty does OrderedFusa incur to provide ordering guarantees compared to Fusa? (Exp#1–#2 in §VI-B)
- How does Fusa perform when serving multiple concurrent workloads? (Exp#3 in §VI-C)
- Can Fusa adapt to workload changes when the access hotspots shift? (Exp#4 in §VI-C)
- How do different system configurations influence the performance of Fusa? (Exp#5–#6 in §VI-D)
- What performance benefits does Fusa provide when PCIe Atomic is enabled? (Exp#7 in §VI-D)
- How do upper-layer RDMA-based systems benefit from integrating Fusa? (Exp#8-#9 in §VI-E)

# E. Ordering Guarantee and Design Choices

We further discuss the ordering guarantees of Fusa and elaborate on the corresponding design choices. The InfiniBand specification [3] mandates that all requests posted to a single QP must be executed by the RNIC strictly in the order in which they are issued by upper-layer applications.

While RNICs typically enforce request ordering by assigning each QP to a PU, Fusa breaks the original ordering semantics by replacing specific RDMA Atomic operations with RPCs, whose execution is shifted to the server CPU and can be performed asynchronously and independently of the requests executed by the PU of the RNIC. Figure 13 shows an example, for a send queue (SQ) with three requests {X, ATOMIC, Y} in ordered, Fusa-RPC converts the atomic operation to a pair of WRITE and RECV operations (Steps ① and ②) and generates an SQ' (with three requests {X, WRITE, Y}) and RQ' (with the request {RECV}) at the client side. After the conversion, the three requests in SQ' are order guaranteed,

where the request Y must be executed after the completion of WRITE; however, as the completion of WRITE only indicates that the the request has been delivered to the server's memory (not the completion of the atomic operation), the execution of Y might be started prior the completion of the atomic operation (Step ③), violating the original execution orders in SQ.

While Fusa might break the semantics of ordering guarantee, we also show that it is applicable for a wide spectrum of applications (including tree-based indexes [63], [65], hash-based indexes [50], [82], transaction systems [12], [69], and databases [9], [60]) without altering the correctness of their operation flows. The rationale is that in these applications, the next operation right after the atomic operation generally depends on the result after completing the atomic operation to decide the subsequent action (e.g., RACE [82] and Sherman [65] need the results of the atomic operation to decide whether to retry or perform a write) or generate the subsequent operations (e.g., PolarDB [9] needs to obtain the CTS timestamp using RDMA\_CAS for ordering subsequent log writes).

For the applications [44], [80] that need strict requirement of the execution ordering, we can also make slight modifications to Fusa and generate a variant of Fusa (named OrderedFusa) that provides ordering guarantees. Different from Fusa that allows the SQ' to continue executing subsequent requests while the RPC request for atomic operation is still pending, OrderedFusa blocks the execution of requests in QP until the RPC request completes. Figure 13 shows an example of OrderedFusa. When an RDMA Atomic operation is converted to an RPC request with a pair of WRITE and RECV operations (Steps 1) and 2), OrderedFusa will append an RDMA WAIT verb [34], [55], [58] right after the WRITE (Step 3) and configure it to be released only upon the completion of the client's pre-posted RECV (i.e., indicating that the atomic operation is completed by the server). After the RECV operation in RQ' is completed, OrderedFusa starts the subsequent requests of the SQ' afterwards (Steps 5). This design ensures that the atomic request replaced by Fusa-RPC is always executed before any subsequent requests on the same QP, thereby preserving per-QP linearizability. We also evaluate the performance of OrderedFusa and show that OrderedFusa still significantly outperforms the baseline RNIC-Only by achieving up to 2.5× higher throughput (see Exp#1 and Exp#2 in §VI-B).

## F. Generality of Fusa When PCIe Atomic is Enabled

While Fusa is primarily designed for environments where PCIe Atomic is disabled, we also evaluate its generality under PCIe Atomic-enabled configurations. A key constraint is the limited processing capacity of the *PCIe Atomic Completer Engine*, which achieves lower atomic throughput than the RNIC (Figure 3). In view of this, Fusa can proactively redistribute more atomic requests to the server-side CPU, thus achieving a balanced workload that fully utilizes both PCIe Atomic and CPU-executed atomic operations. This hybrid handling allows Fusa to effectively break the performance limits of PCIe Atomic and improve overall RDMA performance. We also

evaluate the performance of Fusa in PCIe Atomic-enabled scenarios in Exp#7 (§VI-D).

#### V. IMPLEMENTATION

We implement Fusa on the Mellanox RNIC by modifying the user-space driver to develop Fusa-Driver and Fusa-RPC. We also implement Fusa-Agent and Fusa-Server to identify contention and update the dispatch strategy.

Fusa-Driver: To implement Fusa-Driver, we modify two functions in the user-space driver: (i) mlx5\_create\_qp(): During QP creation, we allocate a dedicated memory region for each QP to log metadata related to the issued atomic requests. (ii) mlx5\_post\_send(): Before sending RDMA Atomic to the RNIC, we intercept and rewrite the requests based on the proactive dispatch strategy. Fusa-Driver maintains transparency to applications by leveraging the LD\_PRELOAD.

Fusa-Agent and Fusa-SHM: Fusa-Agent runs as an

independent control thread that periodically refines the dispatch strategy. The two components communicate through a shared memory region (Fusa-SHM), where Fusa-Driver records perrequest statistics and Fusa-Agent updates strategy bits. We configure the group number as 8,192 by default, since larger values introduce higher metadata overhead without providing measurable performance gains. We reserve 65 KB in Fusa-SHM to manage 8,192 groups, each consisting of a 64-bit counter (split into a 32-bit field for per-second request counts and a 32-bit field for in-flight request counts) and a one-bit flag indicating whether the group requests are dispatched to serverside RNIC ('0') or CPU ('1'). Each QP maintains a 64-bit state (1-bit running state and 63-bit epoch) for client consensus. Each client uses at most 32 QPs (32 threads), incurring a maximum QP metadata overhead of 256 B. We embed the 13-bit group\_id  $(log_2(8192) = 13)$  of each request into the Work Request ID (WR\_ID) [42] of the atomic operation. When Fusa-Driver polls the CQE, it extracts the group\_id from the WR\_ID and decrements the in-flight counter, thereby enabling accurate tracking of in-flight RNIC Atomic requests. Fusa-Server: To provide a global view for contention identification and dispatch strategy selection, we propose Fusa-Server. Fusa-Server aggregates contention statistics from all clients and analyzes them to determine an appropriate dispatch strategy for the next stage. It then broadcasts the updated strategy to all clients, after which each Fusa-Agent updates its local dispatch strategy recorded in Fusa-SHM. Shorter stage improve timeliness but increase consensus overhead; Fusa chooses 1 second as a balanced setting. To prevent concurrent access to the same memory region by both the RNIC and the CPU, we employ a reject mechanism that enforces strategy consensus across clients (e.g., 1) in Figure 10). The server attaches a reject flag to the returned message to indicate that

the request has been denied. On the client side, the user-

space library detects this flag and automatically retransmit the

message, without requiring any awareness from the application.

Table I: General Workloads.

| Workload                  | Update | Read |
|---------------------------|--------|------|
| update-intensive (YCSB-A) | 50%    | 50%  |
| update-heavy (U40R60)     | 40%    | 60%  |
| read-heavy (U30R70)       | 30%    | 70%  |
| read-intensive (U20R80)   | 20%    | 80%  |
| read-dominant (YCSB-B)    | 5%     | 95%  |

## VI. EVALUATION

We evaluate Fusa using microbenchmarks and representative RDMA-based systems. Our objective is to seek the answers for the following questions:

- How do the design techniques impact the end-to-end performance of Fusa? (Exp#1–#2 in §VI-B)
- How much performance penalty does OrderedFusa incur to provide ordering guarantees compared to Fusa? (Exp#1–#2 in §VI-B)
- How does Fusa perform when serving multiple concurrent workloads? (Exp#3 in §VI-C)
- Can Fusa adapt to workload changes when the access hotspots shift? (Exp#4 in §VI-C)
- How do different system configurations influence the performance of Fusa? (Exp#5–#6 in §VI-D)
- What performance benefits does Fusa provide when PCIe Atomic is enabled? (Exp#7 in §VI-D)
- How do upper-layer RDMA-based systems benefit from integrating Fusa? (Exp#8-#9 in §VI-E)

