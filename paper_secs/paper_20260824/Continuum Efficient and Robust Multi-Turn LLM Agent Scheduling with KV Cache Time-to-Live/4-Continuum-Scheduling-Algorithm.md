# 4 Continuum Scheduling Algorithm

Given the failure of previous work, we identify the key question in serving agentic workloads: How to efficiently and robustly retain KV cache in multi-turn scenarios?

An optimal KV cache retention policy should include the following features:

- It should retain KV cache for requests that will reuse them soon after tool calls, minimizing prefill/loading overheads.
- It should consider the multi-turn continuity of agent programs, reducing waiting and preserving program order.
- It should be robust to varying tool call latencies.

In order to achieve the robustness guarantee, we propose to borrow the idea of Time-to-live (TTL) from traditional systems: for each request's KV cache, we give a TTL value to define the maximum duration for it to remain in GPU memory. This prevents long-running or failed tool calls from blocking GPU resources indefinitely while retaining KV cache.

However, setting appropriate TTL values for each KV cache entry is challenging compared with static preserve operations. First, the TTL value should not be too large. If the timeout duration is too long as shown in Figure [6,](#page-4-0) the pinned KV cache occupies GPU memory unnecessarily, blocking other requests and reducing overall system throughput. On the other hand, if the pin time for the specific KV cache is too short, the KV cache is evicted before the tool call completes, still causing expensive recomputation or scheduling bubble despite wasted GPU occupation time.

Given these tradeoffs, the TTL value should be set carefully. Only if we can set appropriate TTL values based on based on tool call durations, prefill/loading costs, and the measurement to program continuity, we can balance the benefit of cache

Algorithm 1: Continuum's Scheduling Algorithm

<span id="page-4-1"></span>Global state :waiting queue *Q*; TTL map *P* (records pinned programs and their TTLs); historical tool-call records *S*, where *S*[ *f* ] denotes the recorded tool-call information for tool *f*

```
1 Function OnRequestArrive(request r):
2 Q ← Q∪ {r}, id ← Program ID of r;
3 If id is a seen program then
4 (f,t) ← Tool-call information from r;
5 S[ f ] ← S[ f ]∪ {t};
6 Function OnRequestFinish(request r):
7 If r is the last request of its program then
8 Free KV cache used by r;
9 else
10 f ← Next tool to be called after finishing r;
11 id ← Program ID of r;
12 P[id] ← CalcTTL(r,S[ f ]);
13 Function Schedule():
14 While Q is not empty do
15 For each id in P.keys do
16 If current time > P[id] and id ∈/ Q.programs
             then
17 Free KV cache used by id's last request;
18 P ← P\ (id,P[id]);
19 r ← argmaxr
                   ′∈Q CalcPriority(r
                                  ′
                                  ,P);
20 If r cannot fit into memory then
21 break;
22 else
23 Q ← Q\ {r};
24 Issue r to running;
25 id ← Program ID of r;
26 If id ∈ P.keys then P ← P\ (id,P[id]) ;
```

reuse against the need to maintain system throughput for other requests to achieve good performance.

