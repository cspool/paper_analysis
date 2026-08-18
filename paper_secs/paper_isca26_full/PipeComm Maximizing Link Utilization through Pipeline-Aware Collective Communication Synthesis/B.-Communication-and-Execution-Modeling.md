# *B. Communication and Execution Modeling*

α-β model [19] is a widely used foundational abstraction in parallel and distributed computing for estimating communication costs. This model characterizes the time required to transfer a message of size N as Tcomm = α+β ×N, where α

TABLE I COMPARISON OF EXISTING FRAMEWORKS, WHERE DT DENOTES SUPPORT FOR DISTINCT BANDWIDTH, AS DENOTES SUPPORT FOR ASYMMETRIC CONNECTIVITY, AND SC DENOTES SCALABILITY

| Synthesis      | Topology |       | Collective |                 | Algorithm |                      |
|----------------|----------|-------|------------|-----------------|-----------|----------------------|
| Framework      | DT       | AS    |            | A2A Pipeline SC |           | Strategy             |
| SCCL [5]       | ✘        | ✔     | ✔          | ✘               | ✘         | Solver-based         |
| Blink [60]     | ✘        | ✔     | ✘          | Limit           | ✘         | Solver-based         |
| TACCL [51]     | ✔        | Limit | ✔          | ✘               | ✘         | Solver-based         |
| TE-CCL [31]    | ✔        | Limit | ✔          | ✘               | ✔         | Solver-based         |
| Themis [46]    | Limit    | ✘     | ✘          | ✘               | ✔         | Composition          |
| MultiTree [20] | ✘        | ✔     | ✘          | ✘               | ✔         | Heuristic            |
| TTO [27]       | ✘        | ✘     | ✘          | Limit           | ✔         | Heuristic            |
| TACOS [63]     | Limit    | ✔     | ✘          | ✘               | ✔         | Heuristic            |
| PipeComm       | ✔        | ✔     | ✔          | ✔               | ✔         | Solver / Incremental |

represents the fixed startup latency to initiate a communication, and β captures the inverse of bandwidth, or the per-word transfer time. Due to its simplicity and effectiveness, the αβ model is commonly used for analyzing and optimizing collective communication [21], [55].

Software pipelining [26] enables instruction-level parallelism by overlapping execution stages. Inspired by this, we apply pipelining to communication to improve bandwidth utilization. Multiple communications are initiated at regular intervals, allowing them to proceed concurrently across different stages of execution. A key metric in this model is the Initiation Interval (II), defined as the number of time steps between the start of two consecutive iterations. Given a total of N data chunks to be communicated and a pipeline depth of D, the overall communication latency step T can be caculated as: T = D + (N −1) ∗ II. By adopting this pipelined abstraction, we can effectively model and optimize collective algorithms in heterogeneous systems, leading to improved communication efficiency and scalability.

