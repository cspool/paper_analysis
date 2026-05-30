# *CCS Concepts:* • Networks → Cloud Computing.

**Keywords:** serverless computing

#### **ACM Reference Format:**

Yuhan Deng, Akshay Srivatsan, Sebastian Ingino, Francis Chua, Yasmine Mitchell, Matthew Vilaysack, Keith Winstein. 2026. Fix: externalizing network I/O in serverless computing. In 21st European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland, UK. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3767295.3769387

#### 1 Introduction

For a decade, cloud-computing operators have offered "serverless" function-as-a-service products. These systems let users upload functions to be invoked on request. When this happens, the function is allocated a slice of a physical machine's RAM, CPU, and NIC, and the customer is billed for the time until it finishes [1, 2]. In practice, cloud functions are typically used for asynchronous services where each invocation runs independently, but researchers have also explored their use for large jobs that launch thousands of parallel invocations working together with complex dataflow: video processing [18], linear algebra [25, 39], software compilation and testing [16], theorem proving [43], 3D rendering [17], ML training [24], data analysis [26], sorting [30], etc.

![](_page_0_Figure_15.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

EUROSYS '26, Edinburgh, Scotland, UK
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2212-7/26/04
https://doi.org/10.1145/3767295.3769387

Despite this interest, effective use of serverless computing remains elusive. In this paper, we argue that a root cause is an *underconstrained notion of networked computation*, one where the I/O and dependencies of user functions are opaque to the platform. Consider a common serverless application: a cloud function that resizes an image [38]. A user creates the function by uploading a piece of code—call this f. When the function is invoked, the provider finds a physical server with enough available RAM and cores, transfers and unpacks the code if not already present, claims a slice of RAM, and runs the function, generally as a Linux process in a pre-warmed VM. After seeing the invocation payload (an HTTP request or other event), the function requests the image file x from network storage, e.g. Amazon S3.

From the user's point of view, the invocation was always meant to compute f(x) (the resized image), but from the provider's perspective, f is a running Linux process, and its dependency on x wasn't known until after the code was placed and running. If S3 has cached x nearby, the retrieval happens quickly. Otherwise, the function will wait, occupying and mostly idling its slice of RAM until retrieval finishes.

For computations that are short relative to network and storage latencies [29, 30], limitations of this service model can be significant. If the user had been able to express that the invocation represented "f(x)" in a way the provider understood, the provider might have attempted a better strategy to place or schedule it, e.g.:

- simultaneously transfer *f* (the code) and *x* (the image) to the server so the task can finish sooner,
- wait to allocate f's RAM and CPU until x arrives, letting other functions run there in the meantime,
- instead of starting f on a server chosen without regard to x, choose a server close to whichever dependency (f or x) is bigger,
- delay the invocation, hoping to aggregate multiple tasks that depend on *x* to run in a batch on one server,
- if f(x) is part of a pipeline g(f(x)), and if y = f(x) is hinted to be large, then transfer f and x close to their downstream dependency g, and run f(x) on that server before running g on the result—avoiding the need to transfer y over the network,
- or if x can be computed deterministically by h(z), then if easier, fetch h and z and recompute x instead of transferring it.

In many cases, these strategies could improve the job throughputs, latencies, RAM and CPU utilization, and perhaps costs of serverless platforms. But they probably aren't feasible today, even for an image-resizing function of one input, because f's dataflow was "internal": it fetched x by opening a socket, sending a request, and receiving arbitrary data. Even if x came from the provider's own storage service, the provider didn't observe the dependency until after the task was placed and running. For the sorts of jobs surveyed in the first paragraph, jobs that launch thousands of parallel invocations with complex dataflow among them, the need for good placement and scheduling will be even greater.

This paper presents FIX, an architecture for serverless computing that *externalizes* I/O, making application dataflow visible to, and performed by, the underlying networked system. In FIX, function invocations are described in a low-level ABI (application binary interface) that specifies a sealed container where execution occurs, containing dependencies that are addressed in a way the program and provider both understand—maybe as the output of another invocation.

In Fix, programs can choose to capture only the minimum data needed to make progress at each step of a larger job. The underlying platform uses its visibility and flexibility to place and schedule tasks and transfers to reduce starvation and use of the network, e.g. via the strategies above.

This paper's main contribution is in Fix's design and the demonstration that I/O externalization, with the ability to express precise and dynamic data-dependencies with little overhead, can boost performance and efficiency. Fix is a realization of I/O-compute separation [11, 14, 27, 28, 34] as well a mechanism for programs to provide the platform with visibility—perhaps partial visibility, refined as computation proceeds—into future data- and control flow. Fix does this in a declarative way that can be parsed anywhere, avoiding round-trips to a scheduler when invoking a new task.

Fix's design and implementation have a number of mutually reinforcing characteristics that lead to efficient execution. Fix's invocations are concisely described in a packed binary format designed to minimize runtime overhead. We implemented a runtime for Fix, called Fixpoint, that has a per-invocation overhead of about 1.5 µs. This means that applications can afford to use fine-grained containers that capture only data needed to make progress at each stage. Minimizing the data "footprint" of each invocation helps Fixpoint reduce cold-start times and optimize the scheduling and utilization of CPUs, RAM, and the network.

Fix has significant limitations. It represents a constrained model of computation: to describe each task in a placement-agnostic way, invocations must be of pure functions applied to content-addressed data or to the outputs of other invocations. Functions can't access data outside the container. At least at present, Fix doesn't support calls to nondeterministic services e.g., clocks, true random number generators, multi-user databases, or arbitrary Web APIs. Fix is its own ABI and doesn't run Linux executables; it runs some POSIX programs (e.g. CPython, clang) but we had to recompile them with a

Fix-targeting toolchain to achieve this. We haven't measured Fix's ease of use or effect on developer productivity.

**Summary of results.** We found that Fix's approach can unlock significant advantages in performance and efficiency (as well as reproducibility and reliability, aspects we did not evaluate quantitatively). We evaluated several applications run on Fixpoint, compared with OpenWhisk, MinIO, and Kubernetes (open-source analogs of AWS Lambda and S3), Pheromone [46], and Ray [32]; full results are in Section 5.

FIXPOINT creates hermetic containers without spawning OS processes, by requiring that functions be converted ahead-of-time to safe machine code. This results in lower overhead than systems based on Linux containers (OpenWhisk) or higher-level programming languages (Ray). To invoke a trivial function that adds two 8-bit integers, FIXPOINT's containers show lower overhead (fig. 7a):

| Approach         | Time              | slowdown vs. Fix |
|------------------|-------------------|------------------|
| Fix              | 1.46 µs           | 1×               |
| Linux vfork+exec | 449 µs            | 307×             |
| Pheromone        | $1.05\mathrm{ms}$ | 720×             |
| Ray              | $1.29\mathrm{ms}$ | 881×             |
| Faasm            | 10.6 ms           | 7,260×           |
| OpenWhisk        | 30.7 ms           | 20,980×          |

In a different experiment, we used Linux's CPU-state statistics to measure how much of these gains come from avoiding starvation—by co-scheduling computations and transfers, and waiting to allocate CPU and RAM until dependencies have arrived. We wrote a program to count non-overlapping strings in a 96 GiB dataset from Wikipedia and ran it on a 320-core, 10-node cluster. Fix's approach avoids a substantial amount of CPU starvation (fig. 8b):

| Approach                  | Time   | CPU waiting %         |
|---------------------------|--------|-----------------------|
| **                        |        | (idle + iowait + irq) |
| Fix                       | 3.25 s | 37%                   |
| FIX (with "internal" I/O) | 33.8 s | 92%                   |
| OpenWhisk + MinIO + K8s   | 63.9 s | 92%                   |

Finally, we implemented a key-value store represented on disk as a B+-tree, using Fix and two other approaches. Each version traverses the B+-tree node-by-node to retrieve the value corresponding to a key. As we decrease the maximum number of children of each B+-tree node, this process results in a smaller memory footprint and total amount of data accessed, at the cost of more function invocations. Compared with Ray, Fix's semantics let users benefit from breaking down programs with fine granularity (fig. 9):

| Approach (B+-tree of arity 256)            | Time   | slowdown vs. Fix |
|--------------------------------------------|--------|------------------|
| Fix                                        | 0.14 s | 1×               |
| Ray                                        | 2.8 s  | 19.6×            |
| Ray (broken into fine-grained invocations) | 5.74 s | $40\times$       |

Fix represents a fundamentally different approach to outsourced computing: one that's more constraining and probably more difficult to program for, but ultimately advantageous for customers (whose jobs run faster) and providers (whose infrastructure is used more efficiently). Current service abstractions represent something of a "pay-for-effort" system—by billing customers for each millisecond that a function occupies a machine slice, idle or not, providers aren't directly incentivized to improve scheduling and placement. Even if a provider wanted to do this, current systems lack the visibility into application dataflow to do it well. Fix's approach suggests a shift towards "pay-for-results": computations described in a way that permits providers to innovate in the placement and scheduling of computation and I/O, so long as they arrive at the correct answers.

This paper proceeds as follows. In section 2, we discuss the substantial context of related work across several areas. We describe Fix's design (sec. 3) and its implementation in the Fixpoint runtime (sec. 4). We report our evaluation in section 5, finishing with limitations (sec. 6) and a conclusion.

#### <span id="page-2-0"></span>2 Related work

Fix relates to prior work across workflow orchestration (Hadoop [42], Spark [47], etc.), techniques that optimize serverless platforms with lightweight containers for dense packability or locality hints, tools that run highly parallel workloads on current function-as-a-service platforms, containerization and execution systems (Docker [31], NixOS [15], etc.), and content-addressed storage. We discuss how Fix relates to this prior literature in several areas.

Cluster orchestration systems. Cluster orchestration systems like Spark [47], Dryad [22], CIEL [33] and Ray [32] allow programmers to express applications as a group of tasks, and orchestrate execution of the tasks across a cluster. Task interdependencies can be represented at runtime as a static DAG (Spark and Dryad) or dynamic task graph (CIEL and Ray). These systems generally employ language-level mechanisms: users spawn tasks using domain-specific languages (CIEL and Dryad), or with a pre-existing programming language (Python for Ray, Scala for Spark, etc.)

Fix's computation model represents interdependencies in a dynamic graph (similar to CIEL or Ray), in a somewhat more general sense: Fix's invocations describe all data-dependencies that code will have access to; Fix can capture subselections of existing data objects and the relationships between application data structures (e.g. the relationships between nodes in a B+-tree); this kind of dataflow can't generally be exposed to the runtime by current systems.

Fix enforces I/O externalization: all data-dependencies that code need access to must be made explicit. This allows it to freely schedule tasks at different execution locations. In comparison, existing systems allow programs to states not captured by the computation representation, such as local

file system. This makes Fix more amenable to outsourcing computations to cloud services.

Previous work relies on runtime infrastructure to track dependency information at a centralized scheduler or a designated physical node. In contrast, Fix unifies the description of data flow—inputs and outputs of invocations—with control flow—which function should be invoked with the results of another—in a single serializable format. Dependency information is shipped with data defining a function, avoiding round-trips. This leads to lower dependency-resolution overhead that allows finer-grained function invocations.

Scheduling and containers for serverless platforms. Much prior work is aimed at optimizing the performance of function-as-a-service platforms with conventional architectures for applications with interdependent workflows. This includes adding long-living caches beyond individual function invocations [35], and providing locality hints [10] to for better placement decisions. The line of work most similar to Fix is workflow-based serverless systems [37, 46] with a static function dependency model, e.g. the outputs of a function f are always consumed by another function q. Fix represents data-dependencies in a richer way at a finer-grained per-invocation level. Prior work has proposed the model of I/O-compute separation [14] and realization of the model [27, 28] that targets at better elasticity for spiky serverless workloads. Fix focus on designing the abstraction and mechanism for representing computational workloads in a I/O-compute separation way.

Another line of work designs lightweight containers to allow denser packability, such as Firecracker [5], Virtine [41], AlloyStack [45], Junction [19], Faasm [40], and WasmBox-C/wasm2c [49]. As part of our work on Fix, we became significant contributors to and maintainers of the wasm2c codebase; Fix's toolchain includes this tool.

Massively burst-parallel applications. There has been considerable interest in using serverless platforms for short-lived, large-scale, highly parallel jobs, including video processing [18], linear algebra [25, 39], software compilation and testing [16], theorem proving [43], 3D rendering [17], ML training [24], data analysis [12, 26], etc. Fix aims to be a better platform for these kinds of applications.

Build environments and content-addressed storage. Fix's computation-addressed dependencies for user programs resemble execution-environment languages like Docker [31], NixOS [15], or Spark [47] (discussed above). Fix's binary representation of dependencies draws inspiration from content-addressed systems such as Git [6], Bittorrent [4], Named Data Networking [23], and IPFS [7].

