# *B. Configurations*

To determine the effectivenes of LRM-GPU, we evaluated the following configurations:

MCM-GPU [2]: MCM-GPU employs techniques such as L1.5 cache, first-touch page allocation, and distributed CTA scheduling to enhance the performance of multi-chiplet GPUs. Its synchronization mechanism is similar to that of traditional GPUs. During an acquire operation, the L1 and L1.5 caches are invalidated. It is necessary to ensure that dirty data is flushed back to the LLC when a release operation occurs. Since both the L1 and L1.5 caches adopt a write-through policy, stale data is guaranteed not to exist in the LLC, thus eliminating the need to flush the L1 and L1.5 caches. This principle also applies to other configurations.

hLRC [1]: It was originally designed for monolithic GPUs, we implemented it in GPGPU-Sim and extended its described mechanisms to multi-chiplet GPUs. hLRC tracks the registration locations of synchronization variables to exploit synchronization locality among SMs. It also caches synchronization variables in caches at all levels and employs a write-back policy for them. To ensure the consistency and atomicity of synchronization variables, whenever the location of a synchronization variable changes, the remotely cached synchronization variable must be written back. Until this write-back is completed, no other thread can access this synchronization variable. For relaxed atomic operations, the traditional GPU approach is still adopted.

HMG (NHCC)[43]: HMG is a state-of-the-art chiplet-based, multi-GPU coherence protocol. Although primarily designed for multi-GPU systems, we focused on its application in multi-chiplet GPUs and implemented it in GPGPU-Sim. HMG utilizes an SM-side LLC instead of an L1.5 cache. The LLC uses the write-through policy, which writes all data to the home node of the LLC and memory partition. Therefore, the home node always contains the latest data. Consequently, HMG employs an LLC coherence directory to track all data and maintain the LLC coherence, with each GPU chiplet having 12K entries, each entry covering four cache lines.

TABLE IV EVALUATED BENCHMARKS

| Microbenchmarks [48] (inputs: 8 512 2) |                     |                             |               |  |
|----------------------------------------|---------------------|-----------------------------|---------------|--|
| atomicTreeBarr                         | lfTreeBarr          | spinMutex                   | sleepMutex    |  |
| (ATB)                                  | (LTB)               | (SPM)                       | (SLM)         |  |
| faMutex                                | spinSem2            | spinSem10                   | spinSem120    |  |
| (FAM)                                  | (SPS2)              | (SPS10)                     | (SPS120)      |  |
|                                        |                     | with global synchronization |               |  |
| benchmarks                             | inputs              | benchmarks                  | inputs        |  |
| reduce(R)[9]                           | 8192 32             | scan(S)[9]                  | 16384         |  |
| histogram(H)[28] 262144 256            |                     | pagerank(P)[4]              | coAuthorsDBLP |  |
|                                        |                     |                             | .graph        |  |
| barnes–hut                             | 262144 4 0          | hash-table                  | 65536 2048    |  |
| (BH)[31]                               |                     | (HT)[51]                    |               |  |
| minimum spanning tree(MST)[31]         |                     | USA-road-d.NY.gr            |               |  |
| without global synchronization         |                     |                             |               |  |
| benchmarks                             | inputs              | benchmarks                  | inputs        |  |
| b+tree(BT)[5]                          | command.txt         | backprop(BP)[5]             | 65536         |  |
| bfs[5]                                 | graph65536.txt      | dwt2d[5]                    | 192.bmp       |  |
| nn[5]                                  | filelist 32         | lavaMD[5]                   | 10×10×10      |  |
| vgg16 fw(vgg)[33]–                     |                     | gpt2 fw(gpt)[32]            | –             |  |
| hotspot(HS)[5]                         | temp 512, power 512 |                             |               |  |

