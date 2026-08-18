# <span id="page-7-2"></span>*D. Efficient Many-Sharer Directory Entries*

From past work [\[22\]](#page-13-4), it is known that a storage-efficient directory design needs to support only a few sharers for most of the lines, and many sharers for very few lines. To support such a design, Dorado includes a new scheme that we call *Per-Set Overflow Pointers* (*SetOverflow*). As we describe the scheme, directory refers to both TD and ED.

In *SetOverflow*, each set of a set-associative directory has a structure with sharer pointers (*PointerSpace*) that can be used by one or more of the ways in the set when such ways overflow their own set of pointers. Figure [9a](#page-7-1) shows one set of a W-way set-associative directory with SetOverflow. It shows the directory of each way (which includes two sharer pointers) and the PointerSpace. The PointerSpace has two arrays: *SharerPointer* and *OwnerWay*. SharerPointer has the overflow pointers; OwnerWay has the IDs of the ways that are using those pointers. To save space, each of the T<sup>1</sup> entries in OwnerWay owns T<sup>2</sup> consecutive entries in SharerPointer: e.g., inserting a way ID in the first entry of OwnerWay allows that way to use the first T<sup>2</sup> entries in SharerPointer, and similarly for the other entries.

In a directory set, when one of the ways runs out of pointers (i.e, the two pointers in Figure [9a](#page-7-1)), it sets its *Overflow into PointerSpace* (O) bit and uses pointers from the PointerSpace. To do so, the way claims an empty entry in OwnerWay, where it puts its way ID. This allows it to use T<sup>2</sup> entries in SharerPointer (starting at offset T1xT2). If the way needs

![](_page_7_Figure_7.jpeg)

<span id="page-7-1"></span>Fig. 9. *SetOverflow* design.

more pointers, it can continue to claim more OwnerWay (and SharerPointer) entries. Since multiple ways can claim entries in the PointerSpace, it is possible that a way requesting pointers finds that all the pointers in PointerSpace are in use. In this case, that way sets its *Broadcast* (B) bit, clears its O bit, and releases all of its pointers in PointerSpace.

Overall, the SetOverflow design allows one or more "manysharer" directory entries in the same set to grow into the PointerSpace with minimal performance impact. Indeed, the PointerSpace is not accessed if the O bit of the directory entry is clear. This occurs when the entry is: (1) in state M or E, or (2) in state S with few sharers (e.g., with at most two sharers as shown in the figure, which is enough to capture migratory sharing [\[60\]](#page-14-12)), or (3) in state S with many sharers and the B bit set. PointerSpace is only accessed on a read or write to a line in state S with the O bit set or about to be set. On a read, the access to the PointerSpace is not in the critical path of sending the line to the requester, since the line is sent first. On a write to a line with the O bit set, the access to the PointerSpace is in the critical path, since the transaction needs to identify all the sharers.

Figure [9b](#page-7-1) shows the operation performed on PointerSpace on such a write. We need to collect all the pointers owned by the target entry to send invalidations. The hardware accesses all the T<sup>1</sup> OwnerWay entries in parallel and compares them to the target way ID. On a match, the corresponding T<sup>2</sup> SharerPointer entries are read out. Then, all the read-out pointers are accumulated into a final array, so that invalidations can be sent. To perform these hardware operations, we add 3 cycles to the directory operation in this case. In the background, the hardware clears the corresponding entries in PointerSpace and the O bit of the way.

SetOverflow is different than Way Combining [\[64\]](#page-14-18), which only allows a line to steal *all* the sharer pointers of an *unused* line in the same cache set. Moreover, these sharer pointers must be returned when a new line is inserted into that cache entry. In contrast, SetOverflow enables fine-grained assignment of the extra sharer pointers to different lines in the set, and does not need cache line entries to be unused to work.

To validate the correctness of the Dorado protocol, we formally specify it in TLA+ [\[34\]](#page-13-29) and model-check it using TLC [\[34\]](#page-13-29). The TLA+ specification models a finite collection of clusters, cores, directory slices, and cache lines. Cache states are modeled as {M, E, S, I, MS}. Each directory entry encodes a sharer set, a dirty bit, and a home type (Global or Temporary). For every line, the model maintains a Global home and a (possibly empty) set of Temporary homes. Pointer types (LLptr, LRptr, RLptr) are modeled symbolically.

The Global home is modeled as the serialization point for conflicting transactions. Memory operations modeled include read/write hits/misses, invalidations, downgrades, home allocation/deallocation, cache/directory entry eviction, and overflow pointer allocation/reclamation. Multi-step memory operations are decomposed into ordered transitions.

To keep the state space tractable, we parameterize the model with 4 clusters, 4 cores per cluster, and 4 lines per core. Even such a small instantiation admits thousands of interleavings that cover concurrent cross-cluster writes, read–write races, and many corner cases.

Verification of safety and liveness. We verify five properties. The first one enforces *single-writer multiple-reader semantics*, namely that: 1) at most one cache may hold a line in state M, 2) if a cache holds an M line, no other cache can hold it in any state but I, and 3) if a cache holds an MS line, no other cache can hold it in any state but MS or I. The second property enforces *dirty-bit consistency*, ensuring that if a cache holds a line in M or MS, the line has a directory entry in the Global home and at most in the local Temporary home (if they are not the same), and both have the dirty bit set. The third property enforces *home consistency*: every Temporary home must be reflected in the metadata of the corresponding Global home. The fourth property guarantees *sharer soundness*: every single line in any cache has a valid entry in at least one directory. The fifth property is *read correctness*: a read to a memory location must return the value of the latest update to that location.

Verification of the absence of deadlock and livelock. Deadlock freedom requires that no reachable global state exists in which no transition is possible. Livelock freedom ensures that a memory request cannot be indefinitely postponed by cyclic interference among transactions. These properties rely on the invariant that any transaction that acquires the Global home entry cannot be preempted or permanently delayed by conflicting transactions to the same line that have not yet acquired the Global home. In particular, a transaction that may change the state of the Global home cannot lock any resource in a Temporary home until the transaction has reached and successfully locked the entry in the Global home.

TLC reports no invariant violations or deadlocks. Several corner cases are exercised. In concurrent write races from different clusters, model checking confirms that exactly one transaction acquires the Global home first, while the others observe the updated sharer metadata. In scenarios where a Temporary home exists and a transaction needs to modify the

TABLE VII ARCHITECTURAL PARAMETERS USED IN THE EVALUATION.

<span id="page-8-1"></span>

| Processor Parameters |                                                     |  |
|----------------------|-----------------------------------------------------|--|
| Package              | 1024 6-issue OoO cores, 352-entry ROB, 3GHz         |  |
| Clusters             | 32 clusters of 32 cores each                        |  |
| Sharer ptr size      | LLptr, LRptr, RLptr: 5b for ID + 1b for type        |  |
| L1 D/I caches        | 64/32KB, 8-way, 4 cyc. round trip (RT), 64B line    |  |
| L2 cache             | 2MB, 16-way, 16 cycles RT, 80 MSHRs                 |  |
| L3 cache             | Slice: 6MB/core, 12-way, 60 cyc. RT, 160 MSHRs      |  |
| L1 D/I TLBs          | 256/128 entries, 4-way, 2 cycles RT                 |  |
| L2 TLB               | 2048 entries, 12-way, 12 cycles RT                  |  |
| Page translation     | 4-level radix page tables with page walk caches     |  |
| Network; Protocol    | 2D mesh across clusters and within clusters; MESI   |  |
| SetOverflow          | 2 6-bit ptrs/entry + 12 6-bit ptrs in PointerSpace. |  |
|                      | T1=6, T2=2. Additional latency of dir access on     |  |
|                      | write with O=1: 3 cycles                            |  |
| Way Combining        | 3 6-bit ptrs/entry                                  |  |
| Network              |                                                     |  |
| Within-cluster       | 5 cycles/hop (4 in router + 1 in wire) [7]          |  |
| Cross-cluster RT     | 60 cycles to go to another chiplet and back [27]    |  |
| Main-Memory          |                                                     |  |
| Organization         | 512GB, 1GHz; DDR; 32 mem controllers                |  |
| Max. bandwidth       | 100GB/s per DRAM memory controller                  |  |

Global home, verification confirms that there is no deadlock or state inconsistency. In directory eviction scenarios with inflight invalidations, the model guarantees that eviction cannot discard metadata required to complete the transaction. Finally, in a directory set, the per-way directory state is consistent with the PointerSpace state.

# <span id="page-7-2"></span>*D. Efficient Many-Sharer Directory Entries*

From past work [\[22\]](#page-13-4), it is known that a storage-efficient directory design needs to support only a few sharers for most of the lines, and many sharers for very few lines. To support such a design, Dorado includes a new scheme that we call *Per-Set Overflow Pointers* (*SetOverflow*). As we describe the scheme, directory refers to both TD and ED.

In *SetOverflow*, each set of a set-associative directory has a structure with sharer pointers (*PointerSpace*) that can be used by one or more of the ways in the set when such ways overflow their own set of pointers. Figure [9a](#page-7-1) shows one set of a W-way set-associative directory with SetOverflow. It shows the directory of each way (which includes two sharer pointers) and the PointerSpace. The PointerSpace has two arrays: *SharerPointer* and *OwnerWay*. SharerPointer has the overflow pointers; OwnerWay has the IDs of the ways that are using those pointers. To save space, each of the T<sup>1</sup> entries in OwnerWay owns T<sup>2</sup> consecutive entries in SharerPointer: e.g., inserting a way ID in the first entry of OwnerWay allows that way to use the first T<sup>2</sup> entries in SharerPointer, and similarly for the other entries.

In a directory set, when one of the ways runs out of pointers (i.e, the two pointers in Figure [9a](#page-7-1)), it sets its *Overflow into PointerSpace* (O) bit and uses pointers from the PointerSpace. To do so, the way claims an empty entry in OwnerWay, where it puts its way ID. This allows it to use T<sup>2</sup> entries in SharerPointer (starting at offset T1xT2). If the way needs

![](_page_7_Figure_7.jpeg)

<span id="page-7-1"></span>Fig. 9. *SetOverflow* design.

more pointers, it can continue to claim more OwnerWay (and SharerPointer) entries. Since multiple ways can claim entries in the PointerSpace, it is possible that a way requesting pointers finds that all the pointers in PointerSpace are in use. In this case, that way sets its *Broadcast* (B) bit, clears its O bit, and releases all of its pointers in PointerSpace.

Overall, the SetOverflow design allows one or more "manysharer" directory entries in the same set to grow into the PointerSpace with minimal performance impact. Indeed, the PointerSpace is not accessed if the O bit of the directory entry is clear. This occurs when the entry is: (1) in state M or E, or (2) in state S with few sharers (e.g., with at most two sharers as shown in the figure, which is enough to capture migratory sharing [\[60\]](#page-14-12)), or (3) in state S with many sharers and the B bit set. PointerSpace is only accessed on a read or write to a line in state S with the O bit set or about to be set. On a read, the access to the PointerSpace is not in the critical path of sending the line to the requester, since the line is sent first. On a write to a line with the O bit set, the access to the PointerSpace is in the critical path, since the transaction needs to identify all the sharers.

Figure [9b](#page-7-1) shows the operation performed on PointerSpace on such a write. We need to collect all the pointers owned by the target entry to send invalidations. The hardware accesses all the T<sup>1</sup> OwnerWay entries in parallel and compares them to the target way ID. On a match, the corresponding T<sup>2</sup> SharerPointer entries are read out. Then, all the read-out pointers are accumulated into a final array, so that invalidations can be sent. To perform these hardware operations, we add 3 cycles to the directory operation in this case. In the background, the hardware clears the corresponding entries in PointerSpace and the O bit of the way.

SetOverflow is different than Way Combining [\[64\]](#page-14-18), which only allows a line to steal *all* the sharer pointers of an *unused* line in the same cache set. Moreover, these sharer pointers must be returned when a new line is inserted into that cache entry. In contrast, SetOverflow enables fine-grained assignment of the extra sharer pointers to different lines in the set, and does not need cache line entries to be unused to work.

To validate the correctness of the Dorado protocol, we formally specify it in TLA+ [\[34\]](#page-13-29) and model-check it using TLC [\[34\]](#page-13-29). The TLA+ specification models a finite collection of clusters, cores, directory slices, and cache lines. Cache states are modeled as {M, E, S, I, MS}. Each directory entry encodes a sharer set, a dirty bit, and a home type (Global or Temporary). For every line, the model maintains a Global home and a (possibly empty) set of Temporary homes. Pointer types (LLptr, LRptr, RLptr) are modeled symbolically.

The Global home is modeled as the serialization point for conflicting transactions. Memory operations modeled include read/write hits/misses, invalidations, downgrades, home allocation/deallocation, cache/directory entry eviction, and overflow pointer allocation/reclamation. Multi-step memory operations are decomposed into ordered transitions.

To keep the state space tractable, we parameterize the model with 4 clusters, 4 cores per cluster, and 4 lines per core. Even such a small instantiation admits thousands of interleavings that cover concurrent cross-cluster writes, read–write races, and many corner cases.

Verification of safety and liveness. We verify five properties. The first one enforces *single-writer multiple-reader semantics*, namely that: 1) at most one cache may hold a line in state M, 2) if a cache holds an M line, no other cache can hold it in any state but I, and 3) if a cache holds an MS line, no other cache can hold it in any state but MS or I. The second property enforces *dirty-bit consistency*, ensuring that if a cache holds a line in M or MS, the line has a directory entry in the Global home and at most in the local Temporary home (if they are not the same), and both have the dirty bit set. The third property enforces *home consistency*: every Temporary home must be reflected in the metadata of the corresponding Global home. The fourth property guarantees *sharer soundness*: every single line in any cache has a valid entry in at least one directory. The fifth property is *read correctness*: a read to a memory location must return the value of the latest update to that location.

Verification of the absence of deadlock and livelock. Deadlock freedom requires that no reachable global state exists in which no transition is possible. Livelock freedom ensures that a memory request cannot be indefinitely postponed by cyclic interference among transactions. These properties rely on the invariant that any transaction that acquires the Global home entry cannot be preempted or permanently delayed by conflicting transactions to the same line that have not yet acquired the Global home. In particular, a transaction that may change the state of the Global home cannot lock any resource in a Temporary home until the transaction has reached and successfully locked the entry in the Global home.

TLC reports no invariant violations or deadlocks. Several corner cases are exercised. In concurrent write races from different clusters, model checking confirms that exactly one transaction acquires the Global home first, while the others observe the updated sharer metadata. In scenarios where a Temporary home exists and a transaction needs to modify the

TABLE VII ARCHITECTURAL PARAMETERS USED IN THE EVALUATION.

<span id="page-8-1"></span>

| Processor Parameters |                                                     |  |
|----------------------|-----------------------------------------------------|--|
| Package              | 1024 6-issue OoO cores, 352-entry ROB, 3GHz         |  |
| Clusters             | 32 clusters of 32 cores each                        |  |
| Sharer ptr size      | LLptr, LRptr, RLptr: 5b for ID + 1b for type        |  |
| L1 D/I caches        | 64/32KB, 8-way, 4 cyc. round trip (RT), 64B line    |  |
| L2 cache             | 2MB, 16-way, 16 cycles RT, 80 MSHRs                 |  |
| L3 cache             | Slice: 6MB/core, 12-way, 60 cyc. RT, 160 MSHRs      |  |
| L1 D/I TLBs          | 256/128 entries, 4-way, 2 cycles RT                 |  |
| L2 TLB               | 2048 entries, 12-way, 12 cycles RT                  |  |
| Page translation     | 4-level radix page tables with page walk caches     |  |
| Network; Protocol    | 2D mesh across clusters and within clusters; MESI   |  |
| SetOverflow          | 2 6-bit ptrs/entry + 12 6-bit ptrs in PointerSpace. |  |
|                      | T1=6, T2=2. Additional latency of dir access on     |  |
|                      | write with O=1: 3 cycles                            |  |
| Way Combining        | 3 6-bit ptrs/entry                                  |  |
| Network              |                                                     |  |
| Within-cluster       | 5 cycles/hop (4 in router + 1 in wire) [7]          |  |
| Cross-cluster RT     | 60 cycles to go to another chiplet and back [27]    |  |
| Main-Memory          |                                                     |  |
| Organization         | 512GB, 1GHz; DDR; 32 mem controllers                |  |
| Max. bandwidth       | 100GB/s per DRAM memory controller                  |  |

Global home, verification confirms that there is no deadlock or state inconsistency. In directory eviction scenarios with inflight invalidations, the model guarantees that eviction cannot discard metadata required to complete the transaction. Finally, in a directory set, the per-way directory state is consistent with the PointerSpace state.

