# *E.* FlushRange *Utility in Multicore Programs*

We now explore the utility of the FlushRange primitive in a multicore setting by considering a multicore version of the wbf litmus test (§V-D). Figure 12 contains 2 implementations

| Core 0                                                | Core 1                |                    | [x].OnMiss   |         |
|-------------------------------------------------------|-----------------------|--------------------|--------------|---------|
| (i1) RMW([x],<br>, 1)                                 | (i4) RMW([x],<br>, 2) |                    | (i7) [x] ← 0 |         |
| (i2) FlushRange[x]                                    |                       | (i5) FlushRange[x] |              |         |
| (i3) r1 ← [y]                                         | (i6) r2 ← [z]         |                    |              |         |
| [x].OnWB<br>(1)                                       | [x].OnWB<br>(2)       |                    |              |         |
| r3 ← [y]<br>(i8)                                      |                       | (i12)              | if x = 1:    |         |
| (i9)<br>if r3 = 0:                                    |                       | (i13)              |              | [y] ← 1 |
| [y] ← 1<br>(i10)                                      |                       |                    | else:        |         |
| else:                                                 |                       | (i14)              |              | [z] ← 2 |
| [z] ← 2<br>(i11)                                      |                       |                    |              |         |
| phiR<br>(with OnWB<br>(1)):                           |                       |                    |              |         |
| program racy under our tak¨ o MCM ¯                   |                       |                    |              |         |
| phiNR<br>(with OnWB<br>(2)):                          |                       |                    |              |         |
| no race, r1 = 0, r2 = 0 forbidden by our tak¨ o MCM ¯ |                       |                    |              |         |

![](_page_9_Figure_1.jpeg)

Fig. 12: (a) The phiR (with OnWB (1)) and phiNR (with OnWB (2)) litmus tests. (b) Execution snippet showing phiR race due to OnWB from core 0's write occurring after FlushRange on core 1. (c) Execution snippet showing how phiNR avoids the race by branching on evicted value in the OnWB.

(c)

of a program (named phiR and phiNR) in which multiple cores concurrently write updates to a phantom address [x] and the OnWB of [x] publishes the results back to different locations ([y] and/or [z]), depending on how many updates have previously been published (for phiR) and which update was written last (for phiNR). This logic is inspired by how the tak¨ o paper's acceleration of scatter-updates uses the number of ¯ updates in the line at eviction to determine whether to apply the updates in place or log them [55]. We additionally use RMW instructions ((i1) and (i4)) to update [x] in both tests because using stores for (i1) and (i4) would result in a race between those two accesses.

In phiR, the program runs with the OnWB (1) implementation of the OnWB for [x]. In this OnWB implementation, (i8) first reads [y]. If [y] has a value of 0 (i.e., if [y] has not yet been written to), (i10) writes 1 to [y]. If [y] has already been written to by a prior OnWB for [x], (i11) writes 2 to [z] instead. Thus, the first time the OnWB runs, it will write to [y], and if it runs a second time, it will write to [z].

The phiR litmus test is racy, as demonstrated by the execution snippet in Figure 12b. The core cause of the race is that the RMWs (i1) and (i4) can occur in either order, but the OnWB will always write to [y] in its first iteration and [z] if it runs a second time. Consider the case where (i4) runs first (not shown in Figure 12b) and then (i5) invokes the OnWB and causes [y] to be updated to 1 (this OnWB is also not shown). Figure 12b shows that when (i1) subsequently writes to [x] and (i2) then invokes the OnWB, this will trigger the write to [z] in (i11). However, nothing stops this write from racing with the read of [z] in (i6) on core 1, giving us a race.

This execution demonstrates a key requirement when using FlushRange for synchronization: for FlushRange to be able to eliminate a race, callbacks should not be able to run accesses that cause the race after the FlushRange in question has committed. Here, it is possible for the OnWB to be triggered and run (i11) after (i5) has committed, so (i5) is unable to prevent (i11) from racing with (i6).

The above requirement is fulfilled by the phiNR litmus test from Figure 12a that uses OnWB (2) as the OnWB for [x]. Here, the OnWB uses the evicted value to determine which value to write. (i12) checks the evicted value. If it is 1 (i.e., the last update was from Core 0), then (i13) writes 1 to [y]. If it is 2 (i.e., the last update was from Core 1), then (i14) writes 2 to [z]. Thus, once (i5) commits, any write to [z] is guaranteed to have completed – the OnWB only writes 2 to [z] if [x] is 2, and that can only happen after (i4) and before (i5) commits. Thus, there is no write to [z] that can race with (i6) in any execution. (Similarly, once (i2) commits, there is no write to [y] left that can race with (i3), eliminating races on [y] as well.) Figure 12c shows how there is no write to [z] in an OnWB triggered by (i1) and (i2), thus eliminating the race seen in Figure 12b.

In phiNR, the outcome r1=0,r2=0 is forbidden by our MCM, because neither (i3) nor (i6) can run before at least one FlushRange instruction (i.e., (i2) or (i5)) commits. Since the FlushRange instructions are each after RMW instructions that write to [x] in program order, the first FlushRange to commit will cause the OnWB to run, which will update one of [y] or [z]. Thus, at least one of [y] or [z] will have been written to before (i3) or (i6) run, preventing them both from returning 0 and forbidding the outcome of r1=0,r2=0.

In phiR and phiNR, the OnWB callback runs a maximum of two times, so the complexity is relatively easy to manage. Next, we investigate tak¨ o's implementation of accelerated ¯

|      | Core 0          |       | [e].OnMiss  |       | [e].OnEvict  |
|------|-----------------|-------|-------------|-------|--------------|
| (i1) | RMW([e], r1, 1) | (i4)  | r3 ← [g]    | (i9)  | if [e] ̸= 1: |
| (i2) | FlushRange[e]   | (i5)  | if r3 ̸= 1: | (i10) | [ℓ] ← 1      |
| (i3) | r2 ← [ℓ]        | (i6)  | [g] ← 1     |       |              |
|      |                 | (i7)  | [e] ← 0     |       |              |
|      |                 | else: |             |       |              |
|      |                 | (i8)  | [e] ← 1     |       |              |

hatsR (without (i9)): program racy under our tak¨ o MCM ¯ hatsNR (with (i9)): no race, r1 ̸= r2 forbidden by MCM

![](_page_10_Figure_2.jpeg)

Fig. 13: (a) hatsR and hatsNR litmus tests. (b) Execution snippet showing hatsR race due to OnEvict writing to the log post-traversal. (c) Snippet showing how hatsNR eliminates the race by only logging valid edges in the OnEvict.

graph traversal, which is more difficult to write correctly because it has writes in OnEvict callbacks that can run an arbitrary number of times.

