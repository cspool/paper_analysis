# F Actors

#### <span id="page-20-0"></span>F.1 Processor

Algorithm 2: *Processor Actor*: executed by a block

```
1 begin
2 tQ ← GetTQ()
3 signal ← 0
4 // shared memory variables
5 task ← {}
6 interrupt ← False
7 complete ← False
8 while interrupt == False do
9 if warpId == 0 then
10 if threadId == 0 then
11 awaitTaskFromScheduler(interrupt, signal)
12 FencedNotifyRQ(ready)
13 end if
14 syncwarp()
15 warpReadTQ(tQ, signal, task)
16 end if
17 syncthreads()
18 if interrupt == False then
19 switch task.Type do
20 case GEMM0 do
21 // fused GEMM, epilogue and async tile staging
22 fGET(GEMM0, task)
23 if threadId == 0 then
24 complete ← NotifyTileCompletion()
25 end if
26 syncthreads()
27 if complete == True then
28 NotifySchedulerNextGEMM(tQ)
29 end if
30 end case
31 case GEMM1 do
32 // fused GEMM, epilogue and async tile transfer
33 fGET(GEMM1, task)
34 end case
35 case Combine do
36 combine(task)
37 end case
38 end switch
39 end if
40 end while
41 end
```

### <span id="page-21-0"></span>F.2 Scheduler

Algorithm 3: *Scheduler Actor*: executed by one warp

```
1 begin
2 scheduled ← 0
3 tT B ← 0
4 tqState ← {}
5 pT DB ← GetProcessorDoorbell()
6 sT DB ← GetSubscriberDoorbell()
7 taskBound ← GetTaskBound()
8 tT B ← AtomicLoad(taskBound)
9 // circular buffer ready queue
10 rQ ← {}
11 // Populate ready queue with Processor ids
12 PopulateRQ(rQ)
13 while scheduled < tT B do
14 lt ← 0
15 do in parallel
16 Sweep doorbells and populate observed task counts into tqState
17 Aggregate locally observed task counts into lt
18 end
19 qS, taskT ally ← 0
20 // qS is the inclusive output
21 WarpInclusiveSum(lt, qS, tasktally)
22 while tasktally > 0 do
23 Repopulate rQ with ready processor ids
24 do in parallel
25 Starting at rQ[qS], signal processors about task indices from tqState
26 end
27 end while
28 if threadId == 0 then
29 tT B ← AtomicLoad(taskBound)
30 end if
31 tT B ← WarpBroadcast(tT B)
32 end while
33 InterruptSubscribers()
34 InterruptProcessors()
35 end
```

#### <span id="page-22-0"></span>F.3 Subscriber

#### Algorithm 4: *Subscriber Actor*: executed by three warps

```
Input: Tϕ ∈

            R
              2
               E×C
                   , Gϕ ∈ R
                          S×E O ∈ R
                                   S×H, X ∈ R
                                            E×H×D
1 begin
2 interrupt ← GetSharedInterrupt()
3 flags ← GetSymmetricFlags()
4 tQ ← GetTQ()
5 // Predefined upper bound on the number of tasks.
6 // We modulate this value to the actual task count computed
7 // dispatch signals received from peer GPUs
8 taskBound ← GetTaskBound()
9 while AtomicLoad(interrupt) == False do
10 // dispatch flags
11 do in parallel
12 Visit dispatch flags
13 Atomically retrieve signal
14 if Signal is set and flag is not visited then
15 Mark visited
16 SelfCorrectTaskBound(taskBound, Signal)
17 Enforce memory consistency before consuming packet
18 Decode packet into a set of GEMM0 task descriptors using X
19 Write task descriptors to tQ
20 Notify Scheduler of decoded tasks
21 end if
22 end
23 Advance flags by number of dispatch flags length
24 Atomically retrieve signal
25 // combine signals
26 do in parallel
27 Visit combine flags: one per tile
28 if Signal is set and flag is not visited then
29 Mark visited
30 Enforce memory consistency before consuming packet
31 Decode packet into a set of combine task descriptors using Tϕ, Gϕ, O
32 Write task descriptors to tQ
33 Notify Scheduler of decoded tasks
34 end if
35 end
36 end while
37 end
```

