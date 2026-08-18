# *B. Reasoning About Callbacks*

Consider Figure 4a, the example from §I augmented with an additional OnWB callback that writes the value 1 to address [y] (a regular address with no callbacks registered). If Figure 4a were a regular 3-thread program, the outcome r1=2, r2=0 would be possible. This outcome is actually impossible on tak¨ o, but understanding why this is the case requires reasoning ¯ about the semantics of caches and callbacks.

![](_page_3_Figure_0.jpeg)

Fig. 5: (a) The mp (message passing) litmus test. All addresses are assumed to be 0 initially. (b) An execution graph of mp that is outlawed under sequential consistency (SC).

Figure 4b explains this reasoning. If r1=2, the value for (i2) must have been generated as the result of an OnMiss A instead of by (i1). Thus, an interspersed OnMiss must have run between (i1) and (i2). However, since (i1) would have brought the data for [x] into the cache, that data must also have been evicted B before the execution of the OnMiss that (i2) read r1=2 from. This eviction must have been an OnWB, as the data has been modified by (i1) and is dirty C . Due to the serialization of callbacks ensured by the cache controller D , we can guarantee that (i5) (the write to [y]) has already completed by the time (i4) runs. Thus, the load of [y] in (i4) must read a value of r2=1 for [y] in this execution, making the outcome r1=2, r2=0 impossible.

