# *B. Callback Synchronization*

In tak¨ o, the microarchitecture's prefetching and cache re- ¯ placement policies still control *when* a cache line is moved in and out of the cache, as Figure 2 showed. Thus, callbacks can

![](_page_2_Figure_11.jpeg)

![](_page_2_Figure_12.jpeg)

Fig. 4: (a) An extended version of Figure 2a's program (original program in gray) that depicts the interactions of callbacks and regular address [y]. The outcome r1=2, r2=0 is impossible on tak¨ o. (b) A timeline that explains why the outcome is ¯ impossible, as r1=2 implies the OnWB has completed and written 1 to [y], forbidding r2=0.

be interleaved rather arbitrarily with core thread instructions. While a load reading from an OnMiss cannot commit before the OnMiss completes, OnEvict and OnWB callbacks can execute anytime after their address is brought into the cache or written to in the cache respectively. tak¨ o thus offers a ¯ FlushRange synchronization primitive. A FlushRange causes all cache lines with addresses in the mentioned range to be evicted from the cache, invoking their OnEvict or OnWB and blocking the FlushRange till they complete.

tak¨ o engines serialize all callbacks to the same address ¯ in FIFO order to reduce the possibility of races on those addresses [55]. However, tak¨ o executions can still easily lead ¯ to races and counterintuitive outcomes, as we discuss next.

