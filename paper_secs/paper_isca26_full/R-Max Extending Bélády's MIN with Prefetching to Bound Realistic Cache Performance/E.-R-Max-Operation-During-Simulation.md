# *E. R-Max Operation During Simulation*

Fig. 3 summarizes how each queue interacts with each other when there is a cache demand access coming from the upper levels of memory hierarchy. Prefetches issued by R-Max at the current level of cache are excluded. Algorithm 3 shows the additional detailed operations of how R-Max handles cache access that are not covered by the flow chart.

Step 1 : as in a normal cache, when a miss occurs, the cache first checks MSHRs for possible MSHR merging. See lines 4-6 of Alg. 3. If the counter of the missed address is 0, meaning that while the MSHR is open, enough accesses have been made to that address and the address has no more pending recent accesses, push the address to the Do Not Fill List to skip fill when the MSHR is closed after the block returns.

Step 2 : if step 1 failed, R-Max then checks the pending issue queue. A request can arrive before its prefetch is actually issued. If that is the case, the prefetch can be dropped to avoid duplicate requests to the lower level caches because an MSHR will open. See lines 7-8.

Step 3 : if step 1 2 failed, meaning that the miss is not found in the MSHRs or the Pending Issue Queue, R-max searches the λ Queue. The λ Queue contains counters for the blocks that are previously in the cache but are replaced before their counters drop to 0. See lines 9-10.

Step 4 : if step 1 2 3 failed, the Delayed Prefetch List is searched. R-Max prioritizes λ Queue over the Delayed Prefetch List for address searching because addresses from the λ Queue are prefetched but replaced due to access reordering. The access to addresses in λ Queue may come at a later time. The addresses from the Delayed Prefetch List are those waiting on the counters from the Cache Status Map to drop zero. The addresses and counters from the Cache Status Map and the λ Queue should be consumed before issuing new prefetches. See lines 11-12.

```
Algorithm 3: How R-Max Handles Memory Requests
  Input: Demand access v
  Output: R-Max state and cache state updated
1 if v misses AND MSHR full AND cannot merge then
2 return; // Cache blocked.
3 if v misses then
4 if v found in MSHR then
5 Erase from pending issue queue if found;
6 Do not fill if v's counter = 0;
7 else if v found in pending prefetch queue then
8 Erase from pending prefetch queue;
9 else if v found in λ queue then
10 Call memory reordering handler on λ queue;
11 else if v found in delayed prefetch list then
12 Call memory reordering handler on delayed
          prefetch list;
13 else
14 Push to do not fill list;
15 else if v hits, v's counter = 0 then
16 if v found in λ queue then
17 Call memory reordering handler on λ queue;
18 else if v found in pending prefetches list then
19 Call memory reordering handler on list of
          prefetches;
20 else
21 Set v as the eviction candidate;
22 Update memory access file;
23 if v hits OR v found in {MSHR || pending prefetch
   queue || λ queue || delayed prefetch list} then
24 Update counter;
25 if v's remaining accesses = 0 then
26 Get next prefetch from pending prefetch list;
27 Update R-Max state;
28 if v's remaining accesses = 0 then
29 //Do not fill if found in MSHR, λ queue, or
             pending prefetch list;
```

Step 5 : if after aforementioned operations, a counter in the cache drops to 0, a spot opens up, and R-Max updates the Cache Status Map for that set, the first prefetch in the Delayed Prefetch List is appended to the Pending Issue Queue and used to update the Cache Status Map. The prefetch will be issued when MSHRs become available, in the order in which they are placed into the Pending Prefetch Queue. At the same time, the address is written to the memory access file. Line 23-32 handles the situation that if the access is a hit, or it is found in MSHR, the Pending Prefetch Queue, the λ Queue or the Delayed Prefetch List. The situation means that R-Max

<sup>30</sup> Set for eviction if v is in cache;

<sup>32</sup> Erase v from do not fill list if found;

<sup>31</sup> else

has knowledge of the access and R-Max knows that access should be in or should be placed into the cache, it will update the counter of the access. Line 25 shows that if the counter drops to zero, R-Max will move the next prefetch from the pending prefetch list to the pending issue queue. Because the counter of one access drops to 0 means that block should be evicted to make space for another block that is to be accessed next but not in cache yet. The Cache Status Map is updated as well. After polling the Delayed Prefetch List for possible next prefetch, if the remaining access of v is 0, and v is considered as being inflight if it is found in MSHR, λ Queue or the Delayed Prefetch List, do not fill v. Lines 31-32 handle that case that after polling the pending prefetch list, the next to be prefetched address happens to be the same one seen by the cache, then the address should not be skipped for filling and should be kept in the cache if it is already there.

If the access does not cause a cache miss, steps 1 2 are skipped because a cache hit does not cause a MSHR search and means that the prefetched block is already in the cache. The address of the block should not be found in the Pending Prefetch Queue. If the access does not cause a cache miss and its counter can be found in the Cache Status Map in R-Max, only step 5 is needed to check for a potential prefetch enqueued to the pending prefetch queue.

If the access hits in the cache, but the address does not match with those from the Cache Status Map, start from step 3 . See lines 15-21.

If the address is not found in all the data structures of R-Max, it is issued and returned directly to the upper-level cache without filling the current cache level. Or if the block is in the cache, prioritize the block for eviction. R-Max puts the address to the Do Not Fill Queue. The missed block will be sent to upper level directly without replacement if found in the queue. Because R-Max has no knowledge of the address regarding the access counter and does not know how long to keep it in the cache, to avoid corrupting the whole structure, R-Max chooses to skip fill. See lines 13-14.

Alg. 4 shows how R-Max handles memory access reordering. The algorithm is called upon by either the λ Queue or the Delayed Prefetch List. Both lists have accesses that are not in the cache. The algorithm skips filling the block if the counter is 1. Or fill the block and update the Cache Status Map. Or choose a block in the Cache Status Map to replace, together with addresses and counters. The replacement decision right now is LRU because the access ordering has changed, R-Max cannot know the memory access ordering until the next iteration. R-Max then updates the cache status map to reflect its changed understanding of the cache state. The missed address together with its counter, will be erased from the queue. Note that the queue may have multiple occurrences of the same address. Only the first occurrence will be erased. The address can be accessed multiple times and the replacement decision only affects the first of them. All prefetches get a fair share of one issue. If the address is not found in MSHR or cache, it is not issued yet. The address and its counter are put back to the Delayed Prefetch List.

Algorithm 4: R-Max memory reordering handler

```
Input: Address v and its access counter, queue
Output: Updated R-Max state and queue.
```

```
1 if v has 1 access total then
2 Erase v from the queue;
3 Push v to do not fill queue. Skip fill;
4 else if R-Max cache status map has space then
5 Place v into R-Max cache status map;
6 Erase v from the queue and the pending prefetch
       queue;
7 else
8 Choose a block r for eviction;
9 Update R-Max cache status map, replace r with v
       and v's counter;
10 Erase v from the queue;
11 if r not in MSHR, not in cache then
```

<sup>12</sup> Put back r's address and counter back to the delayed prefetch list;

<sup>13</sup> else

<sup>14</sup> Put r's address and counter to λ queue; <sup>15</sup> Skip filling of r if found in MSHR;

<sup>16</sup> Set r for eviction if found in cache;

After searching queues and lists and making updates in the Cache Status Map, R-Max should have an updated projection of what the cache should have at the moment and how to manage the life expectancy of each cache block. R-Max sets the eviction position for each set and controls the Do Not Fill Queue to manage the cache and actively issues prefetches from the Pending Prefetch Queue once MSHRs become available.

