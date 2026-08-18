# *C. Processing Memory Accesses*

As previously discussed, MIN makes the optimal replacement decision, only in the event of a miss-induced demand replacement. Here we extend this concept to include the the prefetching of future misses instead of waiting until demand time. R-Max looks ahead in the access stream and makes replacement decisions instead of waiting for misses and selecting blocks to replace, as shown in Alg. 1. As the algorithm shows, first the access list is divided up into separate, per-set, lists where each per-set list contains only those accesses destined to a given set in the chosen cache. Then the algorithm processes each of those lists to determine, under MIN, which accesses should be marked for "prefetch" into that set (because they would be misses) versus which accesses would be already in

<sup>2</sup>We find that at most 12 iterations is sufficient for the workloads examined.

Algorithm 1: Setup Memory Access List For Each Set

```
Input: mem: list of (address, timestamp) pairs of
        length x; w, y: number of cache sets and ways
  Output: List Acc with prefetch/hold updated per set
1 for s ← 0 to w − 1 do
2 Acc =<empty list>;
3 for i ← 0 to x − 1 do
4 if mem[i] belongs to cache set s then
5 Append mem[i] to Acc; // Get the memory
             accesses for this set.
6 m = length of Acc;
7 set = container holding max y pairs of (address,
      timestamp) resembling an actual cache set;
8 for i ← 0 to m − 1 do
9 if Acc[i] not in set and set has empty space
         then
10 Prefetch Acc[i] into set; // Prefill the set.
11 else if set full then
12 Break; // Stop the prefill.
13 for i ← 0 to m − 1 do
14 if Address of Acc[i] in set has not been
         demand-accessed then
15 Mark Acc[i] as "prefetch"; // Prefetched,
             not demand accessed.
16 else
17 Mark Acc[i] as "hold"; // Prefetched,
             demand accessed.
18 Update the timestamp of address of Acc[i]
         found in set with its next access time, set to
         ∞ if not found;
19 for j ← i + 1 to m − 1 do
20 if Acc[j] not in set then
21 Find l in set with the largest timestamp;
22 if l has larger timestamp than Acc[j]
                then
23 Prefetch Acc[j] to replace l in set;
24 Break;
25 Output Acc for set s;
```

the cache and thus are marked as "hold". In either case the access time of each access is retained.

After marking memory accesses, dead block counters are generated using Alg. 2 to denote the number of "hit" accesses a block will receive in the set before eviction. This counter counts accesses between the initial "prefetch" marking from Alg. 1 until a second "prefetch" marking, or the end of the memory access record is reached, whichever comes first. If a block has multiple "prefetch" markings, it will be prefetched and evicted multiple times. The "dead block counters" are used to simplify the implementation of R-Max. The algorithm runs once before the simulation and it assumes instantaneous placement of prefetched blocks.

Algorithm 2: Gather Prefetches and Counters per Set

```
Input: Array Acc of length m processed by Alg. 1
 Output: List of prefetches with counter information
1 for i ← 0 to m − 1 do
2 if Acc[i] has prefetch bit set then
3 Set counter c to 1;
4 for j ← i + 1 to m − 1 do
5 if Acc[j] has the same address as Acc[i]
           AND Acc[j]has hold bit set then
6 c + +;
7 else
8 break;
9 Add (Acc[i], c) to the delayed prefetch list;
```

Table II shows an example of applying Alg. 1 to a memory access stream recorded for a set with 4 ways. During prefill (lines 8-12), R-Max will fill the available ways and tag each way with the timestamps for the next use. After prefill, each row shows the time, the demand access address, whether the access is the first one after the block is prefetched, what is the next address that is not found in the set, what is the time t of that address will be accessed, and after comparing t with all the timestamps of the blocks in the set, should a replacement be initiated, and if there is a replacement, what blocks the set has. For example, at time 1, A is accessed. A is prefilled/prefetched but has not yet been demand accessed, R-Max marks A at time 1 a "prefetch". At this time, A's next access time is updated to time 35, as reflected in the upper half of the time 1 row. The set has blocks A, B, C and D. Looking further into the set, the next block that is not found in the cache is E with access time 27. Time 27 is less than one of the timestamps of the blocks in the set. Therefore, E replaces the block with the largest timestamp, A, of the set. Such replacement is reflected in the lower half of the time 1 row.

Counters for the memory accesses shown in Table II are shown in Table IV, generated using Algorithm 2. The same block can be brought into the cache multiple times: block B is brought in and after two demand accesses, it is evicted; after accessing G, B is brought back into the cache again.

