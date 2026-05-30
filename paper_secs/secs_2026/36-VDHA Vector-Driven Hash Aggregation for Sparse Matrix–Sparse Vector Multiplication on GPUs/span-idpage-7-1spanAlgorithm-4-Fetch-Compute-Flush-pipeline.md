# <span id="page-7-1"></span>Algorithm 4 Fetch-Compute-Flush pipeline

**Input:** Column segments seg; two shared-memory buffers buf[0], buf[1]; shared-memory hash table H

```
Output: Result vector y
 1: Fetch(seg_0, buf[0])
 2: Sync()
 3: for i = 0 to N_{\text{segs}} do
         if i \neq N_{seqs} - 1 then
 4:
             Fetch(seg_{i+1}, buf[(i+1)\%2])
 5:
         ind, val \leftarrow buf[i\%2]
 6:
         insert(H, ind, val)
 7:
 8:
         Sync()
 9:
         if hash table full or i == N_{seqs} - 1 then
```

Flush(H, y)

10:

wait\_group to ensure that the next segment has been fully loaded into shared memory (line 2,8). Aggregated entries remain in the shared-memory hash table until it approaches capacity, at which point they are flushed to global memory in bulk before processing continues (line 10).

This design enables a steady overlap: while segment t is being computed, segment t+1 is already in fetch, effectively reducing the cost of hash computation.

