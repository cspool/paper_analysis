# <span id="page-10-0"></span>Algorithm 1 Speculative Decoding with Crossmodel Retrieval

```
Require: Target LM Mq, draft LM Mp, input x1, . . . , xt,
   block size K, target length T, DRAFT, VERIFY, COR-
   RECT, retrieval flag doRetrieval, attention scores s, top-
   k chunks c1, . . . , ck
1: n ← t
2: while n < T do
                 ▷ Retrieve and update draft model cache
3: if doRetrieval then
4: c1, . . . , ck ← SELECTCHUNKS(s)
5: UPDATEDRAFTCACHE(c1, . . . , ck)
6: end if
7: p1, . . . , pK ← DRAFT(x≤n, Mp)
8: Sample x˜i ∼ pi for i = 1, . . . , K
                   ▷ Obtain target model attention scores
                                  for i = 1, . . . , K + 1
9: (qi, s) ← Mq

                    x | x≤n, x˜<i ; doRetrieval
10: if VERIFY(x˜i, pi, qi) then
11: xn+1 ← x˜i; n ← n + 1
12: else
13: xn+1 ← CORRECT(pi, qi)
14: break
15: end if
16: if all K drafted tokens accepted then
17: Sample xn+1 ∼ qK+1; n ← n + 1
18: end if
19: end while
```

