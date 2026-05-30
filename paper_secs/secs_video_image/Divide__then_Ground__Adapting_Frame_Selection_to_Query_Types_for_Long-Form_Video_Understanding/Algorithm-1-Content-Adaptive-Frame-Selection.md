# **Algorithm 1:** Content-Adaptive Frame Selection

```
Input: Distance sequence d = [d1, d2, . . . , dM−1], Frame indices I = [I1, I2, . . . , IM]
  Output: Selected r-frame indices Ridx
1 P ← ∅;
2 for i = 2 to M − 2 do
3 if di−1 < di and di > di+1 then
4 P ← P ∪ {i} ; // A peak is a point higher than its neighbors
5 Pvalid ← ∅;
6 foreach j ∈ P do
7 lmin ← dj ;
8 k ← j − 1;
9 while k ≥ 1 and dk ≤ dj do
10 lmin ← min(lmin, dk);
11 k ← k − 1;
12 rmin ← dj ;
13 m ← j + 1;
14 while m ≤ M − 1 and dm ≤ dj do
15 rmin ← min(rmin, dm);
16 m ← m + 1;
17 pprom ← dj − max(lmin, rmin) ; // Calculate topographic prominence
18 if pprom > 0.1 then
19 Pvalid ← Pvalid ∪ {j};
20 Ridx ← ∅;
21 for i = 1 to |Pvalid| − 1 do
22 p1 ← Pvalid[i];
23 p2 ← Pvalid[i + 1];
24 m ← (Ip1 + Ip2
                  )/2 ; // Midpoints between consecutive prominent peaks
25 Ridx ← Ridx ∪ {m};
26 return Ridx;
```

from the filtered set. For each pair, it calculates the temporal midpoint using their associated original frame indices from *I*: midpoint = (*Ip*<sup>1</sup> + *Ip*<sup>2</sup> )*/*2. These midpoints, which correspond to the center of the most stable segments, are aggregated into the final r\_idx set.

