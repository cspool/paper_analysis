# <span id="page-5-0"></span>Algorithm 1: Greedy search algorithm Input: Inputs-to-experts mapping gating Input: n Result: Communition-efficient expert placement P oE

// Preliminary <sup>1</sup> Toutput ← T ′ (R, H, 0, 0); <sup>2</sup> H, R ← GetH&R(gating); <sup>3</sup> L, n bottoms ← [], []; <sup>4</sup> cnt ← 0;

```
// Iteratively search
5 while not balanced do
     // Get the index of the heaviest
        device
6 i ← arg max
           i
               (H);
7 if i in Used then
8 break;
9 end
10 Used.append(i);
     // Determine n devices saving the
        smallest number of inputs for
        expert-i
11 n bottom ← BottomK(gating, n) ;
12 L.append(i);
13 n bottoms.append(n bottom)
14 s ← size(L)
     // Replace inputs among devices
        according to the expert
        placement
15 H, R ← Replace Inputs(L, n bottoms)
     // Evaluate the expert placement
16 Tchanged ← T
                ′
                (R, H, s, n);
17 if Tchanged < Toutput then
18 Toutput ← Tchanged;
19 cnt = s
20 end
21 end
```

