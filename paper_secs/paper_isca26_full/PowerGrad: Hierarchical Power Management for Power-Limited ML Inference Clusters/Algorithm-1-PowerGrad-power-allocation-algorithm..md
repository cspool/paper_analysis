# Algorithm 1 PowerGrad power allocation algorithm.

```
1: // G: performance gradients, f: average core frequency
2: // P: power consumption, P L: power limit
3: // lr: learning rate, α: decrement rate for unused budget
4: function ALLOCATE POWER(G, f, P, P L, lr, α)
5: communicate(parent, G, f, P) //Report to the parent
6: global P Lnode // limit set asynchronously by the parent
7: P Ltotal ← 0
8: for i ∈ children do // Initial power budget assignment
9: P L′
              [i] ← P L[i] + lr × G[i] − α(P L[i] − P[i])
10: P Ltotal ← P Ltotal + P L′
                                   [i]
11: for i ∈ children do // Adjust the power budgets equally
12: P L′
              [i] ← P L′
                        [i]−(P Ltotal −P Lnode)/Nchildren
13: for i ∈ children do
      // Try to keep processors above the minimum frequency
14: if P L′
                [i] < P L[i] + incmin and f[i] < fmin then
15: P L′
                 [i] ← P L[i] + incmin
16: Re-adjust other processor power limits
17: break;
      return PL'
```

where P<sup>i</sup> and ∂BIP Si/∂P<sup>i</sup> are the power and performance gradient of core i, respectively, and *P* is the sum of the power of all the cores in the processor. For the last expression, we use ∂BIP S/∂P<sup>i</sup> = ∂BIP Si/∂P<sup>i</sup> because ∂BIP Sj/∂P<sup>i</sup> = 0 for all i ̸= j. Also, we set ∂Pi/∂P = Pi/P based on the approximation that when the power-capping method (e.g., RAPL) adjusts the power limit of the processor, each core's power changes proportionally to the core's current power consumption. For example, if the processor power is decreased by 10%, each core decreases its power by 10%. This is a reasonable approximation because when the power-capping method changes the frequency of the processor, the cores consuming more power are affected more than the cores consuming less power.

