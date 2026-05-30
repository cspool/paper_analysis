# Algorithm 2 Workflow of the SpotServe migration planner.

```
▶ Progressive Migration
 1: function MigrationPlanner(context ctx, plan = [])
        plan.append(<migrate, ctx.cache>)
        O ← Layer migration order from MemOptMigPlanner
 4:
        for layer index i in range(0, #layers) do
 5:
            plan.append(<migrate, ctx.weight[O_i]>)
            p \leftarrow Get pipeline stage index of layer O_i
 6:
            if stage p completes all context migration then
 7:
                plan.append(<start, instances of stage p>)
    ▶ Memory Optimized Migration
 9: function MemOptMigPlanner(maximum buffer size U_{max})
        0 \leftarrow [], X \leftarrow \{\}
        Instance buffer memory usage U \in \{0\}^N
11:
12:
        for layer index i in range (0, \#layers) do
            if (migrate, ctx.weight[i]) doesn't exceed U_{max} then
13:
                 Update buffer memory usage U
14.
15:
                 O.append(i)
16:
                 X.add(i)
17:
        while X is not empty do
18:
19:
            x_{opt} \leftarrow
                 \underset{x \in \mathbf{X}}{\arg\min} \max_{0 \le i \le N-1} \{ \mathbf{U}_i \mid (\mathsf{migrate}, \mathsf{ctx.weight}[x]) \}
20:
            O.append(r_{opt})
21.
            X.remove(x_{opt})
```

 $C_t$  (i.e.,  $D_t \times B_t \ge D_{t+1} \times B_{t+1})^2$ , SpotServe discards part of the cached results to avoid exceeding the memory capacity of the new parallel configuration. To minimize recomputation cost, SpotServe keeps the batches of requests with more decoding progresses (i.e., iterations).

