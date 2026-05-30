# Algorithm 1: Branch-and-bound Scheduling

```
Input: [a_1, b_1], [a_2, b_2]: ranges of variables x_1, x_2,
            \epsilon_T, \epsilon_L: tolerance for throughput and latency,
            L_b: latency bound.
   Output: Scheduling Configuration x_1 = v_1, x_2 = v_2
 1 B_0 = \{(a_1, a_2), (b_1, b_2)\};
 <sup>2</sup> B_0.lowr = perf(a_1, a_2); // gives (latency, throughput)
3 T^* = B_0.lowr.thrput; T^*.config = (a_1, a_2);
 4 Q = PriorityQueue(\{B\}); // sorted by lower bound
5 while Q \neq \emptyset do
        B = Q.popMax();
        p_{tl}, p_{br} = perf(B.topLeft), perf(B.bottomRight);
        per f_* \leftarrow \text{higher thrput satisfying } L_b \text{ in } [p_{tl}, p_{br}];
        if perf_* == p_{tl} then B_1, B_2 \leftarrow split B vertically;
        else B_1, B_2 \leftarrow split B horizontally;
10
        foreach B' \in [B_1, B_2] do
11
             B'.upp = perf(B'.topRight);
              B'.lowr = perf(B'.bottomLeft);
13
             if B'.lowr.latency < L_b + \epsilon_L then Q.add(B');
14
        B' \leftarrow \text{higher throughput satisfying } L_b \text{ in}
15
          [B_1.upp, B_2.upp];
        if B'.upp.thrput > T^* then
16
             T^* = B'.upp.thrput; T^*.config = B'.topRight;
             Q \leftarrow Q \setminus \{\hat{B} | \hat{B}.upp.thrput + \epsilon_T < T^* \}
19 return T*.config;
```

run the algorithm separately for RRA and WAA Schedule and select the solution that gives the highest throughput.

When running the algorithm, we treat partial tensor parallelism specially. The variable controls two values: parallelism degree and the number of GPUs to which tensor parallelism is applied. To enforce the monotonicity property, we fix the parallelism degree as a constant and then run the algorithm. With a given degree, increasing the number of tensor-parallel GPUs reduces the latency and decreases the throughput, making the variable monotonic (which we verify in the evaluation section). We run the algorithm multiple times with different degree values to find the optimal solution.

#### 5.2 Dynamic Scheduling for Consistent Workload

Both RRA and WAA Schedule keep average encoder/decoder batch sizes be consistent for their scheduling. However, the encoding and decoding workload can vary due to variations in input and output sequence lengths across different queries. In WAA, for example, a long encoding stage can miss the handover of its output to the decoding stage, potentially resulting in uneven decoding batches and execution times.

To achieve consistent and reliable inference execution, we implement dynamic workload adjustment at runtime. Specifically, we dynamically adjust the encoder batch size to ensure that the encoder workload (i.e., sum of input sequence lengths in a batch) stays within a predetermined threshold of the average workload. Moreover, to maintain consistent decoder workload, we monitor its batch size, and if it falls below/above the threshold compared to the average workload, we increase/decrease the encoder batch size accordingly. These adjustments are taken into account in our scheduling process.

