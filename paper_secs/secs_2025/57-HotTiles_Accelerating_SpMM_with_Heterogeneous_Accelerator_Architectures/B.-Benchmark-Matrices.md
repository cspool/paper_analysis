# *B. Benchmark Matrices*

We use ten square sparse matrices from SparseSuite [20] as benchmarks. They have 100K to 4.2M rows and 22M to 100M nonzeros (Table V). Most of the matrices (pap, del, kro, myc, pac, ser) are the same or scaled-down versions of the ones used in SPADE [24]. Since the high fidelity of the PIUMA simulator leads to significantly increased simulation times, we replaced 4 of the matrices used in the SPADE paper with smaller ones to decrease the PIUMA simulation time. The 4 new matrices (ski, dgr, pok, wik) are selected so that they represent different application domains than the matrices borrowed from the SPADE paper. We set the number of columns of the dense matrices (K) to 32, similar to prior works [24], [31].

