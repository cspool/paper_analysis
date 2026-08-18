# D. Training the PowerGrad Power and Performance Models

As described in Section III-C, PowerGrad uses power and performance models for the Gradient Estimator. The regression coefficients for these models (i.e.,  $a_i$  in (5) and  $w_i$  in (6)) and  $\gamma$  are learned by running training applications that are different from the applications we evaluate. For the Legacy CPUs, since they have no specialized support for ML acceleration, we learn the coefficients using conventional PARSEC 3.0 [1] applications. For the Accelerated CPUs, we use TorchBench [5] to engage the AMX instructions.

To learn the regression coefficients and  $\gamma$ , we follow the workflow of the PPEP framework [39]. We use similar counters as PPEP, namely the following six counters: *instruction-count*, *cycle-count* (non-idle cycles), *uops.executed* (uops), *cache-misses*, *branch-misses*, and *ldm\_stalls\_pending* (memory stalls). For the Accelerated CPUs, we read two additional counters, *exe.amx\_busy* and *fp\_arith\_inst\_retired.vector*, to measure the usage of AMX and vector instructions, respectively. Since the AMX unit performs many operations per instruction, we count each busy cycle of this unit as multiple instructions when measuring the performance in BIPS. Specifically, one AMX busy cycle is counted as 16 instructions. This is because the number of operations done in each AMX busy cycle is equivalent to 16 vector instructions [19].

TABLE III

<span id="page-7-0"></span>AVERAGE ABSOLUTE ERROR (AAE) AND ITS STANDARD DEVIATION FOR POWERGRAD'S POWER MODEL IN TWO DIFFERENT SYSTEMS.

| System      | Avg. Power (W) | AAE (%) | AAE STD (%) |
|-------------|----------------|---------|-------------|
| Legacy      | 74.6           | 4.1     | 5.0         |
| Accelerated | 162.9          | 2.5     | 3.9         |

#### V. EXPERIMENTAL RESULTS

In this section, we first evaluate the soundness of the performance gradient estimation. Then, we present the effectiveness of PowerGrad and, finally, discuss some PowerGrad hyperparameters.

#### A. Accuracy of the Power Model

PowerGrad's efficacy depends on the accuracy of the Gradient Estimator's online power and performance models. PowerGrad follows the methodology of PPEP [39], which uses a linear performance model (3) and a polynomial power model (4). In this section, we validate the power model by running our applications of Table II on one processor of the Legacy and Accelerated systems. At every 100ms, we compute the difference between the prediction of the model and the actual power measured by RAPL. This difference is the absolute error, which is then averaged across all the measurements. Table III shows this average absolute error (AAE) and its standard deviation (AAE STD). The Table also shows the average power of the applications. From the table, we see that the PowerGrad model is accurate. In the Legacy system, the AAE is 4.1% while, in the Accelerated system, it is 2.5%. The Accelerated system has a lower value because it provides access to more fine-grained CPU performance counters, such as vector instruction counts. The values for the AAE standard deviation are 5.0% and 3.9% for the Legacy and Accelerated systems, respectively. In the original PPEP work, the AAE is 4.6% and the AAE STD is 3.6%.

