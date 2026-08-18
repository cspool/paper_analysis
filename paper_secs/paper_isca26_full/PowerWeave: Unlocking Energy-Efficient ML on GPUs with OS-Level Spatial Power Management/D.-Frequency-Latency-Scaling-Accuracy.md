# *D. Frequency-Latency Scaling Accuracy*

To evaluate the robustness of the frequency-latency scaling module, we ablate its two key components: the sensitivity factor and live weight updates. Across disaggregated experiments, we ask the governor to target 10%, 20%, and 50% performance degradation, and measure the resulting TTFT and TPOT slowdown relative to maximum frequency. Figure 12 shows observed degradation for each configuration, with error bars indicating the minimum and maximum across experiments.

The full PowerWeave system achieves high accuracy. The average case deviates by 1.7% from the target, while the maximum error is 5.2% in the worst case. The average accuracy is consistently under the target, which means that PowerWeave does not overestimate performance loss. Without live weight updates, the average deviation increases to 4%, and the maximum error rises significantly to 75%. This configuration produces much larger prediction errors at high load. Without sensitivity, misprediction averages 4%, but tail mispredictions increase, with the highest reaching 10.6%. In addition, mispredictions at 10% and 20% performance slip overestimate performance loss, which leads to SLO violations. These ablations show that both components are necessary. Sensitivity ensures accurate frequency selection across kernels with different compute-memory profiles, while live weight updates provide robustness to runtime workload shifts.

TABLE IV: Area overhead per additional domain.

| Component                    | Area (mm2<br>) | Percentage of die |
|------------------------------|----------------|-------------------|
| Voltage Regulator            | 0.0023         | 0.00009%          |
| Voltage-Domain Boundary Sync | 0.0359         | 0.00224%          |
| Clock Generation             | 0.0036         | 0.00014%          |

