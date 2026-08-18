# *C. Thermal Throttling*

PowerWeave's spatial DVFS effectively eliminates thermal throttling across our experiments. In the disaggregated setting, Qwen3-32B-FP8 reaches the device's power limit for 52.9% of the duration under the default GPU policy. LithOS's DVFS reduces this to 30.1%, while PowerWeave's spatial DVFS reduces it to 2%. Similarly for Qwen3-14B, default GPU settings throttle the device for 25.7% of the experiment's duration. LithOS reduces this to 15.3%, while PowerWeave eliminates thermal throttling entirely. In the multitenancy experiments, the default policy throttles for 11% on average, LithOS for 2%, and PowerWeave again eliminates throttling. The remaining workloads do not throttle even at maximum frequency.

