# *E. Streaming Pattern Prefetching*

Streaming accesses are common in many workloads and can be handled effectively by simpler dedicated mechanisms. Although the main PHT in STEP could, in principle, also learn these patterns, doing so would unnecessarily consume history capacity and interfere with non-streaming footprint learning. We therefore adopt the same lightweight dense-PC streaming detector used in Gaze [14], which includes a Dense PC Table (DPCT) to record recent dense PCs, as well as the same mechanism as in eBingo. This component is orthogonal to STEP's core contribution, which lies in staged trigger-time decisions for spatial footprint issuance.

