# *G. Interface with Hardware*

PowerWeave is well positioned to establish a direct communication interface with the GPU's power-management hardware. Because it interposes above the GPU driver, Power-Weave's DVFS controller learns the kernel sequence each model executes, and distinguishes compute-bound kernels from memory-bound and communication-heavy ones. This lets PowerWeave proactively issue frequency requests for upcoming phases instead of reacting to execution-time signals.

Such an interface could take the form of a per-domain request queue shared between PowerWeave and the GPU power-management firmware. For each domain, PowerWeave would enqueue frequency targets, ahead of upcoming kernel launches. The firmware would consume these requests asynchronously, schedule the required transitions, and settle each domain at the requested operating point before the corresponding kernel begins. By issuing requests early, PowerWeave could hide the communication overhead to the firmware and enable fast, timely frequency switching. This can unlock faster and finer-grained DVFS across both spatial and temporal dimensions, enabling more energy-efficient operation.

