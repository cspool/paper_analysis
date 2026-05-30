# *B. Necessity for Preemptive EDF On-GPU [\[3\]](#page-10-2)*

Capodieci *et al.* [\[3\]](#page-10-2) developed a mechanism to enable preemptive EDF scheduling on the GPU. Task scheduling is done by only inserting channels for a single application at a time on the runlist, and preemption is implemented by resetting the runlist.

The key problem with this work is that a TX2-specific attribute—the presence of only one runlist—is claimed as a general rule of NVIDIA GPU scheduling (as repeated in their later work [\[7\]](#page-10-15)).

Problematic assumptions. In the work of Capodieci *et al.*, they claim that only one task is active on a runlist at a time. This directly contradicts [R4](#page-4-7). They also overlook that other non-compute engines require access to a runlist, a conflict with [R7](#page-7-2). In follow-on work, when observing an instance of [R5](#page-5-5), they instead indirectly question [R3](#page-4-8)—the only claim of their original work that our rules directly align with.

However, are our rules strictly required for this management technique to be safe?

Counter-example. Consider a system of two GPU-using tasks on the NVIDIA Jetson Xavier. Task 1 performs compute and copy operations, whereas Task 2 performs only compute operations. Task 1 has a period of 3s, relative deadline of 2s, cost of 2s, and phase of 0.5s. Task 2 has a period of 3s, relative deadline of 3s, cost of 1s, and phase of 0. In this system, Task 2 will always release part-way through Task 1's execution, and have an earlier deadline, hence higher priority according to EDF. The result will be that Task 2 preempts Task 1 on every release.

According to Capodieci *et al.*, this preemption would be implemented by reconfiguring the compute runlist to only contain Task 2—this should work, per [R3](#page-4-8). However, the Jetson Xavier includes a second runlist for copy operations. Per [R5](#page-5-5), even though Task 1 is no longer active on the compute runlist, its copies may continue unhindered on other runlists. Such corunning copies have been shown to severely delay compute work on the Jetson Xavier [\[7\]](#page-10-15), causing Task 2 to miss its tight deadline. If [R3](#page-4-8), [R5](#page-5-5), and [R7](#page-7-2) were taken into account, the second runlist would not have been overlooked, could have been preempted, and the safety of the system preserved.

