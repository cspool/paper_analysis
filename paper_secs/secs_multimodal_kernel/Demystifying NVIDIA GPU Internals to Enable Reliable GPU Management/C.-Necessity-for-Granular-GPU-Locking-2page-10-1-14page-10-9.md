# *C. Necessity for Granular GPU Locking [\[2\]](#page-10-1), [\[14\]](#page-10-9)*

Glenn *et al.* [\[2\]](#page-10-1), [\[14\]](#page-10-9) propose managing the GPU by making each copy and compute engine a lockable resource, such that existing resource-management analysis can be applied to GPU-using tasks.

[R8](#page-7-4) and [R6](#page-6-6) show that the presence of multiple copy engines does not guarantee their independence, as would be required to treat them as separately lockable resources. This lacking copy independence is the key problem with a granular locking approach.

Problematic assumptions. Glenn *et al.* assume that if a GPU has at least two copy engines, copies in independent directions can proceed independently—a contradiction of [R8](#page-7-4). Furthermore, they assume that—no matter the number of copy engines—those copy engines are scheduled independently of the compute engine. [R6](#page-6-6) contradicts this; copy and compute engines may share a runlist, making scheduling non-independent. Other assumptions of that work are now outdated, such as the non-preemptability of copy engines, but we focus only on the overly-optimistic assumptions, rather than the overlypessimistic ones.

Counter-example. Consider the system of Fig. [10](#page-7-3) on the RTX 6000 Ada: one CPU-to-GPU copy+graphics task, and another GPU-to-CPU copy task on a two-copy-engine-containing system. Given that CUDA reports two copy engines on this system, the following locks would be created: Lock 1 for CPUto-GPU copies, Lock 2 for GPU-to-CPU copies, and Lock 3 for compute/graphics work. Task 2 could freely acquire Lock 2 for its GPU-to-CPU copies at the same time that Task 1 holds Lock 1 or Lock 3—no locks are contended for, and so management is a no-op.

However, due to how the visible copy engines are mapped to the underlying copy hardware, the copies of Task 1 and Task 2 would contend for access to a single PCE, taking double the time expected. This compromises execution time bounds, leading to unreliably-met deadlines. If hardware mappings were taken into account as required by [R8](#page-7-4)—perhaps extracted via our nvdebug tool—only one copy engine lock would have been created and per-engine mutual exclusion would be restored.

