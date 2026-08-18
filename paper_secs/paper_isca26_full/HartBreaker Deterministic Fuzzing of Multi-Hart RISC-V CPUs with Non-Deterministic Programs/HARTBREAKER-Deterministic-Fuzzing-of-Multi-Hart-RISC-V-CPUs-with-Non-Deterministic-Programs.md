# HARTBREAKER: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs

Quentin Bordier *ETH Zurich* bordierq@ethz.ch

Tobias Kovats *ETH Zurich* tkovats@ethz.ch

Flavien Solt *UC Berkeley* flavien.solt@berkeley.edu Kaveh Razavi *ETH Zurich* kaveh@ethz.ch

*Abstract*—Hardware bugs threaten the correctness and security of modern CPUs. Relying on a deterministic correct baseline, pre-silicon fuzzing has proven to be an effective strategy for discovering deviations from correct behavior (i.e., bugs) in single-core CPUs. Modern CPUs, however, often feature multiple cores with complex interconnects that implement communication channels such as inter-processor interrupts or shared memory. Is it possible to effectively fuzz multicore CPUs despite their inherent non-deterministic operations?

We make a key observation that multi-hart interactions may result in non-deterministic data flows, control flows, or combinations thereof. An efficient fuzzing campaign needs to manage this non-determinism without limiting the exploration of the possible state space that may lead to bugs. Our new multi-hart RISC-V fuzzer, called HARTBREAKER, achieves this with a judicious use of three *determinism anchors*: control- and data-flow anchors enable non-deterministic control- and dataflow interactions between harts while ensuring a correct execution of multi-hart test programs, achieving high testing throughput and simplified bug detection. Synchronization anchors bound the non-deterministic window across harts, enabling HARTBREAKER to detect bugs that do not contaminate the control flow. We test HARTBREAKER on five multi-hart designs, namely Rocket, BOOM, Toooba, NaxRiscv and XiangShan. HARTBREAKER discovers five new concurrency bugs in these designs.

