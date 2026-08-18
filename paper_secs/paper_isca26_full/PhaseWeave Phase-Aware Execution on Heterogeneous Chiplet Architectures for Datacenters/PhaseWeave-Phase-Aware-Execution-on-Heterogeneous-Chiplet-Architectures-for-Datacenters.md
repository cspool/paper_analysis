# PhaseWeave: Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters

Joshua Kim *The University of Texas at Austin* Austin, USA joshk@cs.utexas.edu

Chaojie Zhang *Microsoft Azure Research – Systems* Redmond, USA chaojiezhang@microsoft.com

´Inigo Goiri ˜ *Microsoft Azure Research – Systems* Redmond, USA inigog@microsoft.com

Christopher J. Rossbach *The University of Texas at Austin & Microsoft* Austin, USA rossbach@cs.utexas.edu

Jovan Stojkovic *The University of Texas at Austin & Meta* Austin, USA jovan@cs.utexas.edu

*Abstract*—Modern datacenter applications execute in clearly distinct phases, driven by modular microservice architectures, auxiliary "datacenter tax" operations, and diverse execution blocks within the service logic. Our characterization of datacenter workloads reveals that these millisecond-scale phases exhibit recurring patterns and have varying sensitivities to hardware resources such as core frequency, network bandwidth, and memory bandwidth. This fine-grained intra-application heterogeneity leads to inefficiencies when running on traditional homogeneous server architectures, which cannot adapt to dynamic and phasespecific resource demands. As a result, datacenter operators face a trade-off between overprovisioning resources and suffering performance bottlenecks during critical phases.

We propose *PhaseWeave*, a heterogeneous multiple-chiplet server architecture where different chiplet types are optimized for different workload phases (*e.g.*, compute-, memory-, or network-phases). PhaseWeave transparently predicts changes in program phases using hardware counters and the distribution of system calls. Then, it uses OS signals to migrate workloads to chiplets that are best suited for the next phase. By dynamically and transparently steering execution across specialized chiplets, PhaseWeave improves resource utilization and performance without requiring changes to user code. Full-system simulations with a diverse set of datacenter workloads show that PhaseWeave reduces tail latency of datacenter applications by 65% at high loads, increases throughput by 1.6×, and improves Performance/Watt by 1.9× compared to homogeneous iso-area baseline.

*Index Terms*—Datacenter Servers, Microservices, Heterogeneous Architecture, Phase Detection

