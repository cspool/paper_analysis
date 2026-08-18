# Five-Minute Rule 40 Years Later: A First-Principles Revisit for Modern Memory Hierarchy

Tong Zhang ScaleFlux, CA, USA Vikram Sharma Mailthody NVIDIA, IL, USA

Fei Sun ScaleFlux, CA, USA

Linsen Ma ScaleFlux, CA, USA Chris J. Newburn NVIDIA, IL, USA

Teresa Zhang

Yang Liu

Jiangpeng Li

Hao Zhong

Wen-Mei Hwu

Stanford University, CA, USA ScaleFlux, CA, USA ScaleFlux, CA, USA ScaleFlux, CA, USA NVIDIA, IL, USA

*Abstract*—In 1987, Jim Gray and Gianfranco Putzolu introduced the five-minute rule, a simple, storage-memory-economicsbased heuristic for deciding when data should live in DRAM rather than on storage. Subsequent revisits to the rule largely retained that economics-only view, leaving host costs, feasibility limits, and workload behavior out of scope. This paper revisits the rule from first principles, integrating host costs, DRAM bandwidth/capacity, and physics-grounded models of SSD performance and cost, and then embedding these elements in a constraint- and workload-aware framework that yields actionable provisioning guidance. We show that, for modern AI platforms, especially GPU-centric hosts paired with ultra-high-IOPS SSDs engineered for fine-grained random access, the DRAM↔flash caching threshold collapses from minutes to a few seconds. This shift reframes NAND flash memory as an *active data tier* and exposes a broad research space across the hardware–software stack. We further introduce MQSim-Next, a calibrated SSD simulator that supports validation and sensitivity analysis and facilitates future architectural and system research. Finally, we present two concrete case studies that showcase the software system design space opened by such memory hierarchy paradigm shift. Overall, we turn a classical heuristic into an actionable, feasibility-aware analysis and provisioning framework and set the stage for further research on AI-era memory hierarchy.

*Index Terms*—memory hierarchy, solid-state drive (SSD), storage systems, performance modeling, data placement

