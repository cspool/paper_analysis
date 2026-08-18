# *C. Software-Hardware Interaction*

Beyond RTL simulation, the software domain has developed a rich ecosystem of testing tools and methodologies [5], [18], [23], [47]. Widely adopted unit-testing frameworks and fuzzing engines [15], [54] lower the barrier to systematic testing. Newer models such as crowd-sourced testing further scale defect discovery [4], and community-driven projects have resolved millions of bugs through such processes [14], [25], [27]. Industry reports emphasize the benefits of involving software developers in hardware bring-up and validation [19], [32], and recent work explores reusing software testing infrastructure for hardware verification [50], [51]. For example, Minotaur adapts concepts from software testing to analyze software vulnerability to hardware-induced errors [31], while Instruction-Level Abstraction (ILA) [22] provides a formal instruction-grained model of SoC behavior. These trends make it attractive to execute part or all of the verification environment as ordinary software, rather than solely as HDL.

## *D. Cocotb and coroutine based software framework*

Cocotb [37] is a representative simulator-centric framework that executes test logic as Python coroutines within an RTL simulator process. The simulator invokes cocotb via registered foreign interface callbacks at points determined by the kernel's event scheduling, and each invocation grants the software side a brief execution window before returning control to simulation. Cocotb then runs a callback-driven dispatcher that maps kernel notifications to ready triggers, resumes the corresponding coroutines, and performs signal reads and writes through the simulator foreign interface. The dispatcher is reactive to simulator callbacks and does not own time advancement or define observation boundaries, so timing, ordering, and observability semantics are inherited from the simulator kernel and software execution is split into short callback-bounded slices. In addition, signal access is typically dynamic and name based, so developers often identify signals by hierarchical paths and obtain handles through lookup or traversal, which requires RTL knowledge and manual access RTL code.

#### III. MOTIVATION AND CHALLENGES

The growing complexity of modern hardware designs, particularly in processor architectures, presents new challenges for hardware-software co-verification tools. This complexity appears in several forms: interleaved and concurrent timing behaviors, dependence on extensive reference models, and difficulty of inferring internal states through external I/O.

These challenges lead to three core requirements for verification tools: timing precision, functional integration, and observability. First, precise temporal control over ports with multi-cycle interactions is crucial for driving complex scenarios. Second, integration of existing components developed with hardware-centric methodologies is needed to avoid costly redevelopment or fragile manual integration via serialized channels. Third, strong debuggability is required to expose internal design states without sacrificing execution efficiency.

However, current hardware-software co-verification tools fall short of these requirements:

## *A. Timing Model in Software*

When verification logic runs in software, the testbench timing model must satisfy two core properties. It needs enough expressiveness to describe event-driven protocols and multiple concurrent requests, and it must preserve timing correctness, keeping a stable cycle-accurate ordering of updates that does not depend on simulator-specific delta cycles. Further, compatibility with modern asynchronous runtimes allows reuse of software testing libraries that rely on asynchronous controls, enabling broader tooling support with little additional effort.

Existing software-based frameworks mostly expose two kinds of interfaces based on simulator execution primitives. Cycle-accurate control (*step-peek*) keeps timing fully under software control, avoiding simulator-specific delta cycles and remaining naturally compatible with host asynchronous runtimes. However, it advances time only in coarse cyclesized steps, which limits the expressiveness of describing asynchronous or mid-cycle behavior.

Callback-driven interfaces, typified by Cocotb, execute user code through callbacks triggered inside the simulator. Because the simulator controls *when* each callback fires, the software may be invoked before all dependent signals have finished updating. Even a simple combinational path such as b = f(a) can cause software to read a stale value of b if the callback is triggered immediately after a changes [13], [20]. In addition, simulator-controlled callback loops are not compatible with software-native asynchronous runtimes, since the simulator, rather than software, owns the control flow, making it difficult to integrate with external asynchronous libraries [6].

Thus, cycle-accurate control gives strong correctness and native compatibility by keeping all timing under software control, whereas callback-driven interfaces improve expressiveness but delegate timing to the simulator, weakening both correctness and compatibility. The challenge is therefore to construct a *software-native timing model* that preserves software-controlled correctness and ecosystem compatibility while restoring hardware-style event-level expressiveness within a clean, cycle-accurate abstraction (Section V-A).

#### *B. Reusing the Hardware Verification Components*

There exist many mature verification IPs in hardware design, such as reference models and stimulus generators [33], [38], [45], that significantly accelerate the verification. However, these tools depend on hardware-side events and transactions, which are isolated by the software–hardware boundary and thus inaccessible from the software environment. Reusing such VIPs thus introduces two additional requirements.

Event Synchronization. The hardware simulator execution flow is driven by a sequence of events. Due to the separation between the software environment and the hardware simulator, events occurring within one domain are invisible to the other. Achieving cross-domain execution flow requires a mechanism capable of synchronizing event states between domains.

Transaction Scheduling. The software and hardware environments must execute alternately in a coordinated manner. Although DPI-C provides an immediate bidirectional communication interface between the two domains, actual VIP transactions often span multiple cycles. Without proper scheduling, CPU contention or even deadlocks may occur.

To address these challenges, we introduce a transparent mapping mechanism that enables bidirectional event synchronization and coordinates transaction scheduling between software and hardware, while preserving software programming styles. The implementation is detailed in §V-B.

## *C. Debugging Performance Optimization*

Verifying large-scale designs, such as multi-core SoCs, requires both high performance and strong debuggability. Performance ensures the efficient execution of long-running tests, while debuggability demands visibility into internal module states, as relying solely on final outputs may delay or obscure the detection of errors. However, existing simulators struggle to satisfy both requirements simultaneously.

Even state-of-the-art tools [12], [28], [36], [42] face this trade-off. The root of this limitation originates from the

![](_page_4_Figure_0.jpeg)

Fig. 3: UnityChip Verification overview. The left shows the verification and packaging workflow, while the right shows the runtime platform structure, illustrating the interaction between the verification environment and the packaged design. Dashed boxes indicate optional supports for UVM VIP.

handling of intermediate states. Merging them improves performance but reduces observability, while preserving them increases visibility at the expense of efficiency. Debugging facilities such as VPI and DPI introduce extra intermediate states, with VPI further restricting key optimizations.

To overcome this limitation, we introduce a method that accesses pointers to underlying circuit states without instrumenting additional intermediate signals. Corresponding signal values are computed on demand, with computation selectively activated according to signal relevance. This design achieves fine-grained control over debugging visibility while maintaining high performance. Details are presented in Section V-C.

# *C. Software-Hardware Interaction*

Beyond RTL simulation, the software domain has developed a rich ecosystem of testing tools and methodologies [5], [18], [23], [47]. Widely adopted unit-testing frameworks and fuzzing engines [15], [54] lower the barrier to systematic testing. Newer models such as crowd-sourced testing further scale defect discovery [4], and community-driven projects have resolved millions of bugs through such processes [14], [25], [27]. Industry reports emphasize the benefits of involving software developers in hardware bring-up and validation [19], [32], and recent work explores reusing software testing infrastructure for hardware verification [50], [51]. For example, Minotaur adapts concepts from software testing to analyze software vulnerability to hardware-induced errors [31], while Instruction-Level Abstraction (ILA) [22] provides a formal instruction-grained model of SoC behavior. These trends make it attractive to execute part or all of the verification environment as ordinary software, rather than solely as HDL.

## *D. Cocotb and coroutine based software framework*

Cocotb [37] is a representative simulator-centric framework that executes test logic as Python coroutines within an RTL simulator process. The simulator invokes cocotb via registered foreign interface callbacks at points determined by the kernel's event scheduling, and each invocation grants the software side a brief execution window before returning control to simulation. Cocotb then runs a callback-driven dispatcher that maps kernel notifications to ready triggers, resumes the corresponding coroutines, and performs signal reads and writes through the simulator foreign interface. The dispatcher is reactive to simulator callbacks and does not own time advancement or define observation boundaries, so timing, ordering, and observability semantics are inherited from the simulator kernel and software execution is split into short callback-bounded slices. In addition, signal access is typically dynamic and name based, so developers often identify signals by hierarchical paths and obtain handles through lookup or traversal, which requires RTL knowledge and manual access RTL code.

#### III. MOTIVATION AND CHALLENGES

The growing complexity of modern hardware designs, particularly in processor architectures, presents new challenges for hardware-software co-verification tools. This complexity appears in several forms: interleaved and concurrent timing behaviors, dependence on extensive reference models, and difficulty of inferring internal states through external I/O.

These challenges lead to three core requirements for verification tools: timing precision, functional integration, and observability. First, precise temporal control over ports with multi-cycle interactions is crucial for driving complex scenarios. Second, integration of existing components developed with hardware-centric methodologies is needed to avoid costly redevelopment or fragile manual integration via serialized channels. Third, strong debuggability is required to expose internal design states without sacrificing execution efficiency.

However, current hardware-software co-verification tools fall short of these requirements:

## *A. Timing Model in Software*

When verification logic runs in software, the testbench timing model must satisfy two core properties. It needs enough expressiveness to describe event-driven protocols and multiple concurrent requests, and it must preserve timing correctness, keeping a stable cycle-accurate ordering of updates that does not depend on simulator-specific delta cycles. Further, compatibility with modern asynchronous runtimes allows reuse of software testing libraries that rely on asynchronous controls, enabling broader tooling support with little additional effort.

Existing software-based frameworks mostly expose two kinds of interfaces based on simulator execution primitives. Cycle-accurate control (*step-peek*) keeps timing fully under software control, avoiding simulator-specific delta cycles and remaining naturally compatible with host asynchronous runtimes. However, it advances time only in coarse cyclesized steps, which limits the expressiveness of describing asynchronous or mid-cycle behavior.

Callback-driven interfaces, typified by Cocotb, execute user code through callbacks triggered inside the simulator. Because the simulator controls *when* each callback fires, the software may be invoked before all dependent signals have finished updating. Even a simple combinational path such as b = f(a) can cause software to read a stale value of b if the callback is triggered immediately after a changes [13], [20]. In addition, simulator-controlled callback loops are not compatible with software-native asynchronous runtimes, since the simulator, rather than software, owns the control flow, making it difficult to integrate with external asynchronous libraries [6].

Thus, cycle-accurate control gives strong correctness and native compatibility by keeping all timing under software control, whereas callback-driven interfaces improve expressiveness but delegate timing to the simulator, weakening both correctness and compatibility. The challenge is therefore to construct a *software-native timing model* that preserves software-controlled correctness and ecosystem compatibility while restoring hardware-style event-level expressiveness within a clean, cycle-accurate abstraction (Section V-A).

#### *B. Reusing the Hardware Verification Components*

There exist many mature verification IPs in hardware design, such as reference models and stimulus generators [33], [38], [45], that significantly accelerate the verification. However, these tools depend on hardware-side events and transactions, which are isolated by the software–hardware boundary and thus inaccessible from the software environment. Reusing such VIPs thus introduces two additional requirements.

Event Synchronization. The hardware simulator execution flow is driven by a sequence of events. Due to the separation between the software environment and the hardware simulator, events occurring within one domain are invisible to the other. Achieving cross-domain execution flow requires a mechanism capable of synchronizing event states between domains.

Transaction Scheduling. The software and hardware environments must execute alternately in a coordinated manner. Although DPI-C provides an immediate bidirectional communication interface between the two domains, actual VIP transactions often span multiple cycles. Without proper scheduling, CPU contention or even deadlocks may occur.

To address these challenges, we introduce a transparent mapping mechanism that enables bidirectional event synchronization and coordinates transaction scheduling between software and hardware, while preserving software programming styles. The implementation is detailed in §V-B.

## *C. Debugging Performance Optimization*

Verifying large-scale designs, such as multi-core SoCs, requires both high performance and strong debuggability. Performance ensures the efficient execution of long-running tests, while debuggability demands visibility into internal module states, as relying solely on final outputs may delay or obscure the detection of errors. However, existing simulators struggle to satisfy both requirements simultaneously.

Even state-of-the-art tools [12], [28], [36], [42] face this trade-off. The root of this limitation originates from the

![](_page_4_Figure_0.jpeg)

Fig. 3: UnityChip Verification overview. The left shows the verification and packaging workflow, while the right shows the runtime platform structure, illustrating the interaction between the verification environment and the packaged design. Dashed boxes indicate optional supports for UVM VIP.

handling of intermediate states. Merging them improves performance but reduces observability, while preserving them increases visibility at the expense of efficiency. Debugging facilities such as VPI and DPI introduce extra intermediate states, with VPI further restricting key optimizations.

To overcome this limitation, we introduce a method that accesses pointers to underlying circuit states without instrumenting additional intermediate signals. Corresponding signal values are computed on demand, with computation selectively activated according to signal relevance. This design achieves fine-grained control over debugging visibility while maintaining high performance. Details are presented in Section V-C.

