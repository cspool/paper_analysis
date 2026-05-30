# <span id="page-1-1"></span>2 Related Work

This section reviews existing related work on GPU cluster management (Section [2.1](#page-1-2) and GPU power monitoring (Section [2.2\)](#page-2-1).

## <span id="page-1-2"></span>2.1 GPU Cluster Management

Cloud data centers provide access to GPUs through various service models, ranging from IaaS offerings to MLaaS platforms. These differences influence how GPUs are allocated to workloads, which we briefly outline below.

2.1.1 Temporal allocation. In its basic form, sharing a GPU device is similar to scheduling a single CPU core. Each process has sequential exclusive access to the device depending on the scheduler (e.g., round-robin). Access is managed through time slices and context switches.

While this behavior is the default one when different processes require access to the CUDA API, processes do not get direct access to the accelerator in the cloud. Instead, they are encapsulated within their software stack through containers or Virtual Machines (VMs). The underlying virtualization stack may affect temporal sharing. With Kubernetes, the number of concurrent processes is controlled through GPU replica settings using the GPU Operator driver [\[8\]](#page-13-6). This allows exposing more GPUs than are physically available, using a ratio that can be specified at the cluster or node scale [\[9\]](#page-13-7), similarly to what exists for CPU oversubscription [\[10](#page-13-8)[–12\]](#page-13-9). Outside the Kubernetes environment, this time-slice sharing can be controlled with NVIDIA vGPU technology [\[13\]](#page-13-10), where GPUs can lead to multiple profiles exposed to VMs.

2.1.2 Spatial allocation. While time-slice sharing offers a way to divide GPU resources, it may not be optimal for specific workloads, as there is no guarantee that the entire GPU will be utilized during each slice. Several spatial allocation solutions have been proposed to tackle this challenge, enabling different processes to run simultaneously on distinct resources within the accelerator.

Multi-Process Service (MPS) is one such solution that relies on cooperation at the API level. Implemented through a client-server architecture, MPS enables multiple applications (e.g., MPI-based workloads) to share a single CUDA context, facilitating parallel execution of their tasks on the GPU [\[14\]](#page-13-11).

However, in cloud environments, the cooperation of the workload cannot always be assumed. In this case, Multi-Instance GPU (MIG) can partition the GPU into multiple independent instances, each with dedicated CUDA cores, L2 cache, and DRAM area. It provides stronger isolation guarantees than MPS, making it suitable for multi-tenant environments, but it is not available on all architectures. Each MIG instance can be attached to a container using NVIDIA's container drivers or a VM using NVIDIA vGPU technology.

2.1.3 Pass-through allocation. One standard method to enable direct access to hardware in virtualized environments is through Virtual Function I/O (VFIO), a Linux kernel module that allows virtual machines to interact directly with PCI devices. This mechanism ensures hardware isolation by relying on Input-Output Memory Management Unit (IOMMU), which enforces strict memory protection and prevents unauthorized access between virtualized workloads. It remains a widely used approach due to its privacy advantages (by leveraging VMs), performance benefits [15], and direct access to native hardware features [16]. Multi-GPUs servers can host multiple VMs, allowing them to share the server's CPU resources while providing direct access to accelerators.

#### <span id="page-2-1"></span>2.2 GPU Power Monitoring

GPU power monitoring is frequently performed using tools that access the NVML API, such as nvidia-smi or DCGM [17–19]. Readings are claimed to be accurate within 5 watts according to NVIDIA documentation, but these figures may not always hold in micro-benchmarks [20]. To mitigate this, the measurements in our study are typically taken over more extended periods (e.g., modeling batch jobs).

JoularJX [21] can monitor live power consumption for processes on GPUs. It observes the consumption exposed by the nvidia-smi tool and attributes it to a given process. However, the case where multiple processes share the same GPU is not addressed, nor is it applied to hyperscale contexts.

Several power modeling frameworks for various GPU architectures have been proposed, such as GPUWattch [22] and AccelWattch [23]. These tools primarily focus on simulation rather than live power tracking. In contrast, our goal is to model the energy consumption of jobs in shared environments using measurements from the actual device, similar to existing models for CPU process power consumption [6, 24].

We believe there is a mismatch between how GPU power monitoring is conducted and how hyperscalers operate GPUs. In scenarios where cloud providers need to monitor the power consumption of rented GPUs—such as for load balancing, carbon footprint calculation [25, 26], and other use cases—there is often no transparent methodology for monitoring jobs in shared environments.

In this paper, we aim to address these gaps by leveraging simple, practical principles for GPU power monitoring in shared contexts.

