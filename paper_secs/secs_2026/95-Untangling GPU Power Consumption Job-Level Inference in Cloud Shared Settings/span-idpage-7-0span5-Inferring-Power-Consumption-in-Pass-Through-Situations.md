# <span id="page-7-0"></span>5 Inferring Power Consumption in Pass-Through Situations

While sharing GPUs may be beneficial in periodic usage or, as demonstrated, for some kinds of workloads in terms of energy efficiency, others require exclusive usage of the device. In that case, virtualization remains in use, as a multi-GPUs server is still shared, but processes (containers or VMs) may get exclusive access to an accelerator. This approach is particularly prevalent in use cases such as virtual desktops, gaming, and IaaS.

Particularly, assigning GPUs to VM through PCIe passthrough offers near-native performance while preserving the key benefits of virtualization, such as isolation and security. However, this configuration introduces a significant challenge for cloud providers. Once a GPU is passed through to a virtual machine, the host loses all visibility over the PCIe device, including monitoring capabilities. In other words, from the cloud provider's perspective, the GPU operates as a black box with no direct means of tracking its usage.

This section explores a method to infer GPU usage in passthrough scenarios. We propose a monitoring technique to restore visibility into energy usage while maintaining the integrity and security guarantees of VFIO-based GPU assignment. We first introduce our approach (Subsection 5.1), followed by detailing our experimental protocol (Subsection 5.2) and evaluating it (Subsection 5.3).

#### <span id="page-8-0"></span>5.1 Principle

GPU power monitoring in pass-through environments can be implemented through several approaches. However, ensuring the privacy of VMs rules out solutions that involve reimplementing the VFIO module. Additionally, approaches that would require cooperation from clients (e.g., installing an agent) — an assumption that cannot be guaranteed in multi-tenant environments — are also not feasible.

Instead, we adopt an entirely non-intrusive technique. Our approach leverages the various embedded sensors available on modern server motherboards. While GPU-integrated sensors, typically exposed through manufacturer-specific drivers, become inaccessible in a pass-through configuration, we investigate whether temperature sensors from the motherboard can serve as reliable proxies for GPU power consumption.

Given the high and continuously increasing TDP values of modern GPUs, temperature emerges as a natural candidate for indirect power estimation. While previous work has modeled CPU usage from temperature [39], to our knowledge, this is the first study to consider temperature as a proxy for GPUs. We specifically evaluate the ability of *Intelligent* Platform Management Interface (IPMI) temperature sensors to approximate GPU energy consumption. In contrast to the nvidia-smi interface, which becomes unavailable in passthrough scenarios, IPMI sensors remain accessible at the host level, making them particularly relevant for our use case. They are, however, difficult to exploit due to their low sampling frequency, limited precision, unknown placement, and sensitivity to thermal interference between GPUs assigned to different tenants. We designed our experimental protocol to evaluate their viability for GPU power monitoring in multi-tenant environments.

#### <span id="page-8-1"></span>5.2 Experimental Protocol

Our experimental setup is designed to address two key Research Questions (RQs).

- Can GPU usage levels and power consumption be accurately inferred from generic IPMI temperature sensors? By "accurately", we aim not only to distinguish between idle and fully loaded states but also to identify intermediate levels of GPU utilization.
- Can interference between GPUs temperature readings be mitigated? In hyperscale cloud environments, GPUs are deployed in high-density configurations, typically ranging from 2 to 8 GPUs per server. As a result, IPMI temperature sensors may detect heat signatures from multiple accelerators, potentially introducing cross-interference in the measurements. Understanding whether this interference can be controlled is essential for ensuring reliable power estimations.

To answer these questions, we conducted a series of controlled experiments, leveraging different workload intensities and GPU placements within a server chassis. By analyzing the correlation between GPU activity levels and IPMI-reported temperature variations, we evaluate the feasibility and accuracy of using motherboard sensors as a proxy for power monitoring in pass-through settings.

**5.2.1 Build a workload.** We built a GPU workload stressing different GPUs to various levels. Specifically, we selected n workload stress levels, ranging from n=2 (a GPU is either entirely idle or fully loaded) to n=4 (a GPU can be used at 0%, 30%, 60% or 100%). We tested all combinations of GPUs and stress levels. The total number of combinations on a machine with m GPUs is  $(n^m)$ . For a machine having 4 GPUs under 4 distinct levels, this results in 256 different combinations.

The GPU workload is composed of GPU-burn, and the workload level is controlled through GIs (e.g., the 30% usage level is obtained by using 2 compute slices of the 7,  $2/7 \simeq 29\%$  of the SMs). Each combination is run for 5 minutes, during which the power is periodically (5-second interval) read for each GPU instance.

**5.2.2 Identifying appropriate IPMI sensors.** Modern servers typically integrate multiple IPMI sensors, each positioned at different locations on the motherboard. Due to these physical placements, some sensors are closer to specific GPUs than others, potentially capturing temperature variations more accurately. We aim to identify which IPMI sensor exhibits the highest correlation with the power consumption of each GPU unit.

All IPMI sensors recognized as temperature sensors by the ipmitool agent were monitored to achieve this. Then, for each sensor, we computed the Pearson correlation coefficient with the power consumption of each GPU (as exposed by the device, read from nvidia-smi), assuming a linear relationship between power consumption and heat generation

<span id="page-9-1"></span>![](_page_9_Figure_2.jpeg)

**Figure 9.** Pearson correlation between temperature readings from IPMI sensors and the power consumption of four A100 GPUs, restricted to sensors whose labels mention GPU

(as expected from Ohm's law). The correlation results are illustrated in Figure 9. For readability, we display only the sensors labeled with GPU, although correlations were computed using all temperature sensors available.

Although servers are equipped with multiple IPMI sensors, only a subset of them is significantly influenced by GPU power consumption. A coefficient close to 1 indicates a strong positive correlation, a coefficient near -1 suggests a strong negative correlation, while a coefficient around 0 implies no correlation. The most relevant temperature sensor for each GPU power consumption is then selected for our analysis.

#### <span id="page-9-0"></span>5.3 Results

We now seek to address our two previously defined research questions: the precision of the IPMI-based usage model and the potential interference caused by thermal effects from neighboring GPUs in different hardware configurations.

#### 5.3.1 Precision on 4xGPUs air-cooled configurations.

To evaluate precision, we tested all possible combinations of the following stress levels: 0%, 30%, 60%, and 100% on the four A100 GPUs of an Apollo 6500 Gen10 Plus server. Figure 10 presents the results as a kernel density estimate.

As Figure 10 shows, each compute level can result in a range of temperature readings. For instance, the idle state (0% compute level) can produce temperature values between 30°C and 60°C, notably depending on other GPUs state as

we captured all combinations of usage. We found that part of the overlap of temperature ranges for a given usage was caused by the closest GPU neighbor, though this is not a symmetrical relation. For example, if a GPU A's temperature is influenced by a GPU B, the opposite is generally not true, as fans direct the airflow in a single direction.

To automatically correct this, we identify, for each GPU, the closest neighbor by selecting the second most correlated sensor to the given GPU among those identified as good proxies (the first one being itself). Then, we fit a linear regression between the neighbor's temperature delta (*current – min*) and the temperature observed on the GPU under study. If the coefficient is significantly positive, we correct the sensor value by accounting for the neighbor's delta. The entire process is described in Algorithm 1. Note that the  $\beta$  coefficient of the linear regression between the neighbor's temperature delta and the sensor value represents the minimum value of the sensor and can be discarded from the correction formula. Under positive correlation,  $\alpha$  value was typically around 0.3 across the various servers tested. Applying this correction significantly reduces the overlap between GPU usage levels, as shown in Figure 10.

Other approaches exist for identifying and modeling cross-interference between signal sources, such as the Expectation-Maximization Algorithm [40]. Here, we adopt a simpler, guided process, since the temperature influence is primarily dominated by airflow from the closest neighbor.

