# Abstract

As the demand for AI-driven workloads increases, the energy consumption of Graphics Processing Units (GPUs) devices has come under intense scrutiny, particularly in hyperscale data centers where large numbers of accelerators are centralized and leased to diverse clients.

In the context of cloud hyperscalers, GPUs power monitoring presents several challenges that vary depending on the product offered. The monitoring capabilities of physical devices may be limited or even absent for some products. However, given the substantial energy demands of GPUs, power monitoring is essential for both cloud providers and clients. Operators require tools to manage power distribution effectively, such as balancing workloads across Power Distribution Units (PDUs), while clients need visibility into power usage to optimize their workloads for energy efficiency.

To address these challenges, we propose methods for estimating the energy consumption of jobs running on GPU devices in cloud environments, spanning from shared and managed offerings like ML-as-a-Service (MLaaS) to less managed products (e.g., Infrastructure-as-a-Service (IaaS)). Our models demonstrate the benefits of sharing GPUs for small AI workloads, as well as the current sub-optimal utilization

[ACM acknow](https://creativecommons.org/licenses/by/4.0)ledges that this contribution was authored or co-authored by an employee, contractor or affiliate of a [national government. As such, the Government retains a](https://creativecommons.org/licenses/by/4.0)  nonexclusive, royalty-free right to publish or reproduce this article, or to allow others to do so, for Government purposes only.

EUROSYS '26, [Edinburgh,](https://doi.org/10.1145/3767295.3769333) Scotland Uk © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 https://doi.org/10.1145/3767295.3769333

of GPUs in cloud hyperscalers, based on insights from an IaaS GPU cluster.

CCS Concepts: • Hardware → Power estimation and optimization; • Computing methodologies → Modeling methodologies; • Computer systems organization → Cloud computing; Parallel architectures.

Keywords: Cloud computing, GPU, Power consumption

#### ACM Reference Format:

Pierre Jacquet, Maxime Agusti, Eddy Caron, Camille Coti, Marcos Dias De Assunção, Laurent Lefèvre, and Anne-Cécile Orgerie. 2026. Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings. In European Conference on Computer Systems (EU-ROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, [17](#page-16-0) pages. <https://doi.org/10.1145/3767295.3769333>

