## **DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions** 

Ahmed Alquraan[∗] University of Waterloo 

Abdelrahman Baba[∗] University of Waterloo 

Rafael Mendes da Silva Microsoft Research 

Sameh Elnikety Paul Batum Yan Chen Microsoft Research Microsoft Microsoft Hamid Henry Sa! Seth Fine Samer Al-Kiswany Microsoft Microsoft University of Waterloo 

## **Abstract** 

Azure Functions maintains pools of pre-warmed containers to avoid the high container-allocation latency. The size of a pool is important: a pool that is too small leads to high allocation latency, whereas a pool that is too large wastes resources and increases cost. Service providers typically oversize pools to meet service-level objectives (SLOs). Our !ndings indicate that the cost of maintaining pre-warmed container pools dominates the overall platform cost, motivating the need for e"ective pool management strategies. 

We characterize container-allocation traces from Azure Functions, revealing key !ndings: the demand for container allocation is highly bursty, while the supply of new containers is virtually unlimited but may have long delays. Traditional resource management approaches, which rely on prediction or reactive techniques, fail to reduce costs while meeting the target SLO due to the bursty nature of the workload. These insights lead us to develop a new statistical, data-driven pool optimization method that uses historical traces to compute the size of each pool. Our evaluation shows that the proposed method meets the target SLO while reducing the operational cost by 41% compared to the static approach employed earlier in the production platform. 

## _**CCSConcepts:**_ • **Computersystemsorganization** → **Cloud computing** ; • **Social and professional topics** → **Pricing and resource allocation** . 

_**Keywords:**_ serverless computing, cloud platforms, resource management, performance optimization 

## **ACM Reference Format:** 

AhmedAlquraan,AbdelrahmanBaba,RafaelMendesdaSilva,Sameh Elnikety,PaulBatum,YanChen,HamidHenrySa!,SethFine,andSamer 

∗Both authors contributed equally to the paper 

ThisworkislicensedunderaCreativeCommonsAttribution4.0International License. _EUROSYS ’26, Edinburgh, Scotland Uk_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 h!ps://doi.org/10.1145/3767295.3769350 

Al-Kiswany. 2026. DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions. In _21st European Conference on Computer Systems (EUROSYS ’26), April 27–30, 2026, Edinburgh, Scotland Uk._ ACM, New York, NY, USA, 17 pages. h!ps://doi.org/10.1145/3767295. 3769350 

## **1 Introduction** 

Serverless computing o"ers an attractive computing model for developers, allowing them to express their application as event-driven functions. The provider is responsible for deploying, maintaining, and managing resources. In addition, it o"ers automatic dynamic scalability and a !ne-grained billing model, in which users are charged only for the actual execution time. Leading serverless platforms, such as Azure Functions [1], AWS Lambda [2], and Google Cloud Run [3], support a broad range of runtime environments (e.g., Python, .NET, and Node.js) and diverse resource con!gurations (e.g., 1-core and 2-core containers). 

TheAzureFunctionsplatformiscomposedofasetofvirtual machines (VMs). Each VM hosts a set of smaller nested VMs (NVMs), and each nested VM runs a single container with a speci!c runtime. The platform uses nested virtualization (i.e., each container runs in a nested VM) to isolate tenants. The platform does not employ resource oversubscription: each nested VM reserves exclusive cores and memory from the corresponding VM. For the remainder of this paper, we use _VM_ to refer to the VM that hosts nested VMs, and _container_ to refer to both the container and its associated nested VM. 

A user registers a serverless function and speci!es its runtime version, container size, and concurrency settings. When the user invokes the function for the !rst time, a containerallocationrequestisgenerated.Theserverlessplatformful!lls the allocation request by providing a container with the requested runtime and size from a pool of pre-warmed containers. A _pre-warmed container_ is a container that is initialized with a runtime but not with the user code. After processing the user invocation and as soon as the container becomes idle, the container is deleted and its resources are recycled. Notice that the concurrency setting allows a single container to process multiple concurrent invocations, which reduces the cost 

1281 

**==> picture [217 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
VM Pool Container Pools<br>VM1 VM2<br>NVM1 NVM2 Py .NET .NET Py ..<br>NVM3 .. Java Py Java .NET<br>**----- End of picture text -----**<br>


**Figure 1.** The architecture of the serverless platform. 

per invocation for the user and, as a side e"ect, extends the lifetime of the container. 

The platform aims to satisfy every allocation request immediately using a pre-warmed container. If the containerallocation request is not satis!ed immediately, we call this an _allocation failure_ . A failed allocation request is handled by creating a new container, which requires allocating resources from a VM. The ful!llment latency of the container-allocation request depends on the availability of required resources. If an existing VM has su#cient available cores and memory, a new container can be immediately created, resulting in a ful!llmentlatencyrangingfromhundredsofmillisecondstoa few seconds. We refer to this scenario as a _container-allocation failure_ . In contrast, if no existing VM has su#cient available cores and memory, the platform must provision a new VM, leading to a much higher ful!llment latency, typically on the order of minutes. We refer to this scenario as a _core-allocation failure_ .Inthiswork,weusethetargetsuccessrateofcontainerallocation requests as the service level objective (SLO). 

Figure 1 illustrates the architecture of the platform. To eliminate container-allocation failures, the platform maintains a dedicated pool of pre-warmed containers for each supported runtime and con!guration size. Additionally, it maintains a pool of ready VMs to prevent core-allocation failures and enable rapid replenishment of the container pools. The VM pool and each container pool have static sizes. The platform actively maintains these pools: whenever a resource (i.e., a VM or container) is consumed from a pool, it triggers a replenishment to create a new resource and restore the pool to its predetermined size. 

Balancing performance and operational cost is a signi!cant challenge. The size of pre-warmed resource pools signi!cantly impacts the performance and the costs. Oversized pools result in substantial underutilization and unnecessary expenses, whereas undersized pools fail to meet performance expectations, leading to increased latency and potential SLO violations. Typically, these pools are oversized in production environments. For instance, the pre-warmed resource pools in our production platform incur 76% of the overall cost of the platform (Section 7). 

This research investigates the optimization of pre-warmed resource pools (i.e., VM and container pools) to reduce the operational costs, while maintaining platform performance and meeting the target SLO. The contribution of this work is threefold. 

First, we present a comprehensive analysis of containerallocation traces from the production environment of Azure Functions. The container-allocation trace di"ers from the function invocation trace, as not every user function invocation results in a container allocation. We highlight that this work is the !rst to study this kind of trace at a large scale. We study various features of the container-allocation workload, including runtime popularity, container lifecycle, and request arrival patterns, burstiness, and periodicity. We !nd thatruntimesvarysigni!cantlyintheirpopularityandarrival rates, with a few runtimes dominating the workload. In addition, our analysis shows that the workload is highly bursty, lacks periodicity, and exhibits long-tailed creation latencies, making e#cient pool size management more challenging. 

Second, we assess the viability of using traditional forecasting and reactive approaches to manage resource pools. Our !ndings (Section 3) indicate that workload forecasting using state-of-the-art statistical and machine learning models [4–9] is an ine"ective method to manage resource pools because it fails to accurately predict future workload bursts. Also, our evaluation (Section 7) shows that reactive methods [10] cannot handle bursty workload due to their inherent lagging behavior. 

Motivated by our !ndings, we introduce DROPS, a novel data-driven resource optimization method to manage serverless resource pools. The core of DROPS is a statistical analysis of the demand and supply of the workload. DROPS captures workload burstiness relative to resource creation latencies through an e#cient sliding window analysis over the container-allocation trace and the container-lifecycle trace. Based on this analysis, DROPS builds a mapping that accurately determines the minimum pool size for a target SLO. Our evaluationdemonstratesthatDROPSgeneratespoolsizesthat consistently meet the target SLO, while reducing the operational cost by 41% compared to the static approach that was previously deployed in the production environment. DROPS is currently deployed in production in all Azure regions. 

The rest of the paper is organized as follows. Section 2 presents the workload characterization. Section 3 assesses theviabilityofusingworkloadforecastingtomanageresource pools. Section 4 discusses common resource pool management methods. Section 5 introduces DROPS, the proposed resource optimization method. Section 7 presents the evaluation results. Section 8 discusses related work. Finally, the paper concludes in Section 9. 

## **2 Workload Characterization** 

In this section, we !rst de!ne the production datasets we use in this work (Section 2.1). Then, Section 2.2 presents a detailed analysis of the workload characteristics. Finally, Section 2.3 analyzes the workload burstiness. 

1282 

**==> picture [193 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
��<br>��<br>��<br>��<br>��<br>�<br>� ��� ��������<br>���������� �%����������� �%����������� ������������ �%����������� �%����������� ���� ����������� ���������� ������������ ���������<br>**----- End of picture text -----**<br>


**==> picture [218 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
VM Pool Container  Container Pools<br>Creation Python<br>C1 C2 C i .NET .NET<br>C3 .. ..<br>.NET<br>Invocation<br>Container  Container<br>Request<br>Recycling Allocation<br>**----- End of picture text -----**<br>


**Figure 3.** The lifecycle of a container. 

**Figure 2.** Proportion of the top 10 pools. 

## **2.1 Datasets** 

We collected our dataset from data centers in one region of the serverless platform. The collected data includes the containerallocation requests over 14 days between November 1, 2024 and November 14, 2024. The dataset includes two traces: 

- **Container-allocationtrace** :Atimeseriesofcontainerallocation requests. It includes the container ID, request timestamp with microsecond accuracy, the target runtime environment, the runtime version, and the container size (i.e., number of cores and amount of RAM). Our platform, o"ers a set of container sizes. Each size is identi!ed by runtime/core count and comes with a !xed amount of memory. The trace includes more than 2 _._ 2 million requests for 16 runtime environments. 

- **Container-lifecycle trace** : For each container, the trace includes the duration, in microseconds, spent in each stage, namely, the creation, allocation, user workload, and deletion stages. 

We plan to publicly release our dataset[1] . The dataset contains container-allocation requests, which is di"erent from other public datasets [11–14] that represent user function invocations, which do not directly translate to container allocations. 

## **2.2 Statistical Features of the Workload** 

This section presents a comprehensive analysis of the statistical features of the container-allocation workload. We use the following notation to refer to pools in text: runtime/corecount. For example, "Python 3.11/1" refers to a pool for the Python 3.11 runtime with single-core containers. 

**Pool popularity.** Figure 2 illustrates the popularity of different container pools based on the number of containerallocation requests. The workloads of the .NET 8.0/1 and Python 3.11/1 pools dominate the total workload of the platform, accounting for 75% of the total load. Python 3.11/2 and Node.js 20/1, see moderate usage, jointly accounting for around 15% of container allocations. The remaining 12 pools account for about 10% of the total workload. This signi!cant variation in the load across pools highlights that di"erent 

> 1The dataset will be available at: h!ps://github.com/Azure/ AzurePublicDataset. 

pools require di"erent amounts of assigned resources (i.e., di"erent pool sizes). 

**Requests arrival rate.** Figure 4a shows the CDF of perminute container-allocation requests of di"erent pools. The workload volume varies signi!cantly across pools, with the .NET 8.0/1 and Python 3.11/1 pools exhibiting a long tail, with the 99.9[th] percentile (P99.9) reaching 500 requests per minute. In addition, the workloads of various pools exhibit signi!cant burstiness. The Python 3.11/1 pool has the most bursty workload, with a median of 17 requests per minute and a P99.9 of 530, which is 32 times the median. The .NET 8.0/1 pool shows a slightly lower burstiness, with a median of 48 requests per minute and a P99.9 of 460 requests. Notably, even low-tra#c pools experience sudden bursts; the P99.9 of per-minute requests of Python 3.10/2 is 14 times the median. This high level of burstiness complicates resource allocation, as underestimating these bursts can lead to resource underprovisioning, resulting in SLO violations. 

**Container lifecycle.** Figure 3 shows the lifecycle of a container, which consists of !ve main stages: creation, ready, allocation, user workload, and deletion. The cycle begins with the creation stage, during which the platform allocates the required resources from a VM and bootstraps a container with the target runtime environment. Next, the container transitions to the ready state, in which it remains idle in a pre-warmed container pool. When a container-allocation request is triggered, the platform consumes the container from the pool, and the container transitions to the allocation stage. During the allocation stage, the platform injects user-speci!c code, dependencies, and con!gurations into the container. The container then transitions to the user workload stage, in which it processes user-de!ned function invocations. Once all function invocations are completed, the platform moves the container to the deletion state, in which the container is terminated, and its resources are released back to the VM pool for future use. 

We focus on four container lifecycle stages that critically in$uence resource allocation: creation, allocation, user workload, and deletion stages. The container-creation latency impacts the container pool size required to meet the SLO. A longer creation time means the platform requires a longer time to replenish the pool. As a result, a pool must be larger to account for allocation requests that arrive while the pool 

1283 

**==> picture [504 x 286] intentionally omitted <==**

**----- Start of picture text -----**<br>
���<br>���<br>���<br>���<br>���<br>���<br>�� [�] �� [�] �� [�]<br>��������������<br>(a)  Allocations per minute. (b)  Creation stage. (c)  Allocation stage.<br>��� ���<br>��� ���<br>��� ���<br>��� ���<br>��� ���<br>��� ���<br>�� [�] �� [�] �� [�] �� [�] �� [�] �� [�] � �� �� �� �� ��<br>�������������� ������������<br>(d)  Deletion stage. (e)  User workload stage. (f)  The overhead.<br>���<br>��� ���<br>**----- End of picture text -----**<br>


**Figure 4.** CDFs of (a) allocation requests per minute, (b–e) container lifecycle stage latencies, and (f) the overhead of container management for the top !ve container pools. 

is being replenished. Similarly, the container-allocation latency, container-deletion latency, and user-workload latency impact the VM pool size. The deletion time determines when the resources of deleted containers become available to create new containers. The longer the allocation time and deletion time, the larger the VM pool needed to account for allocation requests arriving while resources are still reserved. 

Figure 4b-e shows the CDF of the latency of the creation, allocation, deletion, and user workload stages for di"erent pools. The distributions of the creation (Figure 4b) and deletion (Figure 4d) stages exhibit similar patterns across various pools. The median and P99.9 container-creation latencies are approximately 1.1 seconds and 26 seconds, respectively, while the median and P99.9 container-deletion latencies are around 6secondsand50seconds.Incontrast,thecontainer-allocation latency (Figure 4c) is dominated by loading user code and dependencies. As a result, it varies signi!cantly across pools. For instance, the median latencies of the .NET 8.0/1 and Python 3.11/1 pools are 1.48 seconds and 1.30 seconds, while the P99.9 latencies of the .NET 8.0/1 and Python 3.11/1 pools are 7.83 seconds and 22.49 seconds, respectively. 

Figure 4e shows the CDF of user time across various pools. User time is the time spent running user workload. The .NET 8.0/1 pool has the shortest median user time at approximately 2 minutes, while Python 3.11/1 has the longest median user time of around 6 minutes. The median user times of the remaining pools fall between these two extremes. Despite these 

relativelysmallmedians,allpoolsexhibitlong-tailedusertime distributions, with notable variation in the extent of these tails. Python runtimes exhibit the longest-running workloads, with the P99.9 durations extending to 23 hours. In contrast, Node.js 20/1 and .NET 8.0/1 have more bounded tails, with the P99.9 values around 2 hours. 

To assess the e#ciency of the platform, Figure 4f reports the CDF of the overhead of various pools. The overhead is de!ned as the ratio of time spent in the creation, allocation, and deletion stages to the total lifetime of a container, which includes creation, allocation, deletion, and user workload durations.Approximately90%ofallcontainershaveanoverhead of around 11%, indicating high resource e#ciency. However, a few containers of each pool experience high overhead that can reach almost 100%, which may result from high creation, allocation, or deletion latencies or exceptionally short user workload durations. 

**Periodicity.** Periodicity measures the presence of recurrent, periodic patterns in a time series, where $uctuations in the data align with regular temporal intervals (e.g., hourly, daily, or weekly). For example, serving enterprise users might exhibit daily periodicity, with requests peaking during business hours and declining overnight. 

To capture periodicity, we compute the autocorrelation of the time series. Formally, let _𝐿_ = { _𝐿_ 1 _,𝐿_ 2 _,...,𝐿𝐿_ } represent a time series of _𝑀_ time units, where _𝐿𝑀_ denotes the number of container-allocation requests in time unit _𝑁_ . Autocorrelation 

1284 

**Table 1.** Periodicity and spikiness of the workload of the top !ve container pools. 

|**Workload**|**Periodicity**<br>**(Daily)**|**Periodicity**<br>**(Hourly)**|**Spikiness**|
|---|---|---|---|
|||||
|.NET 8.0/1<br>Python 3.11/1<br>Python 3.11/2<br>Node.js 20.0/1<br>Python 3.10/2|0.01<br>0.01<br>0.05<br>0.09<br>0.05|0.01<br>0.02<br>0.03<br>0.08<br>0.03|5.52<br>6.87<br>5.81<br>7.82<br>5.81|



**==> picture [128 x 84] intentionally omitted <==**

**==> picture [111 x 81] intentionally omitted <==**

**==> picture [130 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) (b)<br>**----- End of picture text -----**<br>


**Figure 5.** Burstiness analysis: (a) The CDF of the load changes of the container allocation trace, and (b) Original trace with bursts highlighted in orange. 

measures how closely current values resemble past values at various time intervals, known as lags (l), as in Equation 1 [15]. 

**==> picture [192 x 27] intentionally omitted <==**

The values of the periodicity metric are bounded between -1 and 1, where 1 indicates perfect positive periodicity, -1 indicates perfect negative periodicity, and 0 indicates no periodicity. Table 1 presents the hourly and daily periodicity for various pools. Most pools exhibit low hourly and daily periodicity, ranging between 0.01 and 0.09. Low periodicity values suggest that it is di#cult to forecast the workload, as we show in Section 3. 

## **2.3 Burstiness Analysis** 

In this section, we perform a burstiness analysis to measure the role of bursts in determining the pool size to meet the SLO. A burst is a temporary and sudden change in the containerallocation demand during a short time period. 

First, we measure the spikiness of various pools. Spikiness [15, 16] is a metric that measures the degree of $uctuation in the workload over time. For example, the platform may receive a few container-allocation requests during periods of low demand, followed by an abrupt spike (e.g., up to a thousand requests) in the subsequent time unit. The spikiness _𝑆_ is de!ned as in Equation 2. 

**==> picture [194 x 33] intentionally omitted <==**

Spikinessiscalculatedasthenormalizedrootmeansquared deviation between consecutive time units ( _𝐿𝑀_ ) and ( _𝐿𝑀_ +1), normalized by the mean ( _𝑅_ ) to ensure comparability across workloads with di"ering baselines. Table 1 presents the spikiness of di"erent pools using 1-second time intervals. As a baseline, the spikiness of a Poisson workload with the same average interarrivalrateasourworkloadhasavalueof1.FromTable1, we can see that the workload of most pools exhibits substantially higher spikiness than the Poisson workload, ranging between 5 _._ 5 and 7 _._ 8. 

To better understand the burstiness of the workload, we analyze $uctuations in the load over time. Speci!cally, we 

measure the positive changes in the load over a time window of 1 second. We slide two back-to-back windows on the container-allocation trace. At each time step, we measure the change in the load between the !rst window and the second window. The change in the load is computed as the di"erence between the number of requests in the !rst window and the number of requests in the second window. Figure 5a shows the CDF of the positive changes in the workload of the .NET 8.0/1 pool. We select the top 1% of the measured changes as bursts. The value of P99 of the positive change is 38 times larger than the median value of the positive change. 

**Role of bursts in meeting the SLO.** To assess the role of bursts in meeting the target SLO, we measure the volume of the load in the bursts to the total volume of the workload. Figure 5b shows a sample 12-hour trace of the containerallocation trace of the .NET 8.0/1 pool with bursts highlighted in orange. The volume of the load in the top 1% of bursts is 8% of the total load. As a result, in order to meet any SLO higher than 92%, bursts above the P99 must be considered and carefully analyzed. Hence, bursts play a critical role in determining the pool size needed to meet the target SLO. **Predictability of bursts.** We study if it is possible to predict the time and volume of future bursts using forecasting models (detailed in Section 3). We measure the periodicity of bursts for di"erent pools using Equation 1. The periodicity of the bursts of di"erent pools ranges between 0 _._ 03 and 0 _._ 06, indicating that bursts exhibit no periodicity, making them extremely hard to predict. We further measure the error of predicting bursts, as discussed in Section 3. Results show that the average error and the maximum error of predicting bursts are 107% and 1170%, indicating that forecasting models are poor at predicting bursts for any pool. 

**Workloadsummary.** Ouranalysisrevealsthattherealcontainerallocation workload is highly bursty, lacks clear periodic patterns, and involves long-tailed resource creation latencies. These characteristics complicate pool sizing decisions: bursty demand leads to sudden spikes in resource needs, the absence of periodicity hinders prediction-based planning, and long creation times delay resource availability during spikes. Our 

1285 

evaluation(Section7)showsthatreactiveandpredictivemethods struggle to achieve high success rates without incurring excessive overprovisioning. 

## **3 Forecasting Future Load** 

Forecasting methods can be used to predict future workload to proactively adjust the size of pre-warmed resource pools. To assess the viability of this idea, we employ state-of-the-art timeseriesforecastingmethods,includingstatistical,machine learning, and foundation models: 

- **Statistical models.** We evaluate four statistical models: ARIMA [8], Theta [7], ETS [9], and a Naive model that repeats the value of the previous day. 

- **Machine learning models.** We evaluate three machinelearning(ML)models:PatchTST[6],atransformerbasedmodeloptimizedfortimeseries;TemporalFusion Transformer (TFT) [5], which combines LSTM with a transformer layer, and DeepAR [4], a probabilistic autoregressive network based on LSTM architecture. 

- **Other models.** We evaluate Chronos [17], a foundation model pre-trained on massive datasets, enabling zero-shot forecasting without task-speci!c !ne-tuning. Additionally, we evaluate a Weighted Ensemble model, !tted using other top-performing predictors. 

Totrainvariousforecastingmodels,weuseAutoGluon[18], 

which is an open-source AutoML framework that provides a uni!ed interface for di"erent models and handles parameter tuning. The evaluation is conducted using a rolling forecasting approach over the trace. The training period is 7 days followed by a testing period of one day. The rolling window is advanced by one day at each step, ensuring that the entire test week is covered and that performance metrics re$ect models’ generalization capability across di"erent temporal contexts. 

We evaluate the forecasting models using the containerallocation trace of the .NET 8.0/1 pool, as it is the most popular pool (Figure 2). However, our !ndings generalize to other resource pools (Section 7). For training, we use a machine equipped with two Intel E5-2630 CPUs (each CPU has 16 cores or 32 threads running at 2.40 GHz), 128 GB RAM, and 480 GB SSD. 

## **3.1 Container-allocation Workload Predictability** 

We evaluate the forecasting models with four prediction intervals:10seconds,1minute,10minutes,and1hour.Aprediction interval refers to the temporal resolution at which future values are forecasted. Formally, a prediction interval of _𝑇_ seconds implies that the model generates one predicted data point for every _𝑇_ -second interval. We use di"erent prediction intervals to evaluate the capability of a model to capture patterns at di"erent time resolutions. For training, we provide a time series where each data point corresponds to the total number of container allocations within an interval. The model generates 

**Table 2.** Forecasting models performance evaluation with di"erent prediction intervals. 

||**Model**<br>**Interval**<br>**Avg Error**<br>**Max Error**<br>**Bias**<br>**Training Time (s)**|
|---|---|
||**Statistical**|
||Naive<br>10 seconds<br>182%<br>2300%<br>1_._3<br>6_._0<br>1 minute<br>46%<br>2900%<br>4_._0<br>1_._7<br>10 minutes<br>25%<br>842%<br>40_._8<br>1_._4<br>1 hour<br>18%<br>124%<br>245_._0<br>1_._4<br>ETS<br>10 seconds<br>123%<br>1246%<br>↑1_._6<br>6_._3<br>1 minute<br>53%<br>4202%<br>↑16_._9<br>1_._9<br>10 minutes<br>72%<br>638%<br>253_._2<br>3_._4<br>1 hour<br>18%<br>82%<br>19_._2<br>0_._4<br>Theta<br>10 seconds<br>173%<br>3630%<br>1_._3<br>67_._3<br>1 minute<br>66%<br>5404%<br>↑1_._9<br>22_._9<br>10 minutes<br>20%<br>758%<br>↑81_._4<br>21_._9<br>1 hour<br>17%<br>87%<br>↑307_._5<br>7_._1<br>ARIMA<br>10 seconds<br>130%<br>1268%<br>↑0_._6<br>8_._4<br>1 minute<br>34%<br>3195%<br>↑3_._3<br>2_._4<br>10 minutes<br>20%<br>763%<br>↑18_._4<br>21_._1<br>**1 hour**<br>**14%**<br>**76%**<br>**54.5**<br>**1.1**|
||**Machine Learning**|
||DeepAR<br>**10 seconds**<br>**84%**<br>**1628%**<br>**-3.26**<br>**3574.1**<br>1 minute<br>33%<br>2569%<br>↑6_._9<br>743_._1<br>10 minutes<br>40%<br>579%<br>↑27_._0<br>173_._0<br>1 hour<br>28%<br>124%<br>↑120_._4<br>74_._9<br>TFT<br>10 seconds<br>92%<br>889%<br>↑3_._8<br>3597_._9<br>1 minute<br>38%<br>3082%<br>0_._4<br>893_._4<br>10 minutes<br>27%<br>901%<br>↑22_._2<br>347_._1<br>1 hour<br>31%<br>358%<br>209_._5<br>159_._1<br>PatchTST<br>10 seconds<br>96%<br>1562%<br>↑1_._9<br>1413_._1<br>1 minute<br>32%<br>2929%<br>↑3_._4<br>194_._7<br>10 minutes<br>21%<br>714%<br>↑7_._7<br>35_._1<br>1 hour<br>45%<br>167%<br>343_._7<br>31_._3|
||**Other Models**|
||Chronos<br>10 seconds<br>102%<br>2946%<br>↑3_._7<br>7_._6<br>1 minute<br>31%<br>2326%<br>↑13_._5<br>1_._3<br>10 minutes<br>19%<br>678%<br>↑39_._8<br>1_._0<br>1 hour<br>15%<br>74%<br>77_._2<br>1_._4<br>Weighted<br>Ensemble<br>10 seconds<br>99%<br>2946%<br>↑2_._7<br>30_._1<br>**1 minute**<br>**30%**<br>**2522%**<br>**-6.5**<br>**19.0**<br>**10 minutes**<br>**19%**<br>**731%**<br>**-19.6**<br>**20.3**<br>1 hour<br>17%<br>131%<br>15_._7<br>3_._0|



a time series where each point represents the predicted total load within an interval. 

Table 2 reports the average and maximum prediction errors of various models for di"erent prediction intervals. We use the mean absolute percentage error (MAPE [19]) as an error metric. MAPE provides an intuitive, scale-independent measure of forecast accuracy expressed as a percentage, making it easy to interpret. MAPE penalizes overprediction and underprediction of the same magnitude equally. MAPE is computed as shown in Equation 3, where _𝑈𝑀_ is the actual value, _𝑉𝑀_ is the forecast value, and _𝑀_ is the time series length. 

**==> picture [162 x 22] intentionally omitted <==**

1286 

**==> picture [157 x 105] intentionally omitted <==**

**Figure 6.** One-hour training trace with di"erent prediction intervals. 

Table 2 shows that the lowest average errors for the 10second, 1-minute, 10-minute, and 1-hour intervals are 84%, 30%, 19%, and 14%, respectively. Table 2 indicates that forecasting models are incapable of accurately predicting !negrained long-term container-allocation workload because the container-allocation workload exhibits unpredictable, highintensity bursts. These bursts deviate sharply from regular temporal patterns, making them di#cult to capture with forecasting models. 

Table 2 shows that using a smaller prediction interval signi!cantly increases the prediction error for all forecasting models because shorter intervals capture !ner-grained $uctuations, making the training data more bursty and harder to model. Furthermore, the length of the prediction interval has a signi!cant impact on the shape of the trace, which directly impacts the forecasting accuracy. Figure 6 presents a normalized 1-hour training trace using 10-second, 1-minute, and 10-minute prediction intervals. As the prediction interval increases, the workload becomes signi!cantly smoother, suppressing abrupt and sharp bursts. The coe#cients of variation of the same trace with 10-second, 1-minute, and 10-minute intervals are 1 _._ 88, 0 _._ 32, and 0 _._ 05, respectively, illustrating the smoothing e"ect. This smoothing e"ect simpli!es the learning and prediction of the workload. 

**Bias.** Table 2 reports the bias, which is the average di"erence between predicted values and actual values. A positive bias indicates that the model tends to overpredict, while a negative bias indicates that the model tends to underpredict. In the context of resource allocation, overprediction results in larger pools than needed, increasing operational costs. In contrast, underprediction leads to undersized pools, which can result in allocation failures and SLO violations. 

**Training cost.** Table 2 reports the training time required to generate predictions for 1 day across di"erent models. The resultsindicatethatshorterpredictionintervalsrequiresubstantially longer training times than larger intervals because they require !tting the models on larger training datasets. For example,thesizeofthetrainingdatafora10-secondintervalis60 times larger than that for a 10-minute interval. In addition, results show that ML-based models consistently exhibit longer training times than statistical models due to their increased complexity and reliance on iterative optimization algorithms. 

**Summary.** Our evaluation of forecasting models shows that they are not an e"ective solution for managing resource pools, as they fail to predict the workload with high accuracy. In Section 7, we further evaluate an oracle perfect predictor that provides exact future workload values. Even with perfect predictions, our results reveal a fundamental limitation of forecasting: it cannot predict bursts’ temporal proprieties, leading to SLO violations. As a result, tuning current models or using more accurate models will not make forecasting a viable approach for resource pool management. 

## **4 Resource Pool Management Methods** 

E"ective pool size management is critical in serverless platforms as it directly a"ects performance and cost. Oversized pools lead to idle resources and higher costs, while undersized pools result in allocation failures, violating the SLO. 

The resource bu"ering problem is not unique to serverless platforms. For example, operating systems and networked environments typically maintain memory bu"ers to improve performance. This section discusses three main approaches that are widely used to determine a bu"er size: static, reactive, and prediction-based approaches. We evaluate the performance of these approaches in Section 7. 

**Static approach.** In this approach, the system pre-allocates a !xed bu"er size in advance, which remains !xed regardless of the system load. Azure Functions maintains a !xed number of pre-warmed containers for each pool to avoid allocation failures. A high watermark and a low watermark can be used tolazilyreplenishthebu"er;insteadofreplenishingthebu"er immediately after an item is consumed, the bu"er is replenished only if it drops below the low watermark. The main advantage of this approach is its simplicity, as it eliminates the overhead associated with dynamic resizing and runtime monitoring. However, in large-scale serverless platforms, this approach requires manual management of many pools, which is cumbersome and prone to miscon!guration. **Reactive approach.** A reactive approach can be used to dynamically manage the bu"er size based on its usage. The bu"er has a target size, and whenever a resource is consumed from the bu"er, the system starts replenishing the bu"er to its target size. Low and high watermarks are used to trigger changes to the target bu"er size. When many resources are consumed and the low watermark is reached, the target bu"er size is increased. Similarly, if many resources are idle and the high watermark is crossed, the bu"er size is decreased. The reactive approach adapts to the workload, which can yield better cost e#ciency. However, a reactive approach uses a lagging indicator, which does not !t bursty workloads: after a usagespikeisobserved,thetargetbu"ersizeisincreased.This canleadtopotentialallocationfailuresinserverlessplatforms. **Predictive approach.** To overcome the lagging nature of the reactive approach, load forecasting can be used to predict the upcoming workload based on historical data and set the bu"er 

1287 

size in advance. Similar to other approaches, the bu"er is replenished eagerly. Forecasting models have limited e"ectiveness for bursty workloads, as described in Section 3. Another challenge with this approach lies in translating the predicted load into a pool size. We discuss this challenge in Section 7. 

## **5 DROPS** 

This section introduces DROPS, a new statistical, data-driven resource optimization method to manage resource pools in in!nite-capacity systems, such as serverless platforms. In such systems, there are no constraints on the system’s ability to create new resources (i.e., there is no upper bound on the number of resources, such as VMs and containers, that can be created in parallel). 

DROPS utilizes historical traces to generate an accurate pool-size-to-success-rate mapping that can be used to determine the minimum pool size needed to meet a target SLO. DROPS uses an e#cient sliding window analysis to capture burstiness relative to resource creation latencies. In this work, we use the target success rate of allocation requests as an SLO. That is, the SLO speci!es the minimum percentage of allocation requests that must be ful!lled using pre-warmed containers. Section 5.1 presents the proposed method. Section 5.2 and Section 5.3 discuss how we apply the proposed method to optimize the container and VM pools. 

## **5.1 Resource Optimization Algorithm** 

DROPS optimizes resource pools in an in!nite-capacity system that maintains a set of pools. Each pool has a target size, which the system strives to maintain at all times. Three main actions can occur to a resource pool: resource consumption, resource recycling, and resource replenishment. A _resource consumption_ means that the system consumed an item from the pool. The system can keep using the consumed item inde!nitely, or it can release the item after some time. Released resource items may be added back to the pool (i.e., _resource recycling_ ) and can be consumed again. When a resource item is consumed from the pool, the system triggers a _resource replenishment_ to create a new resource item to re!ll the pool. The system can create an unlimited number of resource items in parallel. However, resource creation is not instantaneous and incurs some delay. 

DROPS is based on a statistical demand-supply analysis. The demand represents the consumption of items from the pool,whilethesupplyrepresentsthetimerequiredtocreateresourceitems.Thatis,thedurationfromthetimepointatwhich a request to create a resource item is issued to the time point at which the resource item becomes ready and part of the pool. 

DROPS utilizes the following property of in!nite-capacity systems: the latency to ful!ll a resource consumption request is bounded by the maximum creation latency of a resource. When a consumption request is received, it is either ful!lled immediately if the pool is not empty or must wait until a 

## **Algorithm 1** DROPS 

|1:|**Input:**Demand trace:_𝑄_, Recycling trace:_𝑅_, Resource creation latency|**Input:**Demand trace:_𝑄_, Recycling trace:_𝑅_, Resource creation latency|
|---|---|---|
||distribution:_𝑆_, Number of samples:_𝑇_||
|2:|**Output:**Pool-size-to-success-rate mapping:_𝑈_||
|3:|Initialize an empty list_𝑉_||
|4:|**for**_𝑊_=1 to_𝑇_**do**||
|5:|Set_𝑋𝑌_to P100(_𝑆_)<br>_𝐿_Window size = max creation latency||
|6:|**for**each consumption request_𝑍_in_𝑄_**do**||
|7:|_𝑈𝑌_=0|_𝐿_Start with a pool size of zero|
|8:|Set_𝑎_↓[_𝑀𝑁_↑_𝑋𝑌,𝑀𝑁_]<br>_𝐿_Set window_𝑎_using timestamp of_𝑍_||
|9:|**for**each request_𝑏_in_𝑎_**do**|_𝐿_Analyze requests in_𝑎_|
|10:|Rep_𝑂𝑃𝑀_↓sample from_𝑆_|_𝐿_Latency drawn from_𝑆_|
|11:|**if**_𝑀𝑄_+Rep_𝑂𝑃𝑀>𝑀𝑁_**then**|_𝐿_Is replenishment time_>𝑀𝑁_|
|12:|_𝑈𝑌_=_𝑈𝑌_+1|_𝐿_Increment the needed pool size|
|13:|**end if**||
|14:|**end for**||
|15:|_𝑐_↓count of recycling events|with timestamp↔_𝑎_|
|16:|_𝑈𝑌_=_𝑈𝑌_↑_𝑐_<br>_𝐿_Resource recycling reduces the needed pool size||
|17:|_𝑉_↓_𝑉_↗{_𝑈𝑌_}|_𝐿_Append pool size_𝑈𝑌_to_𝑉_|
|18:|**end for**||
|19:|**end for**||
|20:|Compute_𝑈_as the empirical CDF of_𝑉_||
|21:|**return**_𝑈_||



resource item becomes ready. In the worst case, the consumptionrequestwaitsforthemaximumcreationlatency.Thepool size needed to ful!ll a resource consumption request at time _𝑁𝑍_ depends only on the volume of the consumption and recycling within the interval [ _𝑁𝑍_ ↑ _𝑁𝑑,𝑁𝑍_ ], where _𝑁𝑑_ is the maximum resource creation latency. In other words, it depends only on the consumption requests and recycling events occurring within the creation latency window preceding the request _𝑊_ . 

Algorithm 1 shows the pseudo-code of DROPS, which requiresthefollowinginputs:1)ademandtrace,asequenceofresource consumption requests, 2) a recycling trace, a sequence of resource recycling events, and 3) a supply-latency distribution, a distribution of resource creation latency. DROPS outputs a mapping that can precisely set the pool size to meet any target SLO. 

DROPS uses a sliding window analysis to construct a statistical distribution of the consumption demand over time. To determine the minimum pool size required to ful!ll a request _𝑊_ from the pool, DROPS examines all prior consumption requests within the interval [ _𝑁𝑍_ ↑ _𝑁𝑑,𝑁𝑍_ ] (Algorithm 1, Line 9–14). Recall that every consumption request triggers a replenishmenttore!llthepool.Hence,foreachrequestwithintheinterval,DROPSsamplesareplenishmentlatencyfromthecreation latency distribution (Line 10). If the sampled replenishment completes before _𝑁𝑍_ , then its demand does not a"ect the pool size required at _𝑁𝑍_ . In contrast, if the replenishment arrives after _𝑁𝑍_ , the request still a"ects the pool at time _𝑁𝑍_ , and the pool size is incremented accordingly (Lines 11–12). To determine the !nal pool size required to ful!ll request _𝑊_ , we need to account for recycling events. DROPS counts the number of recyclingeventsthatoccurwithintheinterval,andusesittoadjust the pool size (Line 15–16). A resource recycling event reduces 

1288 

the required pool size, as a recycled resource is added back to thepoolandcanbeusedtoful!llfutureconsumptionrequests. The adjusted pool size is added to the distribution _𝑋_ (Line 17). 

For each consumption request, the distribution _𝑋_ contains one value representing the pool size required to ful!ll that request. The distribution _𝑋_ is transformed to a pool-size-tosuccess-rate mapping by computing the empirical cumulative distribution function (CDF) of _L_ (Line 20), which DROPS generates as an output. This mapping can be used to set the pool size to meet any SLO. For instance, if the target SLO is 99%, setting the pool size to the 99 _[𝑀𝑒]_ percentile of the mapping guarantees meeting that SLO as long as the future workload has the same properties as the input workload. 

DROPS can handle both stationary and bursty workloads. DROPS captures workload burstiness while accounting for the creation latency of resources. DROPS enables accurate sizing for any target SLO, a capability that other methods lack. The accuracy of DROPS depends on the length of the input traces, which production systems typically collect and store for long durations. DROPS is e#cient and does not involve anycomplextrainingalgorithms.Furthermore,asystemmust re-run DROPS only if new pools are introduced or if there are changes in the workload that lead to a higher failure rate or resource underutilization than expected. 

## **5.2 Container Pools Optimization** 

The serverless platform maintains multiple container pools. Each pool has its unique workload and container-creation latency distribution (Figure 4). Hence, we use Algorithm 1 to independently optimize each container pool. For each pool, we use the container-allocation trace as the demand trace and the container-creation latency distribution as the supplylatency distribution. In our platform, containers cannot be reused; when a container is deleted, its resources are released to the VM hosting the container. Hence, we set the recycling trace to an empty trace. The output pool-size-to-success-rate is then used to set the size of the container pool. 

## **5.3 VM Pool Optimization** 

In this section, we discuss how DROPS is applied to optimize the VM pool. We use the empirical distribution of the VM-creation latency. We use the input container-allocation trace and the container-lifecycle distributions to generate the core-allocation and the core-recycling traces. 

**Core-allocation trace.** Recall that the platform maintains a target number of idle VMs in the VM pool. As a result, whenever a VM is consumed, the platform creates a new VM to re!ll the pool. To generate a core-allocation trace, we map each container-allocation request in the container-allocation trace to a core-allocation event, with the number of allocated cores determined based on the target container size (e.g., 1 core or 2 cores). The generated core-allocation trace re$ects the VM demand of all container pools. 

The above assumes that the core-allocation event occurs at the same time as the container-allocation request, which is inaccurate. The core-allocation event must happen earlier than the container-allocation request. Speci!cally, the core-allocation event must precede the container-allocation request by at least one unit of container-creation latency. So, we adjust the core-allocation trace by shifting each coreallocation event backward in time by a duration sampled from the container-creation latency distribution. 

**Core-recycling trace.** In our serverless platform, a container isdeletedonceitbecomesidle.Resourcesofdeletedcontainers are released back to the VM pool (i.e., core recycling). Recycled cores signi!cantly impact the VM pool size as these cores become available to create new containers. Hence, neglecting thecore-recyclingeventsresultsinresourceoverprovisioning and higher costs. 

The core-recycling events depend on the user workload characteristics, which vary from one tenant to another and from time to time. To account for that, we generate a corerecycling trace, a sequence of core-recycling events. For each container-allocation request, we determine the time at which thecontainerwillbedeletedbysamplingthedurationofevery stage in the container lifecycle from the container-lifecycle distributions. 

Now, we have all the inputs required to apply Algorithm 1 tooptimizetheVMpool.Weusethegeneratedcore-allocation trace as the demand trace, the generated core-recycling trace as the resource-recycling trace, and the VM-creation latency distribution as the supply distribution. The generated poolsize-to-success-rate mapping can be used to set the size of the VM pool to meet any target SLO. 

## **6 Implementation and Deployment** 

We implement DROPS in C#, and it is currently deployed across all regions and data centers of Azure Functions. The platform runs DROPS every four hours using the previous week’s traces as input to dynamically resize resource pools. In addition to periodic runs, DROPS is invoked on-demand by monitoring container allocation failures and abnormal resource utilization. For instance, if the failure rate surpasses the target SLO, the platform runs DROPS to reanalyze the workload and adjust pool sizes. DROPS has negligible overhead;itrequiresaboutoneminutetoanalyzeatwo-weektrace containing around two million allocation requests. Deploying DROPS in production resulted in substantial cost savings that vary across geographic regions, ranging between 80%↑90%. 

## **7 Evaluation** 

This section evaluates the performance of various pool sizing methods. Section 7.1 compares the failure rate, cost, and latency of various methods. Section 7.2 evaluates the performance of the forecasting-based methods with di"erent 

1289 

prediction intervals. Section 7.3 presents a sensitivity analysis of the reactive method. Section 7.4 presents a sensitivity analysis of DROPS. Section 7.5 introduces the aggressive container creation optimization. Section 7.6 evaluates DROPS using traces from di"erent geographical regions. **Metrics.** We use the following metrics in the evaluation: 

   - **Failure rate.** The failure rate is the ratio of failed requests to total requests. A failed request is a request that is not satis!ed immediately from the pre-warmed pool of containers. Note that no allocation request is rejected; every request is eventually ful!lled. 

   - **Cost.** We measure the total cost in core-hours that is required to serve the input container-allocation trace. We also provide a breakdown of the cost in various components, including the container pool, VM pool, platform overhead, and user workload cost. The container pool and VM pool costs represent the cost of maintaining idle containers and VMs, respectively. 

- **Latency.** Thelatencyofful!llingacontainer-allocation request. This includes the time from when the request is received until a container is allocated to the request. 

- **Alternatives.** In addition to DROPS, we consider the following alternatives in our evaluation: 

   - **Static method.** The size of each container pool is static and set by an expert operator. For the VM pool, the platform employs a reactive scaling strategy based on the availability of idle CPU cores, which can be used to replenish container pools. When the number of idle cores drops below a low watermark, the platform increases the VM pool size, while if the number of idle cores exceeds a high watermark, the platform reduces the VM pool size. This alternative represents the approach used in production in the Azure Functions platform prior to DROPS. The low and high watermarks are set to 10% and 40% of the platform’s total number of CPU cores. 

   - **Predictive method.** For each container pool, the bestperforming forecasting model is used to predict the future load. The predicted load is used to set the pool size at the beginning of each prediction interval. We evaluate three approaches to map the predicted load to a pool size: constant, Poisson, and concentrated load mapping approaches. The constant load approach assumes that the load is stationary (i.e., no burstiness). It measures the average arrival rate ( _𝑌_ = _𝑃𝑍𝑎𝑏_ / _𝑐𝑀𝑁𝑑𝑊𝑒𝑎𝑃_ ), and the pool size is set to meet this rate. In contrast, the concentrated load approach models an extreme busrtiness by assuming that the entire predicted load arrives at a single time point. The Poisson load approach lies between these two extremes; it assumes that request arrivals follow a Poisson distribution and determines the pool size using the cumulative distribution function (CDF) of the Poisson distribution [20]. 

   - **Reactivemethod.** Areactivepoolmanagementmethod is employed, using a multiplicative increase, additive 

**==> picture [241 x 135] intentionally omitted <==**

**==> picture [205 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Failure rate and cost (b)  Ful!llment latency<br>**----- End of picture text -----**<br>


**Figure 7.** Performance of optimization methods. (a) Failure rate and cost. (b) Ful!llment latency (DROPS and the static alternative have a latency of zero). 

- decrease(MIAD[10])controlalgorithm.Thepoolsizeis exponentiallyscaledupuponeachcontainer-allocation failure, and linearly decreased over time when no failures are observed. This method enables fast adaptation tosuddenloadbursts,avoidsresourceoverprovisioning duringidleperiods,andachievesstabilitybypreventing oscillations in pool size. 

- **Predictive-reactive method.** This alternative combines forecasting with the reactive method. At the beginning of each interval, the method predicts the future load in the next window and adjusts the pool size to meet the predicted load, assuming that all load arrives at a single point in time (i.e., concentrated load). Within an interval, the reactive method adjusts the pool size in response to allocation failures. 

**Workload.** We use a 2-week trace from the production platform in our experiments. We use the !rst week for training and the second week for evaluation. The evaluation follows a rolling approach, where to optimize the pool sizes in a day, the preceding seven days are used as input. We conduct our experiments using a custom-built simulator that replicates the behavior of our serverless platform. We use simulation in our evaluation as it is infeasible to compare alternative approaches in the production environment. The simulator models all stages of the container and VM lifecycles, with delays at each stage sampled from real empirical distributions. The simulator supports di"erent pool optimization methods, replays the container-allocation trace, and collects various metrics, including failure rate, ful!llment latency, core utilization, pool utilization, and cost. We have validated that the simulator accurately emulates the production platform and faithfully replays input traces. The simulator is implemented inC#.ThesourcecodeofthesimulatorandDROPSisavailable at https://github.com/UWASL/DROPS. 

1290 

**==> picture [241 x 105] intentionally omitted <==**

**==> picture [158 x 11] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Failure rate (b)  Cost<br>**----- End of picture text -----**<br>


**Figure 8.** The Failure rate and the cost of di"erent prediction intervals using a perfect predictor. 

## **7.1 Performance Comparison** 

Figure 7a shows the failure rate and the total cost of di"erent optimization methods when the target SLO is 100%. In Figure 7a, prediction-based methods use a 1-hour prediction interval as it has the best performance. Results show that DROPS and the static alternative are the only methods that achieve a 100% success rate, meeting the target SLO. However, DROPS reduces the total cost by 41% compared to the static alternative.Thebreakdownofthecostrevealsakeydi"erence between DROPS and the static approach used in production in the management of idle resources. DROPS reduces the cost of idle containers by 79% and increases the cost of idle cores by 13% (Figure 7a). DROPS is designed to set the container pools to the smallest size capable of meeting the SLO, while keeping the remaining idle cores unassigned to any container pool. This approach enables on-demand core-to-container allocation,leadingtomoree#cientandaccurateallocationdecisions. In contrast, the static approach maintains excessively oversized container pools, which require early allocation of cores to container pools. This may lead to suboptimal and ine#cient core-to-pool allocation decisions. 

To further analyze the impact of this behavior, Figure 9 shows the CDF of the normalized fullness (i.e., the ratio of the number of resource elements in the pool to the size of the pool) of the VM and .NET 8.0/1 pools of DROPS and the static alternative. A 100% fullness means the pool is full, while a 0% fullness means the pool is empty. For the static alternative, the minimum fullness of the VM pool reaches 0% while the minimum fullness of the .NET pool is 55%. This indicates that the VM pool is fully consumed while many containers are idle, risking the occurrence of allocation failures despite the platform being overprovisioned. In contrast, in DROPS, the minimum fullness of both the .NET and VM pools is 0% while not leading to any allocation failures, indicating more e"ective core-to-container allocation decisions. Although the fullness of the .NET and VM pools of DROPS remains high for most of the time, the fact that it reaches 0% indicates that the used pool sizes are necessary to meet the SLO. 

Figure 7a shows that the predictive method with a concentrated load mapping fails to meet the target SLO and incurs 

**==> picture [120 x 85] intentionally omitted <==**

**==> picture [119 x 85] intentionally omitted <==**

**Figure 9.** CDF of the fullness **Figure 10.** Pool size to SLO of the .NET and VM pools. for the .NET and VM pools. 

3 _._ 3↘ higher cost than DROPS. This method assumes that the load arrives in a single burst, leading to oversized pools and incurring higher costs. It also fails to meet the target SLO for two reasons. First, prediction errors can lead to inaccurate pool sizes. Second, this method does not account for the time needed to create new containers and VMs. Hence, requests that arrive while resources are being created may fail. Accounting for the creation latency is challenging as it exhibits long-tail distributions (Figure 4), and accounting only for the worst-case creation latency increases the cost signi!cantly. 

The reactive method leads to the highest failure rate due to its inherent lagging nature; it scales the pool size only after observing a burst. However, because bursts in containerallocation workload are typically short and the creation latenciesofcontainersandVMsarerelativelylonger,thenewlycreated resources become available only after most of the burst’s requests have already failed. The multiplicative increase in pool size under the reactive method leads to a substantial cost escalation, exceeding DROPS’s cost by more than 4 _._ 2 times. Using prediction to augment the reactive method reduces the failure rate as the predicted load is used to adjust the pool size at the start of each interval. 

**Ful!llment latency.** Figure 7b shows the ful!llment latency of container-allocation requests of various methods. DROPS and the static alternative have a latency of zero for all requests. Latencies greater than zero correspond to failed allocation requests. A request can fail due to either a container-allocation failure, which incurs a latency of hundreds of milliseconds to a few seconds, or a core-allocation failure, which incurs a latency in the range of minutes. The predictive, reactive, and predictive-reactive methods exhibit a sharp rise in latency at the tail of the distribution. This sharp rise is attributed to core-allocation failures. 

## **7.2 Performance of the Predictive Method with Di"erent Prediction Intervals** 

Forecasting-based methods are ine"ective for pool sizing for two reasons. First, they are incapable of accurately predicting the load, particularly during burst periods (Table 2). Second, they generate a single value for each prediction interval. Mapping this single value to a pool size requires assumptions about the workload distribution (e.g., Poisson), which may fail to capture the burstiness in the load. To better illustrate this, we assume the existence of a hypothetical perfect predictor: a perfect forecasting model that can predict the true 

1291 

**==> picture [241 x 106] intentionally omitted <==**

**Figure 11.** The failure rate and cost of the reactive method when varying the scale-up and scale-down factors. 

average load in each interval. This removes failures attributed to prediction errors. 

Figure 8 presents the failure rates and costs of the predictive method under di"erent prediction intervals, using three load-to-pool-sizemappingapproaches:constant,Poisson,and concentrated-load. Figure 8 shows that the predictive method fails to achieve a 100% success rate, even when a perfect predictor is used. The concentrated-load approach has the lowest failure rate and the highest cost due to pool oversizing. The constant-based approach leads to the highest failure rate as it does not account for any burstiness in the load. The Poisson approach achieves a lower failure rate compared to the constant approach while slightly increasing the cost, as it captures some level of burstiness. 

Figure 8 shows that larger prediction intervals lead to a slightly higher failure rate for the constant and Poisson approaches. For the Poisson approach, using a prediction interval of 1 minute instead of 10 seconds increases the failure rate from 13 _._ 5% to 18%. Notably, the failure rate plateaus for larger intervals; both 10-minute and 1-hour intervals yield similar failure rates. This occurs because both apply a similar degree of smoothing, resulting in comparable pool size estimates. 

## **7.3 Sensitivity Analysis of the Reactive Method** 

The performance of the MIAD reactive method depends primarily on two parameters: the scale-up factor, which controls how quickly the pool size increases after allocation failures, and the scale-down factor, which controls how gradually it decreases during failure-free periods. Figure 11 shows how varying these factors a"ects the performance of the reactive method. As shown in Figure 11, increasing the scale-up factor reduces the failure rate but signi!cantly increases the total cost. For example, a scale-up factor of 2 _._ 5 lowers the failure rate by 1 _._ 4% while increasing the cost by 57%. Further increases in the scale-up factor are unlikely to bring much improvement to the failure rate as the creation latencies of containers and VMs become the limiting factor. 

Figure 11 shows that increasing the scale-down factor reduces the total cost but leads to a higher failure rate. The scale-down factor controls the rate at which the container pool size is reduced (i.e., the number of deleted containers per second). Increasing the scale-down factor from 0.1 to 4.0 

**==> picture [240 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Failure rate and cost (b)  Ful!llment latency<br>**----- End of picture text -----**<br>


**Figure 12.** Performance of DROPS with di"erent SLOs. 

raises the failure rate by 10% while reducing the total cost by 34%. However, aggressive scale-down (e.g., a factor of 4) introduces platform instability and increases overhead, as containers and VMs are deleted before they are e"ectively utilized. For instance, with a factor of 4, the overhead contribution to total cost is 31%, which is 3 _._ 1↘ and 1 _._ 7↘ larger than the cost of user workload and idle containers, respectively. 

## **7.4 DROPS Sensitivity Analysis** 

**Sensitivity to di"erent SLOs.** DROPS provides a key advantage over other resource management methods by enabling precise pool sizing for a given SLO. DROPS constructs a statistical mapping that translates each pool size to its corresponding success rate. Figure 12a shows the failure rate and the cost of DROPS for di"erent SLOs. The results show that DROPS successfully maintains the failure rate below the target SLO. Notably, achieving a 95% success rate incurs a cost that is 30% lower than that required for a 99 _._ 9% success rate. This demonstrates that satisfying the highest percentiles of load comes at a signi!cantly higher cost compared to satisfying lower percentiles. 

To further validate this, Figure 10 shows the pool-size-tosuccess-rate mapping for the VM and .NET 8.0/1 pools. The VM pool and .NET 8.0/1 pool sizes required to achieve a 99 _._ 9% success rate are 2 _._ 2↘ and 2 _._ 5↘ larger than the size required to achieve a 95% success rate. That is, achieving a 99 _._ 9% success rate requires doubling the amount of idle resources needed to achieve a 95% success rate. 

Figure 12b shows the latency CDF for di"erent SLOs. The results show that the latency CDFs of various SLOs exhibit a long tail that reaches hundreds of seconds, which is attributed to core-allocation failures. 

**Sensitivity to keep-alive intervals.** A keep-alive interval determineshowlongacontainerremainsactiveaftercompleting a function invocation and before being deleted. Di"erent serverless platforms adopt di"erent keep-alive intervals. For instance, AWS Lambda [2] and Google Cloud Run [3] keep containers active for a short period (typically several minutes) inordertomitigatecoldstarts.Incontrast,inAzureFunctions, containers are immediately deleted once they become idle, and their resources are recycled back to the VM pool. 

1292 

**==> picture [193 x 146] intentionally omitted <==**

**Figure 13.** The cost of DROPS under di"erent kep-alive intervals. The percentages annotated above each bar denote the fraction of container-allocation requests that are ful!lled using kept-alive containers. 

Akeep-aliveintervaldirectlya"ectsthecontainer-allocation trace. A longer keep-alive interval reduces the number of container-allocation requests, as future function invocations can reuse containers that are kept alive rather than triggering new allocation requests. 

DROPS is inherently robust to di"erent keep-alive intervals because it relies on real traces of allocation and recycling events. Hence, DROPS can manage both con!gurations that recycle containers immediately and relaxed con!gurations with long keep-alive intervals that preserve containers for reuse. 

Figure 13 shows the cost of DROPS using di"erent keepalive intervals. The percentages annotated above each bar denote the fraction of allocation requests that are ful!lled using kept-alive containers. DROPS has a 0% failure rate for di"erent keep-alive intervals. Increasing the keep-alive interval increases the ratio of requests that are served using kept-alive containers. For instance, with a 10-minute keepalive interval, 53% of the requests are served using kept-alive containers. However, using a keep-alive interval increases the total cost. For instance, a 10-minute keep-alive interval incurs 19% higher total cost compared to immediately deleting idle containers (zero keep-alive interval). The cost of keeping containers alive accounts for 9% and 14% of the total cost of the 5-minute and 10-minute intervals, respectively. 

## **7.5 Aggressive Container Creation Optimization** 

TheVMpoolmaintainsanumberofidlecoresatanytime.The aggressive container creation optimization improves the success rate of allocation requests without incurring additional costs by utilizing the idle cores to create extra containers and expand the pools beyond their size limits. However, to avoid incurring additional costs, using cores to create extra containers does not lead to VM creation. 

A critical aspect of this optimization is how to distribute the idle cores across container pools. To e"ectively achieve this, we distribute the idle cores across pools proportional to 

**==> picture [240 x 92] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Total failure rate and cost (b)  Per-pool failure rate<br>**----- End of picture text -----**<br>


**Figure 14.** DROPS performance with and without the aggressive container creation optimization with an SLO of 95%. 

the pool size needed to achieve a 100% success rate. Thus, a pool with a bursty load will be assigned a larger number of extra containers than a pool with a lower burstiness. 

The VM pool optimization process of DROPS (Section 5.3) assumes on-demand core-to-container allocation. That is, cores are consumed only to re!ll a container pool. However, the aggressive container creation optimization pre-allocates cores to container pools, reducing the number of idle cores in the VM pool. This may cause a starvation problem where one container pool may become empty, while other pools have extra idle containers. To address this, DROPS redistributes the idle cores across container pools periodically. 

Figure 14 shows the failure rate and the cost of DROPS with and without the aggressive container creation optimization with an SLO of 95%. This optimization reduces the total failure rate from 3 _._ 2% to 2 _._ 3% (25% improvement) without incurring additional costs. Figure 14b illustrates that the reduction in failureratedi"ersacrosspools.Theper-poolimprovementdependsonthenumberofextracontainersassignedtoeachpool andthevolumeofallocationsthatthesecontainerscanhandle. 

## **7.6 Performance Across Geographic Regions** 

ThissectionevaluatesDROPSusingtracesfromfourAzureregions to verify that DROPS is capable of managing workloads with di"erent characteristics. Figure 15 compares DROPS to thestaticapproachthatwaspreviouslyusedintheproduction environment using 1-month traces from four regions: EastUS, EastUS2, WestUS2, and NorthEurope. Across all regions, both DROPS and the static approach achieve 0% failure rate. However, DROPS brings substantial cost reduction compared to ↑ the static approach, ranging from 57 80%, which indicates that the pools in the static approach are oversized to achieve 0% failure rate. 

## **8 Related Work** 

**Cloud workload analysis.** Prior research e"orts have analyzedcloudworkloadcharacteristicstooptimizeperformance and resource utilization. An early study [21] examined Azure VM workloads, detailing distributions of VM size, lifetime, resource consumption, and deployment patterns. Another 

1293 

**==> picture [241 x 139] intentionally omitted <==**

**Figure 15.** The performance of DROPS using traces from di"erent Azure regions. 

e"ort[11]presentsalarge-scaleanalysisofproductionserverless workloads to characterize Azure Functions, revealing extremeinvocationskew,unpredictabletrigger-drivenbehavior, and short-lived executions. Subsequent studies expanded this understanding:Arecentpaper[15]comparedprovisionedand serverless query workloads in Amazon Redshift, while other e"orts [13, 14] analyze serverless functions in Huawei’s platform. These e"orts focus on mitigating cold starts by studying function invocation workloads to identify daily periodicity and execution time variability. Major cloud providers have released production traces to enable broader research, including Microsoft Azure [22], Google Borg [23, 24], Alibaba [25], Huawei [13, 14], and Amazon [15, 16]. In contrast, we study container-allocation workloads to eliminate allocation failures that incur substantially higher latency than cold starts. **Cloudworkloadforecasting.** Priorworkshaveinvestigated the use of forecasting models to predict cloud workload. A recent study [15] on Amazon Redshift [26] evaluates the use of forecasting methods to predict the future query workload. It evaluates various forecasting methods, including time series analysis, ML-based techniques, and foundation models. The study concludes that these approaches often fail to deliver accurate predictions for burst-prone serverless workloads compared to provisioned counterparts. While this work focuses on database queries, its insights into forecasting limitations for dynamic serverless patterns remain relevant. Another recent study [13] explored using forecasting function invocations to optimize resource provisioning and reduce cold starts in Huawei’s cloud serverless platform. This paper identi!es load forecasting challenges, such as sporadic demand spikes and variable execution frequencies. The paper highlights the problem of !ne-grained long-term predictions, as modeling per-second data over weeks poses scalability issues for standard forecasting models. 

While prior e"orts focus on analyzing and predicting user function invocations or generic workload patterns, they primarily evaluate performance using standard forecasting error metrics without assessing system-level impacts such as latency or resource e#ciency. In contrast, our work focuses on the container-allocation workload, which, to the best of our 

knowledge, has not been previously studied. Additionally, we propose a new pool optimization method and compare it with predictive and reactive methods. 

**Cold start mitigation.** A signi!cant body of research focuses on reducing cold start latency in serverless platforms. Mohan et al. [27] pre-allocate reusable virtual network interfaces, decoupling provisioning from container initialization. SOCK [28] optimizes Python function loading in OpenLambda through library caching and lightweight isolation, while SAND [29] eliminates cold starts for subsequent invocationswithinanapplicationviasharedsandboxing.Replayable Execution [30] accelerates JVM-based FaaS startup by checkpointing and sharing memory states across containers. These e"orts focus on reducing the latency of the cold start. Our work is orthogonal to these e"orts as we propose a new optimization method to manage resource pools. 

## **9 Conclusion** 

We present the !rst study of container-allocation workload in serverless platforms. Our characterization of real traces from Azure Functions reveals key workload properties including high burstiness, lack of periodicity, and long-tail creation latencies. We !nd that the cost of maintaining the resource pools dominates the platform operational cost. Our evaluation shows that predictive and reactive pool sizing methods fail to meet the target SLO. Motivated by these !ndings, we present DROPS, a statistical, data-driven method to optimize resource pools in serverless platforms. DROPS leverages historical traces to construct a pool-size-to-success-rate mapping, enabling precise pool sizing to meet target SLOs while minimizing cost. Evaluation on production traces shows that DROPS outperforms existing methods, achieving the desired SLO at a signi!cantly lower cost. 

## **Acknowledgments** 

We thank our shepherd, John Wilkes, for his insightful feedback and guidance. We also thank Bilal Alam, Ricardo Bianchini,RodrigoFonseca,ShanLu,WilliamMoy,PhilipSchowitz, and anonymous reviewers for many helpful comments and suggestions. The research team was supported by grants from the National Cybersecurity Consortium (NCC), Natural Sciences and Engineering Research Council of Canada (NSERC) (ALLRP-561423-20 and RGPIN-2025-03332), Ontario Research Fund – Research Excellence program (ORF-RE012051). Ahmed is supported by a Fields Postdoctoral Fellowship. 

## **References** 

- [1] Azure Functions. h!ps://azure.microso".com/en-us/services/ functions/. Accessed: 2025-05-16. 

- [2] AWS Lambda. h!ps://aws.amazon.com/lambda/. Accessed: 2025-05-16. 

- [3] Google Cloud. What is Cloud Run. https://cloud.google.com/run/docs/overview/what-is-cloud-run. Accessed: 2025-05-16. 

- [4] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent 

1294 

networks. _International Journal of Forecasting_ , 36(3):1181–1191, 2020. 

- [5] Bryan Lim, Sercan Ö. Arık, Nicolas Loe", and Tomas P!ster. Temporal fusion transformers for interpretable multi-horizon time series forecasting. _International Journal of Forecasting_ , 2021. 

- [6] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In _International Conference on Learning Representations (ICLR)_ , 2023. 

- [7] Vassilis Assimakopoulos and Konstantinos Nikolopoulos. The theta model: A decomposition approach to forecasting. _International Journal of Forecasting_ , 16(4):521–530, 2000. 

- [8] George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung. _Time Series Analysis: Forecasting and Control_ . 1970. 

- [9] Rob J. Hyndman, Anne B. Koehler, J. Keith Ord, and Ralph D. Snyder. _Forecasting with Exponential Smoothing: The State Space Approach_ . 2008. 

- [10] Dah-Ming Chiu and Raj Jain. Analysis of the increase and decrease algorithms for congestion avoidance in computer networks. _Computer Networks and ISDN Systems_ , 17(1):1–14, 1989. 

- [11] Mohammad Shahrad, Rodrigo Fonseca, Inigo Goiri, Gohar Chaudhry, Paul Batum, Jason Cooke, Eduardo Laureano, Colby Tresness, Mark Russinovich, and Ricardo Bianchini. Serverless in the wild: Characterizing and optimizing the serverless workload at a large cloud provider. In _2020 USENIX Annual Technical Conference (USENIX ATC 20)_ , pages 205–218, July 2020. 

- [12] Yanqi Zhang, Íñigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh Elnikety, Christina Delimitrou, and Ricardo Bianchini. Faster and cheaper serverless computing on harvested resources. In _Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles_ , pages 724–739, 2021. 

- [13] Artjom Joosen, Ahmed Hassan, Martin Asenov, Rajkarn Singh, Luke Darlow, Jianfeng Wang, and Adam Barker. How does it function? characterizing long-term trends in production serverless workloads. In _Proceedings of the 2023 ACM Symposium on Cloud Computing_ , pages 443–458, 2023. 

- [14] Artjom Joosen, Ahmed Hassan, Martin Asenov, Rajkarn Singh, Luke Darlow, Jianfeng Wang, Qiwen Deng, and Adam Barker. Serverless cold starts and where to !nd them. _arXiv preprint arXiv:2410.06145_ , 2024. 

- [15] Yanlei Diao, Dominik Horn, Andreas Kipf, Oleksandr Shchur, Ines Benito, Wenjian Dong, Davide Pagano, Pascal Pfeil, Vikram Nathan, Balakrishnan Narayanaswamy, and Tim Kraska. Forecasting algorithms for intelligent resource scaling: An experimental analysis. In _Proceedings of the 2024 ACM Symposium on Cloud Computing_ , SoCC ’24, page 126–143, 2024. 

   - [21] Eli Cortez, Anand Bonde, Alexandre Muzio, Mark Russinovich, Marcus Fontoura, and Ricardo Bianchini. Resource central: Understanding and predicting workloads for improved resource management in large cloud platforms. In _Proceedings of the 26th Symposium on Operating Systems Principles_ , SOSP ’17, page 153–167, 2017. 

   - [22] Ashraf Mahgoub, Edgardo Barsallo Yi, Karthick Shankar, Sameh Elnikety, Somali Chaterji, and Saurabh Bagchi. ORION and the three rights: Sizing, bundling, and prewarming for serverless DAGs. In _16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)_ , pages 303–320, 2022. 

   - [23] Abhishek Verma, Luis Pedrosa, Madhukar Korupolu, David Oppenheimer, Eric Tune, and John Wilkes. Large-scale cluster management at Google with Borg. In _Proceedings of the Tenth European Conference on Computer Systems_ , EuroSys ’15, 2015. 

   - [24] Muhammad Tirmazi, Adam Barker, Nan Deng, Md E Haque, Zhijing Gene Qin, Steven Hand, Mor Harchol-Balter, and John Wilkes. Borg: the Next Generation. In _Proceedings of the !fteenth European conference on computer systems_ , pages 1–14, 2020. 

   - [25] Qizhen Weng, Wencong Xiao, Yinghao Yu, Wei Wang, Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin, and Yu Ding. MLaaS in the wild: Workload analysis and scheduling in Large-Scale heterogeneous GPU clusters. In _19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)_ , pages 945–960, Renton, WA, April 2022. 

   - [26] Amazon Web Services. Amazon Redshift. h!ps://aws.amazon.com/ redshi"/, 2024. Accessed: 2025-04-22. 

   - [27] Anup Mohan, Harshad Sane, Kshitij Doshi, Saikrishna Edupuganti, Naren Nayak, and Vadim Sukhomlinov. Agile cold starts for scalable serverless. In _11th USENIX Workshop on Hot Topics in Cloud Computing (HotCloud 19)_ , 2019. 

   - [28] Edward Oakes, Leon Yang, Dennis Zhou, Kevin Houck, Tyler Harter, Andrea Arpaci-Dusseau, and Remzi Arpaci-Dusseau. SOCK: Rapid task provisioning with Serverless-Optimized containers. In _2018 USENIX annual technical conference (USENIX ATC 18)_ , pages 57–70, 2018. 

   - [29] Istemi Ekin Akkus, Ruichuan Chen, Ivica Rimac, Manuel Stein, Klaus Satzke, Andre Beck, Paarijaat Aditya, and Volker Hilt. SAND: Towards High-Performance serverless computing. In _2018 USENIX Annual Technical Conference (USENIX ATC 18)_ , pages 923–935, Boston, MA, July 2018. 

   - [30] Kai-Ting Amy Wang, Rayson Ho, and Peng Wu. Replayable execution optimized for page sharing for a managed runtime environment. In _Proceedings of the Fourteenth EuroSys Conference 2019_ , pages 1–16, 2019. 

- [16] Alexander van Renen, Dominik Horn, Pascal Pfeil, Kapil Vaidya, Wenjian Dong, Murali Narayanaswamy, Zhengchun Liu, Gaurav Saxena, Andreas Kipf, and Tim Kraska. Why TPC is not enough: An analysis of the Amazon Redshift $eet. _Proceedings of the VLDB Endowment_ , 17(11):3694–3706, 2024. 

- [17] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Syndar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. _arXiv preprint arXiv:2403.07815_ , 2024. 

- [18] Oleksandr Shchur, Caner Turkmen, Nick Erickson, Huibin Shen, Alexander Shirkov, Tony Hu, and Yuyang Wang. AutoGluonTimeSeries: AutoML for probabilistic time series forecasting. In _International Conference on Automated Machine Learning_ , 2023. 

- [19] Rob J. Hyndman and George Athanasopoulos. _Forecasting: Principles and Practice_ . 3rd edition, 2021. 

- [20] Milton Abramowitz and Irene A. Stegun. _Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables_ . 1972. 

1295 

## **A Artifact Appendix** 

## **A.1 Abstract** 

Our artifact is an implementation of the DROPS algorithm, together with the simulator required to run the experiments described in the paper. Speci!cally, it contains the following items: 

- The source code of the serverless platform simulator, written in C#, which includes: 

- Our reference implementation of DROPS. 

- Ourimplementationofalternativeresourceoptimization methods. 

- Our reference implementation of the aggressive container creation optimization. 

- The tools to train the forecasting models. 

- The tools to generate the !gures in this paper. 

## **A.2 Description & Requirements** 

**A.2.1 How to access.** The artifact is publicly available in the following GitHub repository: https://github.com/UWASL/DROPS.git A snapshot of this repository is archived at: https://zenodo.org/records/17051480 

## **A.2.2 Hardware dependencies.** To run our experiments, you need the following: 

- A machine running Linux. Our code is tested on Ubuntu 22.04. However, it should work on other operating systems, including Windows and MacOS. 

- For model training, at least 128 GB of RAM is required to train machine learning models. However, training models is optional: we provide the output trace of the trained models that is needed to reproduce the results in the paper without retraining models. 

**A.2.3 Software dependencies.** Our code has been tested on Ubuntu 22.04. The simulator requires the .NET 9.0 Runtime.Thetrainingprocessrequiresthe AutoGluon library. The plotting scripts require Python 3, along with the pandas, numpy, and matplotlib libraries. 

**A.2.4 Benchmarks.** Reproducing the results in the paper requires two traces (i.e., container-allocation trace and lifecycle trace). Follow the steps under the set-up section (A.3) to download the traces. 

## **A.3 Set-up** 

To download DROPS’ source code, run the following: git clone https://github.com/UWASL/DROPS.git Then, install all dependencies: 

./setup-scripts/install-dep.sh ./training/training-dep.sh 

Then, build DROPS: ./setup-scripts/build.sh 

Then, run the following command to fetch the input traces. 

## ./setup-scripts/fetch-traces.sh 

The environment is now ready to run all experiments. 

## **A.4 Evaluation work#ow** 

Please see the README.md !le in the artifact repository for a detailed description of the process. The rest of this section discusses our major claims and the experiments that support them. 

**A.4.1 MajorClaims.** Ourpapermakesthefollowingmajor claims. 

- **C1.** DROPS meets the target SLO while signi!cantly reducing the cost compared to other approaches. Experiments 1 and 2 and Figure 7 and 12 support this claim. 

- **C2.** DROPSenablesprecisepoolsizingfordi"erentSLO levels, ensuring that the failure rate is below the target SLO. Experiment 2 and Figure 12 support this claim. 

- **C3.** Prediction-based approaches fail to meet the target SLO, and their performance is signi!cantly a"ected by the prediction interval. Larger prediction intervals lead to a higher failure rate for the constant and Poisson approachesandalowerfailureratefortheconcentratedload approach. Experiments 1 and 3 and Figures 8 and 7 support this claim. 

- **C4.** The reactive approach fails to meet the target SLO, anditsperformanceissigni!cantlya"ectedbythescaleup and scale-down factors. Increasing the scale-up factorreducesthefailurerateandincreasesthecost.Onthe other hand, increasing the scale-down factor increases the failure rate and reduces the cost. Experiment 4 and Figure 11 support this claim. 

- **C5.** The aggressive container creation optimization reduces the failure rate without incurring additional cost. Experiment 5 and Figure 14a support this claim. 

## **A.4.2 Experiment E1: Performance Comparison [0.5** 

**compute-hours].** Thisexperimentcomparesthefailurerate, cost, and ful!llment latency of di"erent optimization methods. The results are used to generate Figure 7 and support claims C1, C3, and C4. 

**[Preparation]** None. 

**[Execution]** Run the following commands: 

cd /<repo_root>/experiments/ ./scripts/fig6.sh 

**[Results]** Two !gures similar to Figure 7a and Figure 7b are generatedunder /<repo_root>/experiments/ directory.Raw results for failure rate, cost breakdown, and latency are generatedin csv formatunder /<repo_root>/experiments/fig6 directory. 

**A.4.3 Experiment E2: DROPS Sensitivity Analysis [0.5 compute-hours].** This experiment shows that DROPS enables precise pool sizing for di"erent SLOs. The results are used to generate Figure 12 and support claim C2. **[Preparation]** None. 

1296 

**[Execution]** Run the following commands: cd /<repo_root>/experiments/ ./scripts/fig11.sh 

**[Results]** Two!guressimilartoFigure12aandFigure12bare generatedunder /<repo_root>/experiments/ directory.Raw results for failure rate, cost breakdown, and latency are generated in csv format under /<repo_root>/ experiments/fig11 directory. 

**A.4.4 ExperimentE3:PredictionMethod[0.5computehours].** This experiment evaluates the prediction method usingdi"erentpoolsizemappingapproaches(i.e.,constant,Poisson, and concentrated) using di"erent prediction intervals. The results are used to generate Figure 8 and support claim C3. **[Preparation]** None. 

## **A.5 Notes on Reusability** 

Our artifact includes a simulator of a real serverless platform. This simulator can be used to compare the performance and cost of di"erent resource sizing methods. Also, it enables estimating the bene!ts of di"erent optimizations or improvements in the platform (e.g., the estimated cost reduction of reducing the container creation latency by 50%). Our simulator requires input traces in a speci!c format. We include sample traces for the allocation trace and the life cycle trace. Other traces must be transformed to the required format before runningthesimulator.Also,thesimulatoracceptsa JSON !lespecifying a list of experiments to simulate. Each experiment speci!es an optimization method along with a set of parameters. 

**[Execution]** Run the following commands: 

cd /<repo_root>/experiments/ ./scripts/fig7.sh 

**[Results]** Two !gures similar to Figure 8a and Figure 8b are generatedunder /<repo_root>/experiments/ directory.Raw results for failure rate and cost breakdown are generated in csv format under /<repo_root>/experiments/fig7 directory. 

**A.4.5 Experiment E4: Reactive Method [1 computehour].** This experiment evaluates the reactive method with di"erentscale-upandscale-downfactors.Theresultsareused to generate Figure 11 and support claim C4. **[Preparation]** None. 

**[Execution]** Run the following commands: cd /<repo_root>/experiments/ ./scripts/fig10a.sh ./scripts/fig10b.sh 

**[Results]** Two !gures similar to the ones shown in Figure 11 are generated under /<repo_root>/experiments/ directory. Raw results for failure rate and cost breakdown aregeneratedin csv formatunder /<repo_root>/experiments/ fig10a and /<repo_root>/experiments/fig10b directories. 

**A.4.6 Experiment E5: Aggressive Container Creation [0.5 compute-hours].** This experiment evaluates the aggressive container creation optimization. The results are used to generate Figure 14 and support claim C5. **[Preparation]** None. 

**[Execution]** Run the following commands: 

cd /<repo_root>/experiments/ ./scripts/fig12.sh 

**[Results]** A !gures similar to Figure 14a is generated under /<repo_root>/experiments/ directory. Raw results for failure rate and cost breakdown are generated in csv format under /<repo_root>/experiments/fig12 directory. 

1297 

