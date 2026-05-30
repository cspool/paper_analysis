# *C. Performance Isolation*

In this subsection, we compare the performance of Smart-DIMM with other configurations when co-running secure Nginx and a cache-intensive application. Table [I](#page-10-2) compares the average

<span id="page-10-2"></span>**Table I. Slow down of co-running scenario. The Nginx slowdown in each column is normalized to the solo run of the same configuration.**

| Application | CPU   | SmartNIC | QuickAssist | SmartDIMM |
|-------------|-------|----------|-------------|-----------|
| Ngnix       | 15.8% | 7.3%     | 28.7%       | 9.5%      |
| 505.mcf     | 15.5% | 8.7%     | 37.9%       | 10.3%     |

<span id="page-10-1"></span>![](_page_10_Figure_8.jpeg)

**Fig. 12. Performance of Nginx when executing compression on different configurations with 4KB and 16KB message sizes. Higher is better for RPS and lower is better for CPU and memory utilization. All the results are normalized to that of the** *CPU* **configuration.**

slowdown of Nginx's RPS and the mcf workload from the SPEC2017 benchmark suite [\[82\]](#page-16-6) when co-running them on two separate cores (as a baseline we run Nginx and mcf on the server individually). We co-run 10 mcf instances with an Nginx server utilizing 10 threads pinned to 10 separate physical cores. As shown, offloading TLS to SmartDIMM reduces the interference for both Nginx and mcf by 9.5% and 10.3%.

Note that although SmartDIMM experiences 2.2 percentage points higher slowdown for Nginx compared with *SmartNIC*, the absolute requests per second when co-running Nginx is still higher for SmartDIMM: 569609 vs. 377879. The higher absolute requests per second for SmartDIMM results in slightly higher interference for mcf compared with the *SmartNIC* configuration.

