# <span id="page-8-0"></span>5.5 Overall Exploration & Optimization Process

Based on the table-format schedule representations and the modeled cost function for hardware performance, we are now ready to describe the automatic exploration process for inter-layer scheduling. Note that selecting the proper batch splitting plan ( $BS_{\rm sub}$ ), as another important decision factor affecting the overall scheduling scheme, will be integrated into the search process.

Fig. 7 shows the overall exploration for inter-layer scheduling. Given the pre-determined batch size BS by the workload, we first enumerate all its factors as the candidates for  $BS_{\rm sub}$ . Then, for each

possible  $BS_{\mathrm{sub}}$ , the corresponding ScT is built with the constraints described by Eq. 1 - 6. Since all constraints are linear, we can apply piecewise linear approximation and solve the problem using an MILP solver to find the corresponding  $s_i$ 's that minimize the computation-related EDP, a bilinear term that can be linearized via McCormick envelope [25]:

$$\{s_i\}_{\text{opt}} = \arg\min_{\{s_i\}} L_{\text{comp}}^p \times E_{\text{comp}}^q, \tag{22}$$

where p and q are hyper-parameters setting the importance of latency and energy, respectively. With the above-identified  $s_i$ 's, we can specify ScT's and further calculate the corresponding minimized EDP cost for each  $B_{\rm sub}$  candidate. After preserving  $K_1$  possible  $B_{\rm sub}$ 's with the smallest  $K_1$   $L^p_{\rm comp} \times E^q_{\rm comp}$ 's, we further use them and the corresponding  $s_i$ 's to build  $K_1$  MeT's. Again, an MILP solver is applied to minimize the data traffic-related EDP cost:

$$\{\mathbf{MeT}_{i,j}^D, \mathbf{MeT}_{i,j}^S\}_{\mathrm{opt}} = \arg\min_{\{\mathbf{MeT}_{i,j}^D, \mathbf{MeT}_{i,j}^S\}} L_{\mathrm{traffic}}^P \times E_{\mathrm{traffic}}^q. \tag{23}$$

The candidate list for  $B_{\rm sub}$  is then further shrunk by preserving only  $K_2$   $B_{\rm sub}$ 's that correspond to the smallest  $L^p_{\rm traffic} \times E^q_{\rm traffic}$ 's. Note that for the training workload, another round of table construction and  $B_{\rm sub}$  candidate reduction are needed.

Finally, intra-layer scheduling is applied to the remaining  $K_2$  candidates to update total latency and energy consumption after considering the optimization within each hardware tile as follows:

$$L_{\text{Total}} = \max \left( \sum_{i} s_{i} \times \max_{j} \left( L_{\text{comp},i,j} \alpha_{i,j} \right), L_{\text{traffic}} + \sum_{i,j} \beta_{i,j} \right), \tag{24}$$

$$E_{\text{Total}} = E_{\text{comp, unit}} \times \sum_{i,j} \frac{s_i \text{Workload}_j \alpha_{i,j}}{u_{i,j}} + E_{\text{traffic}} + \sum_{i,j} \gamma_{i,j},$$
 (25)

where  $\alpha_{i,j}$  is the factor considering the actual hardware utilization for processing sub-block  $B_j$  in State-i.  $\beta_{i,j}$  and  $\gamma_{i,j}$  are the corresponding data traffic-related latency and energy consumption associated with intra-layer scheduling. Here,  $\alpha_{i,j}$ ,  $\beta_{i,j}$ , and  $\gamma_{i,j}$  are obtained from the intra-layer scheduler. Thanks to the abstraction of hardware tiles in inter-layer scheduling, the existing intra-layer schedulers can be easily applied to our framework as a plug-in.

**Remark** 3 (Addressing Challenge #1: Unified Representation of Core Scheduling Factors). The ScT and MeT table formats, together with sub-batch size selection ( $BS_{\rm sub}$ ), provide a unified and extensible representation of inter-layer scheduling behavior. These tables explicitly record execution states and memory status, which are determined by four design factors (E,P,R,B). This joint representation ensures that all key design factors are consistently captured within a single framework—directly addressing Challenge #1.

Remark 4 (Addressing Challenge #3: Structured and Exhaustive Scheduling via MILP). The table-based formulation using ScT and McT naturally leads to an MILP, which encodes scheduling constraints with high structural regularity. This enables exhaustive exploration of the scheduling space using off-the-shelf MILP solvers, offering both fast convergence and globally optimal solutions for intra-block scheduling—effectively addressing Challenge #3.

