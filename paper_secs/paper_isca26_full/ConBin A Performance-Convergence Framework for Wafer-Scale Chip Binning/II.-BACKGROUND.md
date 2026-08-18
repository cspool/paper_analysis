# II. BACKGROUND

### *A. Manufacturing Faults in Wafer-Scale Chips*

Modern WSCs follow standard CMOS manufacturing test flows, including scan-based structural testing and at-speed timing validation to detect permanent logic defects and delay faults [3], [52], [56]. These tests are conducted at wafer sort and final test, and, together with post-silicon failure diagnosis, can localize defects to standard-cell regions. At the architectural level, such localized failures are abstracted as faulty PEs, routers, or interconnect links, which motivates our modeling granularity. *We assume high structural fault coverage (including stuck-at, bridging, and transition faults) and treat all diagnosed faulty components as permanently disabled.*

The distribution of faults on WSCs exhibits both random and clustered patterns [19], where the clustered faults appear as spatially contiguous defective regions, due to process variations, thermal stress, die warpage or screw holes [1], [21], [24], [35], [39], [55]. These faults can be classified into several

![](_page_2_Figure_7.jpeg)

Fig. 3. Scalability Challenges of Topology-Keeping Fault-Tolerant Methods [11], [45].

common fault patterns, including Local, Scratch, Edge-Local, and Center faults [58].

Although fault patterns and distributions are generally unpredictable, Stapper's analysis [49] revealed a clear distance correlation among wafer-scale defects: nearby defects within roughly four defect-sampling quadrats tend to form spatial clusters. Such correlation causes redundancy to fail when spare resources lie within the same cluster as defective regions, rendering purely locality-based spares ineffective. Motivated by this insight, our design strategically introduces long-range redundant links that account for spatial fault correlation.

# II. BACKGROUND

### *A. Manufacturing Faults in Wafer-Scale Chips*

Modern WSCs follow standard CMOS manufacturing test flows, including scan-based structural testing and at-speed timing validation to detect permanent logic defects and delay faults [3], [52], [56]. These tests are conducted at wafer sort and final test, and, together with post-silicon failure diagnosis, can localize defects to standard-cell regions. At the architectural level, such localized failures are abstracted as faulty PEs, routers, or interconnect links, which motivates our modeling granularity. *We assume high structural fault coverage (including stuck-at, bridging, and transition faults) and treat all diagnosed faulty components as permanently disabled.*

The distribution of faults on WSCs exhibits both random and clustered patterns [19], where the clustered faults appear as spatially contiguous defective regions, due to process variations, thermal stress, die warpage or screw holes [1], [21], [24], [35], [39], [55]. These faults can be classified into several

![](_page_2_Figure_7.jpeg)

Fig. 3. Scalability Challenges of Topology-Keeping Fault-Tolerant Methods [11], [45].

common fault patterns, including Local, Scratch, Edge-Local, and Center faults [58].

Although fault patterns and distributions are generally unpredictable, Stapper's analysis [49] revealed a clear distance correlation among wafer-scale defects: nearby defects within roughly four defect-sampling quadrats tend to form spatial clusters. Such correlation causes redundancy to fail when spare resources lie within the same cluster as defective regions, rendering purely locality-based spares ineffective. Motivated by this insight, our design strategically introduces long-range redundant links that account for spatial fault correlation.

