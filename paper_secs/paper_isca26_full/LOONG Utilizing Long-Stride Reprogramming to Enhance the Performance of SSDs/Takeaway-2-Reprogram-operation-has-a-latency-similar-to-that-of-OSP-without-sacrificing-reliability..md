# Takeaway #2: Reprogram operation has a latency similar to that of OSP without sacrificing reliability.

2) Reliability Impact of Error Sources: The previous section evaluates reliability immediately after finishing LOONG's second step. We here extend this analysis by incorporating key error sources: retention and P/E cycles. For fair comparison, the standard OSP is configured with identical retention and P/E cycle conditions as LOONG.

For retention, the flash chip is heated to 120°C for three hours, which corresponds to retention of one year at 40°C [1]. For P/E cycles, flash blocks undergo repeated program and erase operations (program, reprogram, and erase operations for LOONG) over a total of 1K cycles, consistent with [19]. The voltage distribution and number of bit errors are displayed in Figure 8 <sup>1</sup>. As shown in Figure 8, the voltage distributions of all states can be easily distinguished from those of neighboring states. This indicates that, compared to the standard OSP, the long-stride reprogram operation can still achieve good reliability. However, we also observe that the voltage distribution of the long-stride reprogramming is more likely to be left-shift. This is because, due to BPD, electrons are harder to inject into the desired state, resulting in more cells being clustered on the left side.

Next, we give a more detailed analysis. We collect the occurrence of bit errors per page after OSP and long-stride reprogram operations. Considering the error sources (both P/E cycles and retention time), we observe that the long-stride reprogram operation introduces an average of 96 bit errors per page, which is well within the correctable capability of ECC (1280-bit/page). Interestingly, we also find that the standard OSP tends to generate more bit errors than the long-stride reprogram operation. In the worst case, OSP causes an average of 101.8 bit errors per page. The reason why long-stride reprogram operation has fewer bit errors primarily stems from its two-step programming process. In the first step, the cell is programmed to state E or P1, raising the cell's voltage. In the second step, the long-stride reprogram operation brings the cell's voltage to the desired state. However, due to the impact of BPD, it is harder to increase the cell's voltage to a high value (the same reason why the voltage distribution of longstride reprogram operation in Figure 8 is more left-shifted). As a result, after these two steps, compared to standard OSP, the voltage distribution of reprogram operation has a smaller voltage difference, where state E tends to have a higher voltage

<sup>1</sup>Voltage calibration is inherently involved during the evaluation to fine-tune read reference voltage for minimizing the number of bit errors [2], [5], [22].

![](_page_5_Figure_7.jpeg)

Fig. 8. Distribution of OSP and Long-Stride Reprogramming.

TABLE III
THE COMPARISON OF DISTURB EFFECTS.

|              | Count | Operation Time (µs) | Voltage Drop  |
|--------------|-------|---------------------|---------------|
| One-shot PRG | 1279  | 1100                | $V_{p\_pass}$ |
| pSLC PRG     | 1279  | 114                 | $V_{p\_pass}$ |
| ReP          | 1279  | 955                 | $V_{p\_pass}$ |
| pSLC Read    | 1279  | 25                  | $V_{r\_pass}$ |

while other states have lower voltages. In 3D NAND flash memory, higher cell voltages accelerate electron leakage [2]–[4], [6], a process quantified by the Fowler-Nordheim (FN) tunneling model where leaked electrons follow  $N_e \propto J_{FN} \cdot T_S$  [21], [36], [45]. Here, total number of electrons leaked from a cell is defined by  $N_e$ ,  $J_{FN}$  depends exponentially on the cell's threshold voltage  $(V_{th})$ , while  $T_S$  denotes retention time. Due to its two-step programming, LOONG maintains a lower average  $V_{th}$  than OSP, reducing leakage currents and slightly improving reliability [5], [46].

Takeaway #3: The pages programmed in pSLC mode can be reprogrammed to TLC mode with long stride without sacrificing reliability.

#### C. Disturb Effects

Long-stride reprogramming increases the number of program and read operations per block compared to standard OSP, raising concerns about heightened read and program disturb effects. Like retention issues, disturb errors are governed by the FN tunneling model, where the number of unintentionally injected electrons  $(N_e)$  follows the relationship:  $N_e \propto J_{FN} \cdot T_S$ . In this model, the tunneling current density  $(J_{FN})$  is driven by the voltage differential between the pass voltage  $(V_{pass})$  and the cell threshold voltage  $(V_{th})$ , while  $T_S$  represents the cumulative operation time. Consequently, evaluating disturb severity simplifies to analyzing the variations in this voltage gap and the total stress duration.

For both standard OSP and long-stride programming, the worst-case disturb occurs at the last WL of a block, as it endures cumulative stress from all preceding program and read operations. This impact is most severe for cells remaining in the erased state, where a minimum  $V_{th}$  maximizes the potential difference  $(V_{pass}-V_{th})$ . Under these conditions, the cumulative disturb effects are summarized as follows:

As shown in Table III, the last WL in a 1280-WL block endures 1,279 disturb instances. The stress duration and voltage drop are calculated for the worst-case erased state, where

 $V_{p\_pass}$  and  $V_{r\_pass}$  denote the pass voltages for program and read operations, respectively, with  $V_{r\_pass} < V_{p\_pass}$  [45]. While the first row represents the OSP baseline, the subsequent rows detail the three stages of long-stride reprogramming: pSLC programming, pSLC read, and final reprogramming. Consequently, long-stride reprogramming exhibits a lower total disturb impact than standard OSP, as it reduces both cumulative operation time and the effective voltage difference.

#### V. CASE STUDIES

In this section, we integrate LOONG into two case studies, including GC-oriented optimization and program-oriented optimization, to verify its efficiency.

