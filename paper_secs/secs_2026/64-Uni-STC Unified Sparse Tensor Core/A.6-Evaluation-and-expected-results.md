# A.6 Evaluation and expected results

Upon completion of the experiments, all generated charts are stored in the container directory /root/Sim/fig/. We provide two methods to inspect these results.

#### A.6.1 Result Inspection

#### Option 1: Export to Host (Recommended)

For the best viewing experience and to facilitate comparison with the paper, we recommend copying all generated figures to host. Execute the following command on your host terminal:

```
$ docker cp HPCA-Pap313:/root/Sim/fig ./uni-stc-results
```

*Explanation:* This will create a folder named uni-stc-results in your current directory containing all generated .png files.

#### Option 2: In-Terminal Preview

For users employing modern terminal emulators capable of image rendering (e.g., Kitty, iTerm2, or Ghostty), you can preview results directly inside the container without exporting.

```
# Inside the container
(container)$ qs icat /root/Sim/fig/15.png
```

#### A.6.2 Detailed Analysis

We outline the specific observations required to validate the artifacts below. Note: The simulator provided in this artifact is a lightweight version extracted from Accel-Sim to facilitate rapid verification. As it excludes power modeling for register I/O, the observed energy savings for Uni-STC may be *higher* than the conservative figures reported in the paper.

- Fig. 15 (Format Overhead): Verify that the BBC format spacereduction (y-axis) *increases* as the density (x-axis) increases.
- Fig. 16 (Random SpGEMM Performance): Uni-STC should demonstrate performance that is *equal to or greater than* other baseline hardwares.
- Fig. 17 & 20 (Overall Performance & Efficiency):
  - Fig. 17 (Representative): Confirm that Uni-STC achieves the highest values in speedup, energy reduction, and area efficiency.
  - Fig. 20 (Full Dataset): Confirm that these performance gains are consistent across the full SuiteSparse collection (2,800+ matrices).
- Fig. 18 (Energy Breakdown): Verify that Uni-STC achieves the *lowest total energy consumption*. Observe that the energy consumption is balanced across the three internal operations (Fetch, Schedule, Compute), showing similar values.
- Fig. 19 (Traffic & Network Scale): Verify that Uni-STC incurs the *lowest data traffic* compared to other architectures. Confirm that Uni-STC supports the required enabled network scale as depicted in the figure.
- Fig. 21 (AMG Solver): Uni-STC should exhibit a higher speedup ratio compared to other baseline hardwares.
- Fig. 22 (Scalability EED): Compare the Energy Efficiency Density (EED) between Uni-STC(8) and Uni-STC(4): For SpMV / SpMSpV, Uni-STC(8) is slightly *lower* than Uni-STC(4). For SpMM / SpGEMM, Uni-STC(8) is *higher* than Uni-STC(4).

#### A.7 Methodology

Submission, reviewing and badging methodology:

- https://www.acm.org/publications/policies/artifa ct-review-and-badging-current
- https://cTuning.org/ae

<sup>4</sup> https://drive.google.com/file/d/ 1Pp3BBOvU8nGoB12bb4o3wZs41twiXwXM