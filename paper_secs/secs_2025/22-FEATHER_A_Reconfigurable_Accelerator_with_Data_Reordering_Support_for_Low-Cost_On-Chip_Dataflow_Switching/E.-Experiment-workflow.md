# E. Experiment workflow

- 1) Experiment Set 1, End-to-end Inference on FPGA:
- •We require ZCU104 with pre-built PYNQ image, which provides a jupyter notebook portal.
- Copy the provided "feather/feather.ipynb" into the jupyter notebook and then run all blocks.
- 2) Experiment Set 2, LayoutLoop Analytic Analysis: run the following commands within the pre-built docker. The DSE might take 1 day to finish, depending on the machine.

```
$ docker run -it feather_layoutloop

$ git clone provided_url>
$ cd FEATHER/LayoutLoop/configurations
$ git pull
$ make clean
$ make conv_dse # DSE for ResNet-50, Mob-v3
$ make gemm_dse # DSE for Bert
```

3) Experiment Set 3, Synthesis and PNR under TSMC 28nm: For synthesis, we use design compiler "dc\_shell"

```
<setup environment for synopsys>
$ cd FEATHER/FEATHER_RTL/scripts/
$ source :run_syn
```

For place and routing, we use innovus.

```
<setup environment for innovus>
<Finish Synthesis First>
$ cd FEATHER/FEATHER_RTL/
$ innovus
> source PnR.tcl
```

#### F. Evaluation and expected results

- 1) Exp. Set 1: End-to-end latency on FPGA: The layerwise latency of running various models will be shown in the end of the jupyter notebook. We normalize results from different designs using "normalized throughput per PE", where throughput is measured by inverse of latency under single batch. The visualized result is shown in Fig. 12.
- 2) Exp. Set 2: LayoutLoop Analytic Analysis: Per-layer results from Layoutloop could be found at "LayoutLoop/configurations/results" with following naming pattern.
- design\_name\_layout\_policy\_slowdown.csv
- design\_name\_layout\_policy\_utilization.csv
- design\_name\_layout\_policy\_pj\_commpute.csv
- design\_name\_layout\_policy\_cycle.csv

We calculate GeoMean of "pJ/compute" and "cycle", and then normalize all results by FEATHER's performance with the visualized results shown as Fig. 13.

- 3) Exp. Set 3: Synthesis and PNR under TSMC 28nm: The final reports of synthesizing FEATHER at a specific scale will be listed in the "reports" folder, including
- feather\_top\_area.rpt
- feather top dw area.rpt

- feather\_top\_power.rpt
- feather\_top\_timing.rpt

The final reports of PnR contain

- area.rpt, which contains Post-PnR area value.
- power.rpt, which contains Post-PnR power value.
- time, timingReports. # Both are timing reports.

TABLE V: Post-PnR FEATHER Area/Power at various shapes.

| Shape          | Area (μm <sup>2</sup> ) | Power(mW) | Frequency (GHz) |
|----------------|-------------------------|-----------|-----------------|
| 64×128         | 36920519.69             | 26400.00  | 1.00            |
| $64 \times 64$ | 18389176.19             | 13200.00  | 1.00            |
| $32 \times 32$ | 2727906.70              | 961.70    | 1.00            |
| $16 \times 32$ | 965665.10               | 655.55    | 1.00            |
| 16×16          | 475897.19               | 323.48    | 1.00            |
| $8 \times 8$   | 97976.46                | 65.25     | 1.00            |
| $4\times4$     | 24693.98                | 16.28     | 1.00            |

#### G. Experiment customization

1) Exp. Set 2: LayoutLoop Analytic Analysis:

**Different Configurations:** LayoutLoop adopts the same architecture, dataflow constraint and mapper configurations format as TimeLoop with detailed documentations listed at https://timeloop.csail.mit.edu/v4/input-formats/design. Further, we argument LayoutLoop to support the analysis of layouts. The layout definition is shown in §3. The locations of these configurations are listed below.

- architecture design: "FEATHER/LayoutLoop/configurations/arch\_designs/"
- dataflow constraints: "FEATHER/LayoutLoop/configurations/arch\_designs/systolic\_constraint/mapspace.yaml", the dataflow constraint needs to match hierarchies of components in the architecture design.
- mapper: "FEATHER/LayoutLoop/configurations/mapper/"
- Layout: "FEATHER/LayoutLoop/configurations/layout/"

**Different on-chip reordering modeling methods** are activated by enabling different global macro

- Transpose: ENABLE\_TRANSPOSE
- Line Rotation: MEDUSA

By default, Layoutloop assumes no on-chip reordering.

2) Exp. Set 3: LayoutLoop Analytic Analysis: The provided Verilog implementation of FEATHER is a parameterized scalable template, which allows users to change the shape of FEATHER by modifying the input parameters at the top module "FEATHER/FEATHER\_RTL/RTL/feather\_top.v". Users could modify the following parameters into value from (4,8,16,32,64) to investigate the area and power of FEATHER at different scsales.

```
module feather_top #(
   parameter DPE_COL_NUM = 64,
   parameter DPE_ROW_NUM = 64,
   ...
```

