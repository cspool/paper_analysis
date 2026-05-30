# IV. ARCHITECTURE EVALUATIONS

#### *A. Simulation Environment of Architecture Evaluations*

CamPU is designed in a gate-level synthesis by the Synopsys Design Compiler with 28 nm technology and 500 MHz of clock frequency. Its power is measured through the Synopsys PrimeTime PX. CamPU occupies 0.54 mm<sup>2</sup> of area and consumes 12.9 mW of power on average. The CamPU architecture is evaluated through RTL-level simulation running multi-camera systems that have 80-120◦ field-of-view (FoV) of 256×256 sized camera images and stitching their DNN outputs on a spherical coordinate. The baseline architecture consists of the in-order image projection unit and blending unit; the baseline's image projection unit adopts cache memory with the in-order memory load execution, and its blending unit processes full-sized intermediate spherical images without overlap-aware rectangular image blending.

