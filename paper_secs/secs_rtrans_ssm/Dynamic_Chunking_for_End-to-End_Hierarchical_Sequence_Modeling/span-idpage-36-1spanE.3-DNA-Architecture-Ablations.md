# <span id="page-36-1"></span>E.3 DNA Architecture Ablations

H-Net (1-stage) with an M3T1 encoder achieves 3.6× the data efficiency of an isotropic architecture (Figure [6\)](#page-16-2). As mentioned in the caption of Table [5,](#page-16-2) we found that an M3T1 encoder outperformed a pure Mamba-2 M4 encoder (Table [7\)](#page-36-2). Putting a Transformer in the encoder network does not appear to be helpful for text (Figure [8\)](#page-17-1). Thus, it is possible the Transformer being useful is a DNA-specific result.

Interestingly, the loss curve for the M4 encoder with a pure Mamba-2 main network was more unstable. We then also tried replacing the M15 in the main network with a T1M13T1 architecture, inspired by the finding that Transformer layers are good for dealing directly with compressed input (see Figure [10\)](#page-18-0). The new, principled main network architecture improved stability greatly (Figure [14\)](#page-36-2).

