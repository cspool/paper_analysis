# <span id="page-8-0"></span>6 ISA-specific Test Oracles (TAIDL-TOs)

Figure [14](#page-8-1) shows the architect's and programmer's views of TAIDL and the generated test oracle TAIDL-TO. A computer architect only needs to define the ISA using TAIDL. This will automatically generate an ISA-specific test oracle TAIDL-TO which can be provided to kernel programmers. A kernel programmer can test the correctness of their low-level kernels using this generated test oracle.

