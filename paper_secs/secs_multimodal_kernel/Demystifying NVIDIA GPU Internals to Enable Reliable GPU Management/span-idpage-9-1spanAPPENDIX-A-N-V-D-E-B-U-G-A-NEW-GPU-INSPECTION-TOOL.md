# <span id="page-9-1"></span>APPENDIX A N V D E B U G: A NEW GPU INSPECTION TOOL

The hazards we have identified in prior work highlight the need for greater visibility into the scheduling mechanisms of NVIDIA GPUs. In order to enable other researchers to continue our work, and to verify that future management approaches works as expected, we are open-sourcing our

```
1 user@machine:~$ cat /proc/gpu1/runlist0
2 +---- TSG Entry 1 ---+
3 | Scale: 3 |
4 | Timeout: 128 |
5 | Length: 1 |
6 +---------------------+
7 +- Channel Info 1 -+
8 | Enabled: 1|
9 | Next: 0|
10 | Force CTX Reload: 0|
11 | Enable set: 0|
12 | Enable clear: 0|
13 | PBDMA Faulted: 0|
14 | ENG Faulted: 0|
15 | Status: 0|
16 | Busy: 0|
17 ...
```

Listing 1. Example usage of nvdebug runlist information API.

nvdebug tool. Our tool allows for transparently inspecting and modifying GPU scheduling state, irrespective of the GPU driver in use.

nvdebug is a loadable Linux kernel module, and interacts directly with the GPU via memory-mapped I/O operations. This bypasses the GPU driver, so it works no matter what driver, if any, is installed. nvdebug supports both aarch64 and x86\_64 CPUs, works on both integrated and discrete NVIDIA GPUs, has no dependencies, and requires no configuration. We have tested it on GPUs from Kepler (2011) through Ada (2022).

After being loaded into the kernel, our tool exposes GPU information via a series of virtual files in /proc/gpuX for each GPU X on the system. Key stable APIs include:

- 1) gpuX/device\_info: Print information about GPU engines, including their associated runlist IDs.
- 2) gpuX/runlistY: Print the contents of runlist Y.
- 3) gpuX/disable\_channel: On write, disable the channel associated with the ID written.
- 4) gpuX/enable\_channel: On write, enable the channel associated with the ID written.
- 5) gpux/lce\_for\_pceY: Read which LCE PCE Y is mapped to.
- 6) gpux/shared\_lce\_for\_grceY: Read which LCE (if any) GRCE Y is mapped to.
- 7) gpux/pce\_map: Read a bit mask of which PCEs are available.

We include an example of using nvdebug to print Runlist 0 for the second GPU in the system ,in Listing [1.](#page-10-18)

Among the challenges we overcome while constructing this tool, accessing, parsing, and traversing GPU page tables to access runlist entries in GPU physical memory proved particularly difficult. We encourage those further interested in this topic to consult our code.[17](#page-10-19)

