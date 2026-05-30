# <span id="page-15-0"></span>A Failure Propagation in Monolithic and Decoupled Deployments

### A.1 Case study I: monolithic deployment mode

Fig. 3 (a) presents how a single worker's failure propagates in a monolithic deployment. When *Worker 1* fails during decoding, the collective communicator (CCL) is torn down after a timeout. Importantly, the CCL treats the set of workers as a static communication group [46]: if one worker becomes unavailable, the entire communicator aborts. Consequently, all workers are killed and restarted and the CCL is re-initialized. Then all workers must replay all L prefill layers, all decoding layers for the (i-1) tokens that have already been produced, and finally the first  $\ell$  layers for the current token have to be re-executed. All previously accumulated KV caches and partial outputs—even on healthy workers—are discarded. This results in a long, user-visible inference stall and substantial recomputation overhead.

