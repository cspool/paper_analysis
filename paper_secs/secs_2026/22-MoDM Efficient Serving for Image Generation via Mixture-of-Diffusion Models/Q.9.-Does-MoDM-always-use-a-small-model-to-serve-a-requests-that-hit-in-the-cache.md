# Q.9. Does MoDM always use a small model to serve a requests that hit in the cache?

Response: To maximize system throughput, MoDM defaults to using a small diffusion model for all requests hitting in the cache (throughput-optimized mode, §5.3). Additionally, MoDM offers flexibility for service providers to prioritize image quality by serving cache hits with a large model when request rates are low and SLO requirements allow (quality-optimized mode, §5.3). Fig. 10 illustrates this use-case, showing that when request rates drop below 10 per minute, cache hits can be served by the large model to maximize image quality without violating SLO. However, MoDM can also run in throughput-optimized mode at low request rates if preferred.

