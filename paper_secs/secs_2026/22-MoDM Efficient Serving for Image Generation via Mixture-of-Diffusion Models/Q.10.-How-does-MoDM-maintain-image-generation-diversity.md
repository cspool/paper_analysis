# Q.10. How does MoDM maintain image generation diversity?

**Response:** While MoDM generates images that hit in the cache by reusing previously generated outputs—through controlled noise injection and partial denoising—a key design choice is the adoption of a FIFO-based caching strategy. Unlike utility-based policies (*e.g.*, those inspired by CPU hardware caches), the FIFO-based approach ensures automatic eviction of cached images after a fixed time window. This prevents a small set of highly popular cached images from dominating reuse, thereby encouraging diversity in the

cache and maintaining adaptability to evolving input distributions. A quantitative evaluation of generation diversity is a compelling direction for future work.

