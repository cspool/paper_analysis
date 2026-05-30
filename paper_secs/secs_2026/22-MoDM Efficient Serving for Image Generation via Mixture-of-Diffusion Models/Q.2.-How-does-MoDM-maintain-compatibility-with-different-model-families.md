# Q.2. How does MoDM maintain compatibility with different model families?

**Response:** This is made possible by our caching strategy, which caches *final generated images* rather than intermediate features. Final images are more versatile and recognizable across different models and model families. §5.5 explores model-agnostic caching for serving across multiple model families.

#### O.3. What does MoDM cache?

**Response:** We cache *final generated images*.

