# *A. Acceleration Structure*

Since Gaussian primitives are not natively supported, they are registered as procedural primitives in Vulkan. Unlike the icosahedron-approximated representation adopted in [27], where each Gaussian expands into twenty triangles with twelve vertices, our implementation adopts a one-to-one mapping between a Gaussian and a procedural leaf node in the BLAS (Fig. 4). By avoiding geometric expansion, the procedural representation preserves a compact acceleration structure with a significantly reduced memory footprint. As shown in Fig. 5, compared to the icosahedron counterpart, it reduces BVH size by 26× and accelerates BVH construction by 1.5×, owing to the elimination of redundant triangle primitives. The axis-aligned bounding box (AABB) of each Gaussian is conservatively defined based on its spatial scaling parameters, ensuring complete coverage of its volumetric span.

### *B. Memory Allocation*

We explicitly manage GPU memory to accommodate the Gaussian ray tracing. The corresponding buffers are registered as *Storage Buffer Objects* (SSBO) and bound to programmable shaders via descriptor bindings. The specific implementations are described below.

#### Algorithm 1 Ray-gen Shader

```
Require: pixel, camera, BVH, tMin, tMax, threshold
Ensure: Ray-Gen
 1: rayPayloadEXT Ray
 2: origin, direction = rayCast(pixel, camera)
 3: while Ray.trans > threshold and Ray.thit > 0 do
 4: traceRayEXT(BVH, origin, direction, tMin, tMax)
 5: origin = origin + Ray.thit × direction
 6: end while
 7: imageStore(pixel, Ray.color)
```

Gauss Param Buffer: Gaussian parameters, including position, covariance, opacity, and color, are stored in a compact layout. This buffer is bound to both the intersection and closest-hit shaders, allowing straightforward access to Gaussian attributes during ray-Gauss evaluation.

Closest Hit Buffer: Conventional ray tracing only records the closest mesh hit, while tracing through Gaussians can intersect and blend multiple primitives. To this end, we allocate a Closest Hit Buffer to store the closest-K hits per ray, organized as a 2D array of dimensions Nray × K. Each entry records three essential attributes: (1) hit distance thit, which serves as the sorting criterion to preserve the correct blending order with respect to the ray origin. (2) alpha value, representing the attenuation factor that quantifies the contribution of the intersected Gaussian to the ray's transmittance. (3) primitive index (GID), which provides a reference for the closest-hit shader to retrieve the Gaussian's texture attributes, thereby enabling color computation.

Hit Count Buffer: Vulkan provides the RayPayloadEXT API, allowing the closest-hit shader to access per-ray properties stored in the warp buffer, but this API is restricted in intersection and any-hit shaders. In Gaussian ray tracing, however, any-hit shaders need to update the number of hit Gaussians per ray, denoted as Nhit. To this end, we introduce a separate Hit Count Buffer, accessible for all shader types.

### *C. Programming Model*

As conventional practice, BVH traversal is handled by the RTA, including node visits and other fixed-function operations. While programmable shaders are executed on the SM cores, including the customized ray-Gauss intersection shader that is invoked by the RTA upon detecting candidate primitives. Under Vulkan-Sim's delayed scheduling strategy, intersection shader invocations are recorded into a task table during traversal rather than executed immediately. After all threads within a warp complete traversal, each thread sequentially processes its recorded intersection tasks. As a result, the execution latency of BVH traversal and shader execution can be decoupled for fine-grained analysis. To clarify the overall control flow of Gaussian ray tracing, we next walk through the core shaders in the baseline implementation.

Ray-Gen Shader (Alg. 1): We adopt a one-thread-perray mapping, where the ray-gen shader serves as the main kernel, and other shaders are invoked as function calls.

```
Algorithm 2 Any-hit Shader
```

```
Require: hitAttrib, rayID, HitCount, ClosestHit
Ensure: Any-Hit
 1: Nhit = HitCount[rayID]
 2: tmp = hitAttrib
 3: for entryID = 0 to Nhit do
 4: entry = &ClosestHit[rayID][entryID]
 5: if tmp.thit < entry.thit then
 6: Swap(tmp, entry)
 7: end if
 8: end for
 9: if Nhit < K then
10: ClosestHit[rayID][Nhit] = tmp
11: HitCount[rayID] = Nhit + 1
12: end if
```

Given the volumetric nature of Gaussian models, the ray-gen shader is organized as an iterative loop. In each iteration, the traceRayEXT function is invoked, and the loop terminates once the ray either misses all primitives or its transmittance falls below a predefined threshold. After each invocation, the ray origin marches along its direction by the hit distance thit reported by the closest-hit shader. This incremental marching strategy prevents revisiting previous Gaussians, ensuring correctness and efficiency.

Any-Hit Shader (Alg. 2): Integrated within the intersection shader, once a Gaussian hit is confirmed, the any-hit shader performs a comparison-based insertion to maintain the Closest Hit Buffer in ascending order of hit distance. If the Closest Hit Buffer is not yet full, the new hit is inserted; otherwise, it is discarded when its distance exceeds the farthest record. This process ensures that only the closest-K hits are retained for subsequent blending in the closest-hit shader.

Closest-Hit Shader (Alg. 3): The closest-hit shader reads the ray's Nhit from the Hit Count Buffer to identify the effective hits inside the Closest Hit Buffer. It then traverses these entries, blending their colors weighted by alpha to update the ray's accumulated color and transmittance. Finally, the

```
Algorithm 3 Closest-hit Shader
```

```
Require: rayID, GaussParam, HitCount, ClosestHit
Ensure: Closest-Hit
 1: rayPayloadEXT Ray
 2: Nhit = HitCount[rayID]
 3: T, C = &Ray.trans, &Ray.color
 4: for entryID = 0 to Nhit do
 5: entry = ClosestHit[rayID][entryID]
 6: alpha = entry.alpha
 7: color = GaussParam[entry.GID].color
 8: C += T × alpha × color
 9: T *= 1 - alpha
10: end for
11: Ray.thit = ClosestHit[rayID][Nhit-1].thit
12: HitCount[rayID] = 0
```

![](_page_5_Figure_0.jpeg)

Fig. 6: Profiling of Gaussian ray tracing. (a) Comparison of latency between BVH traversal and shader execution. (b) Breakdown of shader instructions. (c) Comparison of memory access between BVH traversal and shader execution.

shader updates the ray's hit distance thit with the farthest hit recorded in the buffer, marking the maximum distance reached in the current traceRayEXT round.

#### *D. Inefficiency Analysis*

To identify performance bottlenecks in Gaussian ray tracing, we evaluate the proposed pipeline using Vulkan-Sim [44], a cycle-accurate RTA simulator that supports the Vulkan API. The simulator enables fine-grained architectural profiling of shader execution and BVH traversal, which is difficult to isolate on commercial devices due to tightly interleaved SM and RTA activities. Experiments are conducted on eight scenes from the NeRF-Synthetic dataset [8]. Notably, BVH traversal's memory access can benefit from established optimizations such as ray-coherent scheduling [45] and treelet prefetching [46]–[48], both of which are essential for highperformance RTAs. We incorporate these techniques as part of the baseline and will detail our enhancement in Sec. IV-D.

Fig. 6(a) presents the normalized per-ray latency, decomposed into BVH traversal (on RT cores) and software shader execution (on SM cores), which are serialized in Vulkan-Sim. For clarity, traversal latencies with and without treelet optimizations are shown separately, revealing the shifting performance bottleneck under the two configurations. Without treelet optimizations, BVH traversal latency is on average 1.3× higher than shader execution. In contrast, with treelet prefetching enabled (size of 16KB with ping-pong scheme), the traversal is substantially accelerated and shader execution becomes the dominant contributor, accounting for 72.9% of the total ray-tracing latency.

We further profile the shader's instruction and memory traffic. Fig. 6(b) breaks down the instructions into three categories: ALU operations, memory accesses, and other (e.g., control operations, shader calls, and special functions). Among

![](_page_5_Figure_7.jpeg)

Fig. 7: Comparison of ray tracing latency between icosahedron agent and procedural primitive.

them, ALU operations dominate, accounting for an average of 80.3%, 8.2× higher than memory accesses. This highlights the compute-intensive nature of the intersection shader, primarily stemming from the complex mathematical operations required by ray-Gauss evaluation. Fig. 6(c) compares the amount of memory accesses between BVH node fetches handled by the RTA and global memory accesses initiated by shader cores. Shader-side accesses for Gaussian parameter loading and hitbuffer management, contribute ∼ 21% of the traffic relative to BVH traversal, presenting a considerable overhead.

To validate that our simulator-based observations reflect practical behavior, we further profile an NVIDIA RTX 3090 using an NVIDIA OptiX deployment. Fig. 7 compares the endto-end latency of an icosahedron-based representation, which leverages hardware ray–triangle intersection units, with that of a procedural Gaussian primitive executed via SM shaders. The latter exhibits a 1.6× higher latency, corresponding to a shader execution overhead of 38%. This trend is consistent with our Vulkan-Sim profiling results, which report a 43% latency contribution in the vanilla-BVH configuration. The 5% discrepancy is likely attributable to increased BVH traversal overhead in the icosahedron case due to its expanded hierarchy.

Our profiling indicates that Gaussian ray tracing is limited by the software shader execution. The heavy cost of procedural-leaf evaluation and shader-side memory traffic erodes the benefits of hardware-accelerated BVH traversal, highlighting the need for dedicated architectural support for ray-Gauss evaluation and hit-buffer management.

