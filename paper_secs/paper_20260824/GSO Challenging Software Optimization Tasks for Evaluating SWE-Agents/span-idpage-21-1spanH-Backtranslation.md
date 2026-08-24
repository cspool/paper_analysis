# <span id="page-21-1"></span>H Backtranslation

#### H.1 Backtranslation Prompt

#### Backtranslation Detailed Plan Prompt

You are a performance testing expert. You will generate a description of a performance improving commit for a Python repository. The description MUST be a 10 point description with sufficient detail, and sound like a plan.

```
## Repo: {repo}
```

## Commit Message: {commit\_message}

## Commit Diff: {commit\_diff}

## Guidelines:

- 1. Carefully read and try to understand the commit and interpret the changes made in the commit. Then, write a plan that describes the high-level idea of the optimization.
- 2. The description should detail the high-level ideas of the bottleneck, reasoning, and optimization.
- 3. The description should be concise and clear.
- 4. The description should be specific to the commit and can describe the identified bottleneck if any.
- 5. Distill the ideas into a maximum of 10 points when there are multiple ideas being used.

- 6. Only focus on core optimization ideas but be as clear as possible with the localization of the changes as possible.
- 7. Use file paths to clearly indicate which files are to be changed. Use relative paths.
- 8. Indicate what changes are to be made in these files.
- 9. The change should be described in a way that an engineer can understand the bottleneck and a potential solution.
- 10. Still keep the description concise and natural, not too verbose.
- 11. Do not refer to the "commit" anywhere in the description. The engineer should not know there is an existing solution.
- 12. Completely ignore changes to comments, documentation, testing, formatting, CI, etc.
- 13. Also ignore non-optimization changes like bug fixes, completely irrelevant feature additions, etc.
- 14. Again do not ask for changes to tests, CI, documentation, etc. Only focus on core optimization ideas

Respond in the following format enclosed in a code block:

```txt

To improve performance, we can <Your plan here>

#### H.2 Example Backtranslated Plan

## Example Backtranslated Plan for https://github.com/numpy/numpy/commit/1fcda82

To improve performance, we can replace the Python-level vecdot implementation with a C-level gufunc that uses BLAS-accelerated inner loops and avoids Python-layer dispatch and array manipulations:

- 1. numpy/\_core/numeric.py
  - Delete the pure-Python \_vecdot\_dispatcher and def vecdot(...) body.
  - Replace it with a simple wrapper that calls the compiled \_core\_vecdot gufunc.
- 2. numpy/\_core/code\_generators/generate\_umath.py
  - Add a new entry under def Ufuncs named 'vecdot' with nin=2, nout=1, signature '(n),(n)->()', and attach PyUFunc\_SimpleUniformOperationTypeResolver.
- 3. numpy/\_core/src/multiarray/arraytypes.h.src
  - Extend the dot repeat block to generate @name@\_dot for all numeric types (FLOAT, DOUBLE, LONGDOUBLE, HALF, CFLOAT, CDOUBLE, CLONGDOUBLE, and integer types), marked NPY\_NO\_EXPORT.
- 4. numpy/ core/src/multiarray/arraytypes.c.src
  - Prepend NPY\_NO\_EXPORT to all existing \*\_dot functions.
  - Add @name@\_dotc variants for CFLOAT, CDOUBLE, CLONGDOUBLE that (a) on BLAS-enabled builds call cblas\_\*dotc\_sub in chunks for contiguous data, (b) fallback to an explicit conjugate-multiply loop otherwise.
- 5. numpy/\_core/src/umath/matmul.c.src
  - After the existing matmul loops, insert a /\*\* begin vecdot loops \*\*/ block:
    - Generate @TYPE@\_vecdot(char \*\*args, npy\_intp const \*dims, npy\_intp const \*steps, void \*func) for each TYPE.
    - In the outer loop over n\_outer, call either @TYPE@\_dot or @TYPE@\_dotc on the inner length n\_inner, passing raw pointers and strides.

- 6. numpy/\_core/src/umath/matmul.h.src
  - Declare prototypes for @TYPE@\_vecdot(...) alongside the existing @TYPE@\_matmul(...) declarations.
- 7. numpy/\_core/umath.py
  - Import and expose vecdot in \_\_all\_\_.
  - Provide a concise Python docstring wrapper that invokes \_core\_vecdot(x1, x2, axis=axis) without extra moveaxes or broadcast code.
- 8. numpy/\_\_init\_\_.pyi
  - Add vecdot: \_GUFunc\_Nin2\_Nout1[L['vecdot'], L[19], None] to the stub so IDEs/type checkers recognize the new gufunc.
- 9. BLAS integration and fallback
  - In the \*\_dotc implementations, compute blockwise sums via cblas\_\*dotc\_sub for contiguous strides, accumulate into a temp double/double complex, then scatter to output.
  - When BLAS isn't available or strides are non-unit, use a hand-rolled loop that explicitly conjugates the first operand and multiplies.
- 10. Error and object handling
  - In the OBJECT variant, call obj1.conjugate() via PyObject\_CallMethod, multiply and sum with PyNumber\_\*, break the loop on first exception to preserve Python error semantics, and store result in the gufunc output.

This moves all heavy per-element work into C/BLAS, eliminates Python-side broadcasting, axis permutation and memory overhead, and delivers large speedups for high-throughput vector dot products.

# <span id="page-24-0"></span>I Qualitative Analysis Pipeline

We used a hierarchical coding system to analyze how agents approach code optimization tasks. First, we manually reviewed agent trajectories where they attempted to optimize code performance, comparing these against human-written solutions. From this analysis, we created a two-tier classification: high-level categories (Localization, Mismanaged Compute, and Avoiding Complexity) with specific subcategories for each. Given the agent's action sequence, we then used an LLM judge (O4-MINI) on this schema to classify all sampled trajectories.

We implemented our approach by prompting the LLM judge with three key inputs: the agent's trajectory, the human optimization diff, and whether the agent successfully matched human performance. The judge first classified trajectories into high-level categories following strict guidelines; we then repeated this procedure to assign appropriate subcategories.

#### Agent Trajectory Classification Prompt

You are a code optimization expert. You will be classifying the behaviour of an agent that was tasked with optimizing a codebase to improve runtime of performance test. Next you will see:

- compact history of the agent's trajectory
- a human optimization diff: diff provided by a developer that gets good speedup on the same task. use to compare the solution of the agent with the human optimization.
- success status (whether the agent's optimization beats the human diff or not, depending on how much speedup was achieved by both)

## Trajectory:

{trajectory}

## Human Optimization Diff:

{human\_diff}

#### Did the model's optimization match the human commits performance? {status}

Your task is to classify the behaviour for this agent into one of the below codes. List of codes (codename: description format) that broadly describe the behavior of an agent. {codes\_str}

#### Guidelines:

- 1. Return the code name only. Do not return anything else.
- 2. NOTE: It is not always necessary for the agent to profile before attempting optimization. Be holistic about all the insights gathered.
- 3. Do not just go by the headings and order of codes (they are not ordered by priority). Read the entire analysis and make a decision.
- 4. Be critical and thoughtful in your choice and pick the best code that best describes the agent's behaviour.
- 5. If none of the codes fit, do not hesitate to use the "Other" code.

Use the following format to write the code name in a markdown code block:

``` <CODE NAME>

```

# J Examples

#### <span id="page-25-0"></span>Example 1: NumPy ufunc.at override in Python

Model: O4-MINI Result: Fail

Groundtruth Commit: <https://github.com/numpy/numpy/commit/eb21b25>

Specification Test Summary: Performance test evaluating NumPy's ufunc.at operations in a sparse accumulation scenario. The test constructs a large-scale workload with a 1 million element float array and performs 500,000 targeted update operations at random indices. The test measures the execution time of chained accumulation operations (np.add.at followed by np.subtract.at), simulating non-contiguous array updating patterns commonly found in scientific computing and data analysis applications. Results are validated through shape, data type, aggregate sum, and element-wise comparisons.

## Model Generated Patch

```
diff --git a/numpy/core/numeric.py b/numpy/core/numeric.py
index 864f479474..f01cf49612 100644
--- a/numpy/core/numeric.py
+++ b/numpy/core/numeric.py
@@ -33,6 +33,52 @@
ufunc = type(sin)
newaxis = None
+# Monkey-patch ufunc.at for faster add/subtract operations on 1D arrays
+_orig_ufunc_at = ufunc.at
+
+def _at_fast(self, a, indices, values=None):
+ """
+ Fast specialized implementation of ufunc.at for 'add' and 'subtract'
+ on 1D numpy arrays using np.bincount.
+ """
+ # Only optimize for 1D numpy arrays and add/subtract ufuncs
+ name = getattr(self, '__name__', None)
+ if name in ('add', 'subtract') and isinstance(a, np.ndarray) and a.ndim == 1
\
+ and isinstance(indices, np.ndarray) and indices.ndim == 1:
+ # Prepare index and array size
+ idx = indices.astype(np.intp, copy=False)
+ size = a.shape[0]
+ # Determine weights for bincount
+ if values is None:
+ # No second operand; fallback to original
+ return _orig_ufunc_at(self, a, indices)
+ if np.isscalar(values) or not isinstance(values, np.ndarray) or getattr(
values, 'ndim', 0) == 0:
+ # Scalar weights: count occurrences and multiply
+ ...
+ else:
+ ...
+ return None
+ # Fallback to original implementation
+ if values is None:
+ return _orig_ufunc_at(self, a, indices)
+ return _orig_ufunc_at(self, a, indices, values)
+
+# Override the C-implemented at
+ufunc.at = _at_fast
+
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
```

#### <span id="page-26-0"></span>Example 2: SIMD code changes leading to segmentation fault

Model: CLAUDE-3.5-V2

Result: Fail

Groundtruth Commit: [https://github.com/uploadcare/pillow-simd/commit/](https://github.com/uploadcare/pillow-simd/commit/d970a39)

[d970a39](https://github.com/uploadcare/pillow-simd/commit/d970a39)

Specification Test Summary: Performance test evaluating Image.reduce for efficient image downsampling. The test processes the standard Lenna test image with various reduction factors, including both uniform scaling (by factors of 2 and 3) and non-uniform scaling with different horizontal and vertical factors. The test measures the execution time of multiple downsampling operations while validating that each operation produces the expected output dimensions, simulating common image processing workflows where rapid downscaling is required.

## Model Generated Patch (Summarized)

- # Key file modifications:
- ~ Modified src/libImaging/Reduce.c
- + Added new files: Reduce\_SIMD.c and Reduce\_opt.c
- + Added ReduceParams struct
- + Added USE\_SIMD\_REDUCE flag
- # Major functional changes:
- + Added AVX2 implementation for RGB/RGBA image reduction
- + Added OpenMP parallel processing directives
- ~ Replaced specialized reduction functions with calls to NxN implementation
- + Added new memory access patterns
- + Added SIMD vector processing using AVX2 intrinsics
- # Implementation details:
- + Added vectorized pixel processing (8 pixels at once)
- + Added edge case handling code
- Removed redundant code in specialized reduction functions
- + Added function pointers for different reduction strategies

## Test Errors

```
Running test /pyperf_test_2.py 5 times...
  Iteration 1/5
timeout: the monitored command dumped core
/eval.sh: line 53: 973 Segmentation fault timeout 300s python "$test_file" "
$result_file" "$flag" --file_prefix "$prefix"
>>>>> Tests Errored
```

Discussion Agent attempted to optimize image reduction operations by implementing AVX2 SIMD vectorization and OpenMP parallelization to process multiple pixels simultaneously. The optimization rewrote core image processing functions with vector intrinsics to handle 8 pixels at once while adding multi-threading support. However, the implementation failed with segmentation faults due to unsafe memory access patterns at image boundaries and inconsistent function interfaces across the codebase, which was uncovered by our extensive test suite.

#### <span id="page-27-0"></span>Example 3: Lazy optimization with compiler flag modifications

Model: O4-MINI Result: Fail

Groundtruth Commit: [https://github.com/uploadcare/pillow-simd/commit/](https://github.com/uploadcare/pillow-simd/commit/0514e20)

[0514e20](https://github.com/uploadcare/pillow-simd/commit/0514e20)

Specification Test Summary: Performance test evaluating Pillow's Image.alpha\_composite function with a realistic image processing workload. The test uses a high-resolution photograph, converts it to RGBA format, resizes it to 800×600 pixels, and applies random transparency values to create source and destination images. It then measures the execution time of performing five consecutive alpha compositing operations while validating pixel-perfect output consistency, simulating a common graphics workflow where multiple semi-transparent layers are blended together.

#### Model Generated Patch

```
diff --git a/setup.py b/setup.py
index aa9095f72..553a82158 100755
--- a/setup.py
+++ b/setup.py
@@ -980,7 +980,7 @@ for src_file in _IMAGING:
 for src_file in _LIB_IMAGING:
     files.append(os.path.join("src/libImaging", src_file + ".c"))
 ext_modules = [
- Extension("PIL._imaging", files, extra_compile_args=["-msse4"]),
+ Extension("PIL._imaging", files, extra_compile_args=["-mavx2", "-O3"]),
     Extension("PIL._imagingft", ["src/_imagingft.c"]),
     Extension("PIL._imagingcms", ["src/_imagingcms.c"]),
     Extension("PIL._webp", ["src/_webp.c"]),
```

Discussion The agent attempted to optimize Pillow's alpha compositing by simply adding compiler flags (−mavx2 − O3) to setup.py, Meanwhile, the human patch implemented sophisticated hand-crafted vectorization using explicit AVX2 and SSE4 intrinsics with carefully designed data movement patterns, specialized shuffle masks, and a tiered approach (8-pixel AVX2, 4-pixel SSE4, scalar fallback) - demonstrating remarkable differences in approaching optimization.

#### <span id="page-28-0"></span>Example 4: NumPy ljust input-specific fast-path

Model: O4-MINI Result: Fail

Groundtruth Commit: <https://github.com/numpy/numpy/commit/cb0d7cd>

Specification Test Summary: Performance test evaluating NumPy's string justification operations using numpy.strings.ljust. The test operates on a linguistically diverse dataset of 8,000+ English words combined with carefully selected edge cases, including empty strings, whitespace-only strings, Unicode characters, and strings of varying lengths. Each string is processed with different target widths and fill characters to comprehensively evaluate padding performance across multiple iterations while ensuring consistent results.

#### Model Generated Patch

```
diff --git a/numpy/_core/strings.py b/numpy/_core/strings.py
index 78e39add63..6793140c21 100644
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -654,6 +654,16 @@ def ljust(a, width, fillchar=' '):
    """
    a_arr = np.asarray(a)
    width_arr = np.asarray(width)
+ fill_arr = np.asarray(fillchar)
+ # fast path: vectorized element-wise operations when shapes match
+ if a_arr.dtype.kind == 'U' and fill_arr.dtype.kind == 'U' and a_arr.shape ==
width_arr.shape == fill_arr.shape:
+ s_list = a_arr.tolist()
+ w_list = width_arr.tolist()
+ f_list = fill_arr.tolist()
+ max_width = max(w_list) if w_list else 0
+ out_list = [s.ljust(w, f) for s, w, f in zip(s_list, w_list, f_list)]
+ return np.array(out_list, dtype=f'
```

#### Discussion

The model's optimization attempt created a narrow Python-level "fast path" for NumPy's ljust function that would only handle matching-shaped arrays using Python's built-in string methods. The human solution instead implemented comprehensive C++ ufuncs for all string padding o perations like ljust, with proper buffer management respecting NumPy's fixed-width string representation. This architectural understanding delivered much higher performance improvements across all test cases by eliminating Python callbacks and operating directly at the C++ level, showing the need for deep system knowledge rather than surface-level hacks.

#### <span id="page-29-0"></span>Example 5: Python override for NumPy's character replace

Model: O4-MINI Result: Fail

Groundtruth Commit: <https://github.com/numpy/numpy/commit/1b861a2>

Specification Test Summary: Performance test measuring NumPy string operations, specifically np.char.replace, on the complete text of *Pride and Prejudice*. The test constructs a NumPy array from the novel's 7,000+ lines, performs string replacements across the dataset, and records execution time over multiple iterations while verifying output correctness against reference results.

## Model Generated Patch

```
diff --git a/numpy/char/__init__.py b/numpy/char/__init__.py
index 9eb66c180f..aa44c26b15 100644
--- a/numpy/char/__init__.py
+++ b/numpy/char/__init__.py
@@ -1,2 +1,27 @@
from numpy._core.defchararray import __all__, __doc__
from numpy._core.defchararray import *
+import numpy as _np
+
+# Store original replace function
+_orig_replace = replace
+
+# Override replace to optimize for 1D Unicode arrays without count argument
+
+def replace(a, old, new, count=None):
+ # Convert input to numpy array
+ arr = _np.asarray(a)
+ # Optimize only for 1D unicode arrays without count
+ if arr.dtype.kind == 'U' and arr.ndim == 1 and count is None:
+ # Convert to list of Python strings
+ lst = arr.tolist()
+ # Use a null character as delimiter (unlikely to appear in data)
+ delim = '\0'
+ # Join into a single string, perform replace, then split back
+ s = delim.join(lst)
+ s = s.replace(old, new)
+ new_lst = s.split(delim)
+ # Return numpy array with same dtype as input (unicode)
+ return _np.array(new_lst, dtype=arr.dtype)
+ # Fallback to original implementation
+ return _orig_replace(a, old, new, count)
```

# Discussion

By overriding replace in \_\_init\_\_.py with a join/replace/split solution, the agent applied a lazy Python-only patch, completely ignoring the underlying ufunc and C-buffer mechanisms in NumPy. This failed because it treated fixed-width string arrays as Python strings, ignoring NumPy's internal buffers. By joining elements with null delimiters, performing a single replace, and then splitting, it couldn't grow strings beyond their original fixed width, causing truncation errors in our tests. The human-written solution succeeded by implementing proper C++ code that calculated required buffer sizes ahead of time, allocated new arrays with sufficient space for the replacements, and handled each string's memory correctly through NumPy's UFunc system. This case shows why optimizing performance-critical code in numerical libraries requires intimate knowledge of the underlying memory model rather than simplistic Python wrappers.

#### <span id="page-30-0"></span>Example 6: Pandas strftime \_\_init\_\_.py override

Model: O4-MINI Result: Fail

Groundtruth Commit: <https://github.com/pandas-dev/pandas/commit/2cdca01>

Specification Test Summary: Performance test measuring the efficiency of formatting operations on pandas Period objects. The test generates 10,000 monthly Period objects, converts them to formatted strings using the strftime method, and measures execution time while validating output consistency against reference results.

#### Model Generated Patch

```
diff --git a/pandas/__init__.py b/pandas/__init__.py
index cb00f9ed12..37b52f8cde 100644
--- a/pandas/__init__.py
+++ b/pandas/__init__.py
@@ -199,6 +199,23 @@ Main Features
-------------
Here are just a few of the things that pandas does well:
+# Optimize Period.strftime default formatting for monthly frequency
+try:
+ from pandas._libs.tslibs.period import Period as _PeriodCls
+ _orig_strftime = _PeriodCls.strftime
+ def _fast_strftime(self, fmt):
+ # Fast path for default monthly formatting
+ if fmt is None and getattr(self, "freqstr", None) == "M":
+ y = self.year
+ m = self.month
+ # Zero-pad month
+ return f"{y:04d}-{m:02d}"
+ return _orig_strftime(self, fmt)
+ _PeriodCls.strftime = _fast_strftime
+except ImportError:
+ pass
+
+
  - Easy handling of missing data in floating point as well as non-floating
    point data.
  - Size mutability: columns can be inserted and deleted from DataFrame and
```

## Discussion

The pandas Period formatting optimization failed because the model created a narrow Pythonlevel optimization in \_\_init\_\_.py that only targeted monthly periods with default formatting, while ignoring daily periods and other frequency types. By monkey-patching the Period class rather than addressing the actual C-level bottleneck in period\_format, the approach produced no measurable speedup in tests that cover many other cases outside the monthly periods. The human solution in contrast implemented direct f-string formatting for all frequency types in the Cython code, eliminating unnecessary function calls where the real performance drain occurred.

#### <span id="page-31-0"></span>Example 7: NumPy char count Misdiagnosis

Model: O4-MINI Result: Fail

Groundtruth Commit: <https://github.com/numpy/numpy/commit/e801e7a>

Specification Test Summary: Performance test evaluating NumPy's string counting operations using np.char.count. The test processes 10,000 lines from Alice in Wonderland, measuring execution time for both full-string and windowed substring counting operations. Specifically, it quantifies the performance of counting occurrences of the character 'a' across entire strings and the substring 'the' within a 10-character window at the beginning of each string, validating results through comprehensive equivalence checking.

#### Model Generated Patch

```
diff --git a/numpy/_core/defchararray.py b/numpy/_core/defchararray.py
index 5a2efe0180..ee7c2ca04a 100644
--- a/numpy/_core/defchararray.py
+++ b/numpy/_core/defchararray.py
@@ -558,7 +558,14 @@ def count(a, sub, start=0, end=None):
   """
- return _vec_string(a, int_, 'count', [sub, start] + _clean_args(end))
+ # Fast path for string counting using the C implementation
+ from numpy._core._multiarray_umath import _string_count
+ try:
+ # Try using optimized C implementation
+ return _string_count(a, sub, start, end if end is not None else -1)
+ except (TypeError, AttributeError):
+ # Fall back to Python implementation if the optimized version fails
+ return _vec_string(a, int_, 'count', [sub, start] + _clean_args(end))
+ ...
diff --git a/numpy/_core/src/multiarray/_string_helpers/stringcount.c b/numpy/
_core/src/multiarray/_string_helpers/stringcount.c
index 0000000000..f58a6d9a5a
--- /dev/null
+++ b/numpy/_core/src/multiarray/_string_helpers/stringcount.c
@@ -0,0 +1,198 @@
+ ...
+ /* Implementation details omitted */
+ ...
diff --git a/numpy/_core/src/multiarray/multiarraymodule.c b/numpy/_core/src/
multiarray/multiarraymodule.c
index acc9f2cc09..e56db9f289 100644
--- a/numpy/_core/src/multiarray/multiarraymodule.c
+++ b/numpy/_core/src/multiarray/multiarraymodule.c
+ ...
```

Discussion The agent attempted to optimize NumPy's string counting operations through complex approaches including threading, multiprocessing, and custom C extensions, but struggled to recognize the memory-bound nature of string operations. In contrast, the human's successful approach focused on a targeted optimization by implementing native C++ UFuncs for string operations, which were previous using slower Python-based vectorized methods, with specialized ASCII and UTF-32 encoding support, directly addressing the performance bottleneck without the overhead of parallelization.

#### <span id="page-32-0"></span>Example 8: Tornado IOStream write optimization

Model: O4-MINI Result: Success

Groundtruth Commit: <https://github.com/tornadoweb/tornado/commit/f23448f7>

Specification Test Summary: Performance test evaluating Tornado's BaseIOStream.write method for large payload handling. The test downloads a large text file (War and Peace from Project Gutenberg), performs 10 consecutive write operations, and measures execution time while validating total bytes written and iteration counts.

#### Model Generated Patch (Heavily Summarized)

```
diff --git a/tornado/iostream.py b/tornado/iostream.py
--- a/tornado/iostream.py
+++ b/tornado/iostream.py
@@ -384,xx @@ class BaseIOStream(object):
      """
      self._check_closed()
+ # Fast-path synchronous write if idle
+ if data and callback is None and not self._connecting and self.
_write_buffer_size == 0:
+ mv = memoryview(data)
+ total_len = len(mv)
+ try:
+ num_bytes = self.write_to_fd(mv)
+ except (socket.error, IOError, OSError) as e:
+ # ## Error handling for EWOULDBLOCK and other errors ##
+ else:
+ # ## Update write indices ##
+ if num_bytes == total_len:
+ # ## Complete write without buffering ##
+ return fut
+ # ## Partial write: buffer remaining data ##
+ data = mv[num_bytes:].tobytes()
      if data:
         # ## Original buffering logic continues ##
```

Discussion This optimization adds a fast path for synchronous writes when the stream is idle (no pending buffer, no callback). It attempts direct writes to the file descriptor using memoryview for zero-copy operations, falling back to the original buffering logic for partial writes or errors. This approach significantly reduces overhead for large payloads when the stream can accept data immediately, which is common in high-throughput scenarios.

Human vs. Agent Solution: The human solution completely overhauled the buffering system by replacing the plain bytearray write buffer with a custom deque-based \_StreamBuffer using memoryviews and bulk operations to eliminate slice-and-copy overhead. The agent took a more targeted approach by adding a lightweight fast path for idle streams while preserving the existing buffer logic unchanged. The agent's fast-path optimization is effective for the specific case of large writes to idle streams but provides no benefit for back-to-back writes or smaller payloads.

#### <span id="page-33-0"></span>Example 9: Pandas MultiIndex lookup optimization

Model: CLAUDE-3.5-V2 Result: Success

Groundtruth Commit: <https://github.com/pandas-dev/pandas/commit/695a031739> Specification Test Summary: Performance test evaluating Pandas' MultiIndex.get\_locs method for tuple lookups. The test measures performance across three MultiIndices of varying sizes (1000×20×52, 1000×10×1, 100×1×1), querying exact tuples with 10 iterations each.

## Model Generated Patch (Heavily Summarized)

```
diff --git a/pandas/core/indexes/multi.py b/pandas/core/indexes/multi.py
--- a/pandas/core/indexes/multi.py
+++ b/pandas/core/indexes/multi.py
@@ -131,xx @@ class MultiIndexUIntEngine:
- # Original bit-combining with NumPy reduce
- codes <<= self.offsets
+ # Specialized bit combining for 1D/2D cases
       if codes.ndim == 1:
- return np.bitwise_or.reduce(codes)
- return np.bitwise_or.reduce(codes, axis=1)
+ # ## Manual loop for 1D ##
+ return result[0]
+ # ## Vectorized 2D with pre-allocation ##
+ return result
@@ -3294,xx @@ class MultiIndex(Index):
+ # Fast path for exact tuple matches
+ if (isinstance(seq, (list, tuple)) and len(seq) == self.nlevels):
+ # ## Cache lookup ##
+ cache_key = tuple(seq)
+ if hasattr(self, '_loc_cache') and cache_key in self._loc_cache:
+ return self._loc_cache[cache_key]
+
+ # ## Boolean mask matching ##
+ # ## Cache management ##
+ return result
       # ## Original lookup code continues ##
```

Discussion This optimization introduces two improvements: specialized bit-combining in UIntEngine replacing NumPy's bitwise\_or.reduce with manual loops, and a fast path for exact tuple lookups using boolean masks with result caching. The changes accelerate common MultiIndex lookup patterns in time series and cross-sectional data analysis.

Human vs. Agent Solution: The human solution optimized the existing code path by replacing Python-level searchsorted calls with C-optimized algos.searchsorted and delaying intermediate array allocation. The agent implemented a specialized fast path for exact tuple matches with caching and hand-optimized bit manipulation. The agent's approach offers potentially larger gains for repeated exact-match queries but introduces caching complexity and only benefits a narrow subset of lookup patterns, while the human solution improved all lookup types uniformly.

#### <span id="page-34-0"></span>Example 10: Pandas MultiIndex argsort optimization

Model: O4-MINI Result: Success

Groundtruth Commit: <https://github.com/pandas-dev/pandas/commit/9ebb945f10> Specification Test Summary: Performance test evaluating Pandas' MultiIndex.argsort method for lexicographic sorting. The test creates a MultiIndex from 100,000 rows with two string keys (1000 users, 100 groups), performs argsort operations 5 times, and validates sorted order equivalence.

#### Model Generated Patch (Heavily Summarized)

```
diff --git a/pandas/core/indexes/multi.py b/pandas/core/indexes/multi.py
--- a/pandas/core/indexes/multi.py
+++ b/pandas/core/indexes/multi.py
@@ -2209,15 +2209,26 @@ class MultiIndex(Index):
      if len(args) == 0 and len(kwargs) == 0:
- # lexsort is significantly faster than self._values.argsort()
+ # Use numpy.lexsort on sorted codes for faster performance
          target = self._sort_levels_monotonic(raise_if_incomparable=True)
- return lexsort_indexer(
- target._get_codes_for_sorting(),
- na_position=na_position,
- )
+ # Extract and stack codes into 2D array
+ codes = target.codes
+ arr = np.vstack([code if isinstance(code, np.ndarray) else np.asarray(
code) for code in codes])
+
+ # Handle NaN positions (codes == -1)
+ if na_position == "last":
+ # ## Push -1 codes to end ##
+ elif na_position == "first":
+ # ## Push -1 codes to front ##
+
+ # Perform lexsort: reverse rows for correct priority
+ return np.lexsort(arr[::-1])
      return self._values.argsort(*args, **kwargs)
```

Discussion This optimization replaces value-based sorting with direct lexicographic sorting over integer codes. The approach extracts codes from each level, stacks them into a 2D array, handles NaN positioning, and uses NumPy's lexsort for efficient multi-key sorting. This eliminates expensive string comparisons and object array operations, significantly improving performance for large MultiIndex structures.

Human vs. Agent Solution: The human solution refactored the shared lexsort\_indexer and related sorting utilities, standardizing signatures and fixing typing issues. The agent took a more direct approach by inlining NumPy lexsort directly in MultiIndex.argsort, creating a 2D stacked array from codes and handling NaN positions locally. The agent's approach introduces additional memory overhead from the 2D array allocation and duplicates NaN handling logic that could be shared across sorting operations.

#### <span id="page-35-0"></span>Example 11: Pillow TIFF frame counting optimization

Model: O4-MINI Result: Success

Groundtruth Commit: [https://github.com/python-pillow/Pillow/commit/](https://github.com/python-pillow/Pillow/commit/96f95e8e9)

[96f95e8e9](https://github.com/python-pillow/Pillow/commit/96f95e8e9)

Specification Test Summary: Performance test evaluating Pillow's TIFF image handling, specifically n\_frames and is\_animated properties. The test processes multiple TIFF images with varying frame counts (1-5), measures execution time in different call orders, and validates consistency against reference results.

## Model Generated Patch (Key Changes)

```
diff --git a/PIL/TiffImagePlugin.py b/PIL/TiffImagePlugin.py
--- a/PIL/TiffImagePlugin.py
+++ b/PIL/TiffImagePlugin.py
@@ -951,28 +951,35 @@ class TiffImageFile(ImageFile.ImageFile):
   def n_frames(self):
      if self._n_frames is None:
- current = self.tell()
- try:
- while True:
- self._seek(self.tell() + 1)
- except EOFError:
- self._n_frames = self.tell() + 1
- self.seek(current)
+ # fast count IFD entries without decoding tags
+ fp = self.fp
+ orig_pos = fp.tell()
+ endian = self.tag_v2._endian
+ offset = self.__first
+ count = 0
+ while offset:
+ count += 1
+ fp.seek(offset)
+ entry_count_data = fp.read(2)
+ if len(entry_count_data) < 2:
+ break
+ num_entries = struct.unpack(endian + "H", entry_count_data)[0]
+ # skip entries and read next IFD offset
+ fp.seek(offset + 2 + num_entries * 12)
+ next_offset_data = fp.read(4)
+ if len(next_offset_data) < 4:
+ break
+ offset = struct.unpack(endian + "L", next_offset_data)[0]
+ self._n_frames = count
+ fp.seek(orig_pos)
      return self._n_frames
```

Discussion This successful optimization replaces inefficient frame-by-frame loading with direct binary traversal of TIFF's IFD pointers. The original implementation required full decompression of each frame with O(n²) complexity, while the optimized version reads only essential metadata (2-byte entry count, 4-byte offset) and skips tag parsing entirely. Similarly, is\_animated was optimized to check if a next-frame pointer exists rather than attempting to seek to frame 1. These changes reduced complexity to O(n) with minimal memory overhead while maintaining complete functional equivalence, making it particularly effective for scientific and medical imaging where multi-frame TIFFs are common.

#### <span id="page-36-0"></span>Example 12: Pandas IndexEngine allocation optimization

Model: CLAUDE-3.5-V2 Result: Success

Groundtruth Commit: <https://github.com/pandas-dev/pandas/commit/240854014e> Specification Test Summary: Performance test evaluating Pandas' Index-Engine.get\_indexer\_non\_unique method, which finds positions of values in an index. The test constructs a 300,000-element gamma-distributed index and queries 50,000 targets (70% from the index, 30% random), measuring execution time while validating correctness through detailed array equivalence checks.

## Model Generated Patch (Heavily Summarized)

```
--- a/pandas/_libs/index.pyx
+++ b/pandas/_libs/index.pyx
@@ -353,xx @@ cdef class IndexEngine:
- # Fixed-size initial allocation with constant-increment growth
- if n > 10_000:
- n_alloc = 10_000
- else:
- n_alloc = n
- result = np.empty(n_alloc, dtype=np.intp)
+ # First pass: Build value-to-indices mapping
+ # ## binary search optimization code simplified ##
- # Iterative lookup with frequent reallocation
- for i in range(n_t):
- # ## lookup and processing code ##
-
- # Resize with constant increment when needed
- if count >= n_alloc:
- n_alloc += 10_000
- result = np.resize(result, n_alloc)
+ # Second pass: Count exact matches needed
+ total_matches = 0
+ # ## counting code ##
- # Return slices of oversized arrays
- return result[0:count], missing[0:count_missing]
+ # Allocate arrays of exact size needed
+ result = np.empty(total_matches + missing_count, dtype=np.intp)
+ missing = np.empty(missing_count, dtype=np.intp)
+
+ # Final pass: Fill arrays without any reallocation
+ # ## filling code ##
+
+ # Return precisely sized arrays
+ return result, missing
```

Discussion This optimization transforms Pandas memory allocation for index lookups. The original implementation used fixed initial allocation (10,000 elements) with constant-increment reallocation (+10,000 elements), causing frequent resizing and memory waste. The optimized version uses a multi-pass approach: mapping values to positions, counting exact matches needed, then allocating precisely-sized arrays with no resizing. This is particularly effective for large indexes (300,000+ elements) with many lookup targets (50,000+), eliminating all dynamic resizing. The human solution kept the single-pass algorithm but replaced constant-increment resizing with exponential growth (n\_alloc \*= 2) capped at maximum size. The agent's solution restructured into multiple passes to determine exact allocation sizes upfront. Both improved performance significantly, but the agent's solution offers better memory efficiency through exact allocation, while the human's approach was simpler and less invasive.