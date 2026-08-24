# Optimizing Agentic Language Model Inference via Speculative Tool Calls

Daniel Nichols *Lawrence Livermore National Laboratory* Charles Jekel *Lawrence Livermore National Laboratory*

Prajwal Singhania *University of Maryland* Abhinav Bhatele *University of Maryland*

Harshitha Menon *Lawrence Livermore National Laboratory*

## Abstract

Language models (LMs) are becoming increasingly dependent on external tools. LM-based agentic frameworks frequently interact with their environment via such tools to search files, run code, call APIs, etc. Further, modern reasoning-based LMs use tools such as web search and Python code execution to enhance their reasoning capabilities. While tools greatly improve the capabilities of LMs, they also introduce performance bottlenecks during the inference process. In this paper, we introduce novel systems optimizations to address such performance bottlenecks by speculating tool calls and forcing sequences to remain resident in the inference engine to minimize overheads. Our optimizations lead to throughput improvements of several hundred tokens per second when hosting inference for LM agents. We provide a theoretical analysis of our algorithms to provide insights into speculation configurations that will yield the best performance. Further, we recommend a new "tool cache" API endpoint to enable LM providers to easily adopt these optimizations.

## 1 Introduction

Tool and function calling have enabled language models (LMs) to become useful for tasks beyond just conversation by providing the ability to interact with external environments and collect further context [\[12–](#page-12-0)[14,](#page-12-1) [20\]](#page-13-0). In particular, LMbased agentic tools and frameworks are often entirely reliant on external tools as they are designed to interact with the environment to solve some problem or accomplish a task. Popular agents such as software engineering agents (SWE agents) [\[9,](#page-12-2) [19\]](#page-13-1) need to interact with source code files and the command line to execute actions. Although access to external tools yields much richer capabilities and enables LMs to solve long-horizon, real-world tasks, it also introduces several performance bottlenecks in the traditional inference pipeline. Instead of a single, contiguous generation, the model alternates between generation and tool invocation, often across

many concurrent sessions, resulting in gaps due to waiting on tool completion.

> **[图片提取文字 (无描述)]:**
> Overview of Speculative Tool Use (32 Async gpt-oss-I 20b Agents) 1200 +196.4 tok/sec Throughput (tok/sec) 1100 1000 900 Our Approach Vanilla 800 3.0 0.5 2.0 2.5 0.0 Average Tool Latency (sec)
![](_page_0_Figure_10.jpeg)

<span id="page-0-0"></span>Figure 1: Our approach leads to up to 196 tokens/second improvement in the vLLM server when there are 32 gpt-oss-120b agents using it for inference.

As tool-centric agents become more prevalent in code copilots, personal assistants, and autonomous workflows, the overhead of using these agents is increasingly dominated by the time spent waiting on tools. Each tool call forces generation to stop and return to the user, who handles running the tool and sending back its output. This interrupt-driven process introduces a strict sequential dependency, where the latencies of individual tool calls accumulate and can significantly increase total generation time. Additionally, the evicted sequences and tool output must be rescheduled back into the inference engine, causing further overhead. Even with optimizations such as prefix-caching [\[5,](#page-12-3) [7,](#page-12-4) [18,](#page-13-2) [22\]](#page-13-3), there are still significant overheads to removing the sequences, processing the tool output, and rescheduling them for generation. These overheads are further exacerbated in multi-tenant settings where scheduling and prefix-caching optimizations become more challenging with many agents contending for resources, and many concurrent prompts evicting existing entries in the

KV-cache. Removing the sequential dependence and reducing these overheads in tool-heavy workloads is critical as agentic LMs become more widespread and need to be more economically viable and efficient.

Breaking the strict sequential dependency introduced by tool calls and reducing eviction overheads is non-trivial from a systems perspective. Tool latencies can span orders of magnitude and depend on external services, filesystem I/O, or userdefined code, making them difficult to predict or bound. Longrunning tools inevitably dominate end-to-end latency, whereas for short tools, the overheads of eviction and re-entry into the engine can outweigh any potential latency hiding strategies. Furthermore, decoupling tool execution from model progression introduces correctness challenges, and naively running tools early or out of order can lead to wasted computation with unused results or outputs that are inconsistent with the model's eventual decisions. Finally, existing inference optimizations such as speculative decoding [\[3,](#page-11-0) [8\]](#page-12-5) and optimized KV-cache management [\[7\]](#page-12-4) are implemented entirely within the inference engine and treat tools as opaque interruptions, making it difficult for client-side innovations to impact engine performance.

We address these challenges in agent tool calling by introducing *speculative tool execution*, realized through two variants. First, we present a client-side strategy that breaks the sequential nature of tool execution without requiring any modifications to the inference engine, making it immediately compatible with existing inference services. Second, we introduce an engine-level mechanism that enables the inference system to ingest tool outputs directly and eliminate evictionrelated overheads. These strategies allow us to mask portions of tool execution with generation for stateless and cheap-toexecute tools. Furthermore, we provide an analytical performance model for both approaches and discuss the impact of tool durations, prediction accuracies, and decode latencies on the performance improvements. The proposed approaches are evaluated across a range of workloads, demonstrating up to several hundred tokens per second throughput improvements. Figure [1](#page-0-0) highlights the savings of our approach over standard inference with tool calling agents.

In this work, we make the following important contributions:

- Two novel methods for speculating tool calls to optimize inference for agents by speculating work and keeping prompts resident in the engine.
- A theoretical analysis on the limitations of our methods, and which configurations will yield highest performing results.
- A detailed study of our two methods on agent workflows with tool calling.

## 2 Background

In this section we provide background on tool calling and speculative decoding.

## 2.1 Tool Calling

Most modern LMs use *tool calling* (or *function calling*) as an interface for interacting with external systems (command line, APIs, etc). In a typical setup, the user provides (1) a natural language prompt describing the task and (2) a list of available tools with structured argument schemas. Given this input, the model responds with either natural language or a structured *tool call* specifying which tool to invoke and with what arguments. The client then executes the tool and sends the tool output back to the model that includes the original conversation plus the tool output. In LM agents this process is typically repeated until the model has enough context to produce a final output or it has completed its tasks.

From the perspective of the inference engine, each tool call breaks the generation into multiple, shorter requests separated by externally executing the tool. For any request, the engine first processes all tokens in the prompt during the *prefill* phase to generate the first output token, and then produces subsequent tokens auto-regressively during the *decode* phase. When a tool call is generated, further decoding of the sequence is halted, and it is removed from the active batch while the client executes the tool. Once the tool result is available, the client submits a new request consisting of the original prompt, the intermediate text, and the tool output, forcing the engine to prefill the history again before it can decode the next tool call or final output. For agentic workloads with many tools and long multi-step plans, this leads to fragmented generations and overheads from repeatedly restarting generation with prompts increasing in length. In long agent executions prefilling long prompts and waiting on tools can easily become big bottlenecks.

Two common optimizations are used today to mitigate these costs:

Parallel tool calls. Instead of emitting a single tool invocation per turn, models can return multiple independent tool calls in one response [\[4,](#page-11-1) [15\]](#page-12-6). This enables the client to execute tools concurrently and then return results back to the model in a single response. This reduces the number of model/tool roundtrips, but still introduces gaps during inference where the engine has to hand off to the client. Furthermore, sometimes tools need to be called sequentially as they have a dependence on each other or the model needs the output of one tool before it can decide to call the next. Models also need to be trained to support parallel tool calling.

Prefix caching. Many inference runtimes maintain a *prefix cache* of KV state for recently used prompts [\[7,](#page-12-4) [22\]](#page-13-3). This enables the runtime to reuse the KV cache if a user submits

another request with the same history as a request they submitted shortly before; prefix caching drastically reduces prefill overhead during quick multi-turn API use. However, it does not eliminate the overhead of removing sequences from the batch, re-scheduling them, or managing cache eviction under multi-tenant loads, which can actually have high overheads if privacy-preserving algorithms are required [17, 18].

#### 2.2 Speculative Decoding

Speculative decoding is an inference optimization that accelerates LM generation by utilizing a smaller, faster *draft* LM to speculate tokens for a larger, slower LM. Given the same prompt, the draft model proposes several *speculative* tokens ahead of the target model. The target model then verifies these proposed tokens in parallel: as long as the draft tokens are likely under the target's distribution, they are accepted; when a mismatch occurs, the algorithm falls back to sampling from the target model at the first rejection point. This was first proposed by Leviathan et al. [8] and has been modified and improved in many ways since its first introduction [6].

This setup works well for two reasons: draft models can often predict some easier text with high accuracy and the target model can validate tokens in a single forward pass. The latter means there is no slowdown to inference, a forward pass of the target model is run regardless to sample the next token, but if it validates the speculative model, then we get several tokens for free. The only extra cost is the compute needed to run the speculative model.

This work adapts the ideas behind speculative decoding, but with some subtle differences. Generally, draft models in speculative decoding are only 3-10 tokens ahead of the target model. This works great for speculative generation, but in speculative tool calling, if we were to launch tools as the draft model generates them, then we would only be overlapping tool calls with a few generated tokens. This would only save a few milliseconds at most and be largely dominated by the longer tool calling times and API overheads. Instead, with tools it is better to speculate as soon as possible and, in the case of stateless cheap tools, as many times as possible. Our methodology is designed around this insight and key difference with speculative decoding.

### 3 Speculative Tool Calling Algorithms

Our goal in this work is to optimize the performance of tool calling in LM-based agents by speculating tool calls and executing them before they are needed. We propose two approaches for speculative tool calling: strictly client-side speculation that reduces tool latency and an inference-engine-side speculative algorithm that reduces both decode and prefill time.

In traditional tool calling, the user first submits a prompt to an API endpoint inference server. The server starts processing the prompt and then returns the generated tokens to the user when it hits a stop token. This could be the end of the sequence or a tool call token. When the user receives the response from the server, they check why the generation stopped (e.g. tool call, end of sequence, timeout, max tokens, API credits) and, if it is a tool call, then they run the tool locally using the tool parameters generated by the model. Once the tool completes the user passes the output back to the API and the LM can keep generating using the tool output. This process is detailed in Algorithm 1.

**Algorithm 1** Baseline Tool-Calling (Client-side, No Speculation)

```
Require: Prompt p, chat history H, tools \mathcal{T}, main LM \mathcal{M}
Ensure: Final model output y
  1: r \leftarrow \text{CALLAPI}(\mathcal{M}, p, H, \mathcal{T})
 2: if HASTOOLCALL(r) then
          \tau \leftarrow \text{ExtractToolCall}(r)
 4:
          v \leftarrow RunTool(\tau)
 5:
          H' \leftarrow H \cup \{(\tau, v)\}
                                         ⊳ append tool result to history
          y \leftarrow \text{CALLAPI}(\mathcal{M}, p, H', \mathcal{T}) \Rightarrow \text{return final answer}
 6:
 7: else
          y \leftarrow RENDER(r)
                                            ⊳ no tool call; use first pass
 8:
 9: end if
10: return y
```

#### 3.1 Client-side Speculative Tool Calling

We expand on the traditional tool calling approach with speculative tool calling on the client side as described in Algorithm 2. In this approach we asynchronously send our prompt to two API endpoints: the main model  $\mathcal{M}$  and a speculative model  $\mathcal{S}$ . The speculative model is much smaller and should return estimated tool calls with much lower latency than the main model. When  $\mathcal{S}$  returns a tool call, we asynchronously launch the tool and start executing it. We store a *future* for its result in a tool cache. When  $\mathcal{M}$  finally returns generated tokens, we check if a tool call is required. If it is required and there is a future for that tool call in the cache, we wait on the future instead of relaunching the tool. Figure 2 provides an overview of our approach. We can further increase speculation by launching several speculative model instances (represented by  $\lambda$  in Algorithm 2).

It is important to note that we can only speculate *stateless*, *cheap* tools. If a tool or function requires or modifies state, then speculating it is impossible without some undo or roll-back mechanism. This is an interesting potential optimization but out of the scope of this paper. Further, if a tool call is expensive, either in dollar costs or execution time, it is undesirable to launch many attempts of the tool ahead of time if they will not be correct. Fortunately, many common agent tools, such as web-search and file access, are both cheap and stateless providing ample opportunity for speculation.

> **[图片提取文字 (无描述)]:**
> Baseline Inference Engine and Client Setup with Tool Calling Overheads Decode 2 Decode 2 Decode // Engine Prefill Prefill Prefill Client Tool 2 Tool I Tool 3 Client-side Speculative Tool Calling Prefill Prefill Engine Decode Prefill Decode Decode Tool 2 Spec Client Tool I Spec Tool 3 Spec
![](_page_3_Figure_0.jpeg)

Figure 2: Overview of the client-side approach for speculative tool calling. Our approach speculates tool calls and overlaps their execution with generation.

```
Algorithm 2 Client-side Speculative Tool-Calling
Require: Prompt p, chat history H, tools T, sampling factor
      \lambda, main LLM \mathcal{M}, speculative LM \mathcal{S}
Ensure: Final model output y
  1: C \leftarrow \text{EMPTYMAP}()
                                              \triangleright cache: tool call \mapsto future
  2: spawn h_m \leftarrow \text{CALLAPI}(\mathcal{M}, p, H, \mathcal{T})

⊳ start main

      generation asynchronously
  3: for i \leftarrow 1 to \lambda do

          spawn:
          s_i \leftarrow \text{CALLAPI}(S, p, H, T)
  5:
          if HASTOOLCALL(s_i) then
                \tau \leftarrow \text{EXTRACTTOOLCALL}(s_i)
  6:
                if \tau \notin C then
  7:
                     C[\tau] \leftarrow \text{STARTTOOLASYNC}(\tau)
                                                                         ▶ begin
  8:
      executing tool; returns a future
                end if
  9:
10:
           end if
11: end for
12.

13: r_m \leftarrow \mathbf{await} \ h_m
14:
15: if HASTOOLCALL(r_m) then
           \tau_m \leftarrow \text{EXTRACTTOOLCALL}(r_m)
16:
17:
          if \tau_m \in C then
                v \leftarrow \mathbf{await} \ C[\tau_m]
                                                         ⊳ reuse if available
18:
19:
                v \leftarrow \text{RunTool}(\tau_m)

20:
21:
           H' \leftarrow H \cup \{(\tau_m, v)\} > append tool result to history
22:
          y \leftarrow \text{CALLAPI}(\mathcal{M}, p, H', \mathcal{T}) \quad \triangleright \text{ call } \mathcal{M} \text{ with result }
23.
     else
24:
          y \leftarrow \text{RENDER}(r_m)
                                                                 ⊳ no tool call
25:
26: end if
27:
28: return v
```

#### <span id="page-3-4"></span><span id="page-3-1"></span>3.1.1 Analysis of Client-side Speculation

We examine a particular case where the user is repeatedly calling the API and receiving a tool call from the API each time. This closely mimics the usage pattern of most AI agents and is, thus, a reasonable case to consider. The times to process *N* requests from the user in the standard (Algorithm 1) and speculative (Algorithm 2) case are shown below.

avg. 
$$\mathcal{M}$$
generation time

 $T_{\text{standard}} = N(G + T)$ 
avg.  $\mathcal{S}$  generation time

 $T_{\text{spec}} = \alpha N \max\{G, g + T\} + (1 - \alpha)N(G + T)$ 

$$\mathcal{S} \text{ acceptance rate}$$
 $\mathcal{S} \text{ acceptance rate}$ 

The left term in  $T_{\rm spec}$  is the case where the speculative model was correct and, thus, we only wait g+T for the tool to complete. The maximum is necessary since we still need to wait on  $\mathcal M$  to complete to validate the right tool call, so if g+T< G, then we still have to wait G. Assuming T,g,G, and N are positive, non-zero and  $0 \le \alpha \le 1$  we can further analyze the speedup of the speculative algorithm.

$$\begin{split} S_{\text{spec}} &= \frac{T_{\text{standard}}}{T_{\text{spec}}} &> 1 \\ &= \frac{G+T}{\alpha \max\{G,g+T\} + (1-\alpha)(G+T)} &> 1 \quad (2) \\ &\text{If } g+T \geq G \text{, then} \\ &S_{\text{spec}} = \frac{G+T}{\alpha(g-G)+G+T} &> 1 \\ &\alpha(g-G)+G+T &< G+T \\ &\alpha(g-G) &< 0 \end{split}$$

We can immediately see that  $S_{\text{spec}} > 1$  when S has a non-zero acceptance rate and S is faster than M. The case where

 $\Longrightarrow \alpha > 0$  and  $\varrho < G$ 

<span id="page-3-3"></span><span id="page-3-2"></span>(3)

*g* + *T* < *G* is ignored, since this trivially leads to speedups greater than 1 when α > 0 (i.e. speculation and tool calling are completely masked by the main models generation time, so any correct speculation should lead to speedups). Based on Equation [\(3\)](#page-3-2) *to achieve speedups with speculative tool calling on the client side we need to find a speculation model S that is faster than the main model M and has a tool speculation accuracy greater than 0*.

Another key insight from Equation [\(2\)](#page-3-3) is that *S*spec < 2, i.e. *the maximum speedup we can achieve in this approach is 2. Furthermore, the speedup approaches 2 as g* → 0. Figure [3](#page-5-0) shows the distribution of *S*spec across values of α, *g*/*G*, and *T*. We see the above trends, i.e. that *S*spec → 2 as *g* → 0 + and *S*spec < 2, but also that we only see large speedups for values of *T* near *G*. As *T* → 0 the total time is predominantly dominated by *G* and as *T* → ∞ it is dominated by the tool call. Thus, when *T* ≈ *G* there are more values of α and *g*/*G* where we can see substantial benefits. The above insights are also proven below in Lemma [1.](#page-4-0)

<span id="page-4-0"></span>Lemma 1 (Speedup bound). *Let G*,*g*,*T* > 0*, g* < *G, and* α ∈ [0,1]*. Define*

$$S_{\text{spec}}(\alpha) = \frac{G+T}{\alpha \max\{G, g+T\} + (1-\alpha)(G+T)}.$$

*Then:*

- *1. S*spec(α) *is strictly increasing in* α*, hence S*spec(α) > 1 *for all* α > 0*.*
- *2. The maximum over* α ∈ [0,1] *is attained at* α = 1 *and equals*

$$S_{\max} = \frac{G+T}{\max\{G, g+T\}}.$$

*3. Moreover,*

$$S_{\text{max}} \le \frac{2(G+T)}{G+g+T} = 2 - \frac{2g}{G+g+T} < 2.$$

*In particular, the supremum* 2 *is approached only in the limit g* → 0 +*.*

*Proof.* Let *M* := max{*G*, *g*+*T*} and write

$$S_{\text{spec}}(\alpha) = \frac{G+T}{\alpha M + (1-\alpha)(G+T)}$$
$$= \frac{G+T}{(G+T) + \alpha (M-(G+T))}.$$

Set ∆ := *M* −(*G*+*T*). Since *M* ≤ *G*+*T*, we have ∆ ≤ 0, with strict inequality because *G*,*g*,*T* > 0.

(1) Since *M* ≤ *G*+*T* and *G*,*g*,*T*,α > 0, the denominator of *S*spec(α) is strictly decreasing as α increases. Thus, we can conclude that *S*spec(α) is strictly increasing with α. Furthermore, since *S*spec(0) = 1 it follows that *S*spec(α) > 1 for all α > 0.

(2) Monotonicity implies the maximum over [0,1] occurs at α = 1, giving

$$S_{\text{max}} = S_{\text{spec}}(1) = \frac{G+T}{M} = \frac{G+T}{\max\{G, g+T\}}.$$

(3) For any *a*,*b* > 0, max{*a*,*b*} ≥ *<sup>a</sup>*+*<sup>b</sup>* 2 . Applying this with *a* = *G* and *b* = *g*+*T*,

$$\begin{split} S_{\max} &= \frac{G+T}{\max\{G, g+T\}} \\ &\leq \frac{G+T}{\frac{G+(g+T)}{2}} = \frac{2(G+T)}{G+g+T} = 2 - \frac{2g}{G+g+T} \\ &< 2, \end{split}$$

where the strict inequality uses *g* > 0. This proves the stated bound and its strictness.

#### Observation 1: Client-side speculation limit

Client-side speculative tool calling is limited at a 2× speedup: we can only hide one of the two dominant phases (generation or tool execution). For a good speculative model (fast and accurate), the best gains occur when the tool latency *T* is approximately equal to the main model generation time *G*.

### 3.2 Engine-side Speculative Tool Calling

Next, we present our approach of moving our speculative tool calling method from the client to the engine. First, we present how a non-speculative inference engine with tool calling works in Algorithm [3.](#page-6-0) The engine repeatedly ingests new requests from the input queue into an active batch. For each sequence in the batch, it first encodes the prompt to produce the first token (prefill) and then generates subsequent tokens (decode). When a stop token (end of sequence or tool call token) is produced, the engine stops generation. If the stop token is a tool call token, the engine returns the tool call to the client, and evicts the sequence from the active batch. Its KV-cache is discarded unless prefix-caching is enabled.

The engine-side speculative approach adds a tool cache maintained by the inference engine. Similar to the previous approach, the client spawns multiple speculative model instances and launches tools asynchronously. However, instead of storing futures, it waits for each tool execution to finish and then submits the result to the engine, indexed by a normalized key (tool name + canonicalized arguments) and the request ID. On the engine side, we introduce two optimizations based on the tool cache. When a tool start token (the first token indicating that the model is about to produce a tool invocation) is detected for a sequence, the engine performs a cache lookup using the request ID and tool name. If found, the argument tokens of the latest matching tool call are injected as draft

#### <span id="page-5-0"></span>Speculative Tool Calling Speedups from Performance Model

> **[图片提取文字 (无描述)]:**
> $T = 0.25 \cdot G$  $T = 0.5 \cdot G$  $T = 0.75 \cdot G$  $T = 2 \cdot G$  $T = I \cdot G$
![](_page_5_Figure_1.jpeg)

Figure 3: Distribution of speculative tool calling speedups for the client-side algorithm across various acceptance rates, generation times, and tool call times.  $\alpha$  is the acceptance rate of the speculative model S, g/G is the ratio of generation time between the speculative and main models, and T is the tool call time. Values of  $T \approx G$ ,  $\alpha > 0.5$ , and g/G < 0.5 yield higher speedups, but are ultimately limited at a  $2 \times$  speedup.

> **[图片提取文字 (无描述)]:**
> Baseline Inference Engine and Client Setup with Tool Calling Overheads Decode // Decode // Decode // Prefill Prefill Prefill Engine Tool 2 Tool 3 Client Tool I Engine-side Speculative Tool Calling Keep sequences resident; avoid eviction overheads and force prefix caching Prefill Prefill Prefill Engine Dec Dec Dec 2. Shorter decodes due to early exits if speculation is correct Tool 2 Client Spec Spec Tool 3 Spec Tool I
![](_page_5_Figure_3.jpeg)

Figure 4: Overview of our proposed engine-side algorithm for speculative tool calling for agents. First, our approach speculates tool calls and overlaps their execution with generation. Second, if speculated tool calls are available before the target model finishes decoding it, then we can validate the tool call in a single forward pass and early exit decoding. Finally, by posting speculated tool outputs to the server, we prevent the prefill time from continuing to grow and remove overheads from removing the prompt from the batch.

tokens and validated through standard speculative sampling. This ensures correctness and potentially avoids the need to generate the tool call arguments (early exit decoding). When a tool end token (the token indicating that the full tool call has been emitted) is detected, the engine looks up the full key. On a cache hit, the tool result is appended and the KV-cache updated so decoding can continue without eviction. Otherwise, the engine falls back to baseline behavior and emits the tool call to the client. Figure 4 provides an overview of this approach. The client and engine behaviors under this approach are described in Algorithms 4 and 5, respectively.

#### 3.2.1 Analysis of Engine-side Speculation

In the traditional case of tool calling, where the user receives the tool parameters, runs it, and calls the API again, we can model the runtime from the inference engine's perspective as shown in Equation (4) for K consecutive agent turns. This accounts for the prefill and decode phase of each turn where  $X_i$  tokens are prefilled,  $R_i$  reasoning tokens are decoded, and

<span id="page-5-1"></span> $t_i$  tool call tokens are decoded. Then the tool is called for  $T_i$  seconds after which the tool call and output  $(t_i + t_{o,i} \text{ tokens})$  are sent back to the API. Finally, the entire history and new tool calls will be prefilled again, so we sum them in the  $X_i$  recursive term. In between these phases there is also overhead, o, to passing the prompts through the API via HTTP and loading it into the running batch.

<span id="page-5-2"></span>API and eviction overheads Decode rate (s/tok) Tool duration (s)

$$T_{\text{vanilla}} = 2K \text{ o } + \phi \sum_{i=1}^{K} X_i + \delta \sum_{i=1}^{K} \left( R_i + \underbrace{t_i} \right) + \sum_{i=1}^{K} \underbrace{T_i}$$
Prefill rate (s/tok) Reasoning tokens

such that

<span id="page-5-3"></span>
$$X_{i+1} = X_i + t_i + t_{o,i}$$
,  $X_1 = \text{initial prompt tokens}$ 

Here we assume that (1) reasoning tokens from previous turns are not included in the prefill for future turns as is typical

**Algorithm 3** Baseline Inference Engine: Batch Decode with Evict-then-Refill on Tool Calls

```
Require: Input queue Q, model \mathcal{M}, batch size K
 1: B ← Ø
                                        > active batch of sequences
 2:
    while SERVICERUNNING() do
         fill B from Q up to K ready requests
 3:
         for all s \in B do
 4:
              if s.STATUS = NEW then
 5:
                  t \leftarrow \text{PREFILL}(\mathcal{M}, s)

⊳ first token

 6
 7:
                  s.status \leftarrow Decode
              else if s.STATUS = DECODE then
 8:
                  t \leftarrow \mathsf{DECODESTEP}(\mathcal{M}, s) \quad \triangleright \mathsf{get} \; \mathsf{next} \; \mathsf{token}
 9:
              end if
10:
11:
              if IsStop(t) then
12:
                   EMITTOCLIENT(s, final)
13:
                   EVICT(B, s)

14:
15:
              end if
16:
17.
         ingest any new/follow-up requests from client (e.g.
     tool results) into Q
18: end while
```

for most commercial APIs and (2) there are no additional outputs alongside the tool call during each turn. The former assumption is generally true across API models, while the latter depends on how the agent framework is implemented (directly using tools versus thought-actions as in SWE-Agent). Despite this, our model is without loss of generality, since you can absorb these extra decode tokens into the  $t_i$  term.

Note that in the case of prefix-caching we can express the time as shown in Equation (5). Here we do not have to repeat prefills for tokens we have already populated into the KV-cache.

$$T_{\text{cached}} = 2Ko + \varphi\left(X_1 + \sum_{i=1}^{K} (t_i + t_{o,i})\right) + \delta \sum_{i=1}^{K} (R_i + t_i) + \sum_{i=1}^{K} T_i$$
(5)

Next we consider the three primary optimizations: (1) speculating tools prior to execution (OI), (2) validating cached tool calls with a single forward pass during decoding (O2), and (3) avoiding eviction and duplicate prefilling for tool calls where the output is already available (O3). The first optimization (OI) can save us up to  $\sum_{i=1}^{K} T_i$  time as it can potentially mask the tool call time. The O2 optimization can potentially save  $\delta \sum_{i=1}^{K} (t_i - 1)$  time as we reduce the number of decodes from  $t_i$  for generating the tool call to 1 for validating the tool call. Finally, the O3 optimization saves 2(K-1)o overhead in the ideal case and effectively forces prefix-caching to reduce the number of prefill tokens. Based on these optimizations we can write the expected time in terms of the speculation accuracy  $\alpha$  as show in Equation (6).

**Algorithm 4** Client with Speculative Tool Execution and Cache Submission

```
Require: Prompt p, chat history H, tools \mathcal{T}, main LM API
     endpoint E, speculative LLM S, samples \lambda
Ensure: Final output y
  1: (h, rid) \leftarrow \text{CALLAPI}(\mathsf{E}, p, H, \mathcal{T}) \Rightarrow \text{start main request};
     get request id
 2: for i \leftarrow 1 to \lambda do

 3:
           spawn s_i \leftarrow \text{CALLLM}(S, p, H, T)
           if HASTOOLCALL(s_i) then
 4:
 5:
                \tau \leftarrow \text{EXTRACTTOOLCALL}(s_i)
                f \leftarrow \text{STARTTOOLASYNC}(\tau)
 6:
                spawn ONREADY(f):
                    v \leftarrow \mathbf{await} \ f
                    k \leftarrow \text{CANONKEY}(\tau) \triangleright \text{normalize: tool name}
     + canonicalized args
                    SUBMITTOOLCACHE(E, rid, k, v)
           end if
     end for
 9:
11: r \leftarrow \mathbf{await} \ h \triangleright \text{receive result (may be final or a tool call)}
13: if HASTOOLCALL(r) then \triangleright cache miss path: engine
     did not find value in its cache
           \tau^{\star} \leftarrow \text{ExtractToolCall}(r)
14:
15:
           v^* \leftarrow \text{RunTool}(\tau^*)
          H' \leftarrow H \cup \{(\tau^{\star}, v^{\star})\}
16:
           r \leftarrow \text{CALLAPI}(\mathsf{E}, p, H', \mathcal{T}) \triangleright \text{continue normal loop}
17:
           while HASTOOLCALL(r) do
18:
                \tau \leftarrow \text{EXTRACTTOOLCALL}(r)
19:
                v \leftarrow \text{RUNTOOL}(\tau); H' \leftarrow H' \cup \{(\tau, v)\}
20:
21:
                r \leftarrow \text{CALLAPI}(\mathsf{E}, p, H', \mathcal{T})
           end while
22:
23: end if
24: y \leftarrow RENDER(r)
```

<span id="page-6-2"></span>
$$T_{\text{spec}}^* = (1 - \alpha)2Ko + \varphi\left(X_1 + \sum_{i=1}^K (t_i + t_{o,i})\right) + \delta\left(\alpha K + \sum_{i=1}^K R_i + (1 - \alpha)\sum_{i=1}^K t_i\right) + (1 - \alpha)\sum_{i=1}^K T_i \quad (6)$$

We can immediately see this is equivalent to  $T_{\text{cached}}$  for  $\alpha = 0$ . When  $\alpha = 1$  we save  $2Ko + \delta(\sum_{i=1}^{K} t_i - K) + \sum_{i=1}^{K} T_i$  time over prefix-cached inference, since we avoid 2Ko overheads, the tool latencies, and the extra decodes for the tool calls. Speedups will be optimal when the tool call duration  $T_i$  is less than  $\delta R_i$  so that the tool output will be available for speculation (O2). As in Section 3.1.1 the optimal speedups for a single turn will then come when  $T_i \approx \delta R_i$  as we are masking the entire tool call and saving on the most decode steps. How-

25: return y

#### Algorithm 5 Inference Engine with Tool Cache

<span id="page-7-0"></span>Require: Input queue Q, model *M* , batch size *K*, tool cache

```
C keyed by (request id, canon key) and (request id, tool
  name)
1: B ← 0/
2: while SERVICERUNNING() do
3: ASYNCUPDATE(C) ▷ from SUBMITTOOLCACHE
4: fill B from Q up to K
5: for all s ∈ B do
6: if s.STATUS = NEW then
7: t ← PREFILL(M ,s); s.STATUS ← DECODE
8: else if s.STATUS = DECODE then
9: t ← DECODESTEP(M,s)
10: end if
11:
12: if ISTOOLCALLSTART(t) then
13: η ← EXTRACTTOOLNAME(s,t)
14: if C.HAS(rid(s),η) then
15: c ← C.GETTOOLCALL(rid(s),η)
16: t ← SPECULATIVESAMPLE(M ,s,c) ▷
  Validate tool call tokens and update KV cache
17: end if
18: else if ISTOOLCALLEND(t) then
19: τ ← EXTRACTTOOLCALL(s,t)
20: k ← CANONKEY(τ)
21: if C.HAS(rid(s),k) then ▷ cache hit
22: v ← C.GETTOOLRESULT(rid(s),k)
23: t ← PREFILLTOOLRESULT(M ,s, τ,v) ▷
  append tool result tokens to KV cache
24: end if
25: end if
26:
27: if ISSTOP(t) then ▷ cache miss/other stop token
28: EMITTOCLIENT(s,final)
29: EVICTFROMBATCH(B,s)
30: end if
31: end for
32: ingest NEW client requests into Q
33: end while
```

ever, any value *T<sup>i</sup>* < δ*R<sup>i</sup>* should yield runtime savings. This is an ideal finding as many state-of-the-art reasoning models commonly used in agents have several seconds to minutes reasoning traces, meaning we can speculate and execute many common tools in this time-frame.

### 4 Experimental Setup

With the two proposed algorithms defined we now detail how we implemented them and the experiments we ran to test their effectiveness.

## 4.1 Algorithm Implementations

The client-side algorithm is implemented on top of the OpenAI API [\[2\]](#page-11-2) using its async client. Whenever a chat completion or response is posted to the main model, we also post a chat completion/response to the speculative API endpoint asynchronously. If using multiple speculative samples, then we send multiple asynchronous requests. When the speculative model API calls return, we launch the speculated tools asynchronously and put their futures into a tool cache keyed on function name and arguments. This is implemented as a custom Python interface using Python's built-in asyncio library. When the main model returns we check the tool cache if it called a tool. If there is a cache hit, then we wait on the future and use its result as the tool output. Otherwise we call the tool as normal.

For the engine-side algorithm we modify a custom fork of the popular vLLM framework [\[7\]](#page-12-4) to implement our algorithm. A tool cache endpoint is implemented as a standard HTTP POST API endpoint (see later API spec). Similar to the client-side setup, one or more speculative model endpoints run asynchronously in parallel with the main model (*O1*). The results of the speculated tool calls are posted to the toolcache endpoint. Building on vLLM's speculative decoding infrastructure, we implement a tool proposer responsible for detecting tool-call boundaries in the generated token stream. When the start of a tool call is detected, the proposer drafts the tool call tokens matching the latest entry in the cache with the same tool name. These draft tokens are then validated using standard speculative sampling [\[8\]](#page-12-5), maintaining correctness (*O2*). When the end of a tool call is detected, the proposer performs a lookup in the cache using both the tool name and its arguments. On a cache hit, the corresponding tool-output tokens are drafted; in this case, all drafted tokens can be accepted directly, since the match is exact on both name and arguments (*O3*). If the lookup results in a cache miss, generation stops, and the engine returns the tool call to the client for normal execution. For the engine-side algorithm to work in multi-turn scenarios we ensure streaming is enabled, so the client knows when to start speculating new tools.

The custom API endpoint we use is detailed below. This simple interface is all that is needed for existing commercial endpoint providers to expose this optimization to users. Providers are incentivized to use the cache-tool-output API as it can cut down on inference time and increase throughput. End-user can be incentivized by reduced costs and faster turnaround times. The cost reduction is not necessarily implicit, however, many providers already offer discounts on prompts that hit the prefix-cache and our proposed optimization would fit nicely within that existing pricing model.

#### API: **POST /cache-tool-output/{response\_id}**

Description. Cache tool outputs that the inference engine can utilize to keep sequences resident after tool calls. Each entry is keyed by name and (optionally) params and is matched against future tool calls by the model in this response.

Request body. JSON array of objects with the following fields:

| Field                            | Description                                                                                                                                            |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                             | (string, required) Unique tool name used                                                                                                               |
| params                           | as part of the cache key.<br>(object, optional) Canonicalized tool<br>parameters included in the cache key.<br>Calls with the same name and equivalent |
| output                           | params share outputs.<br>(any, required) Serialized tool output to be<br>cached and reused when the model emits                                        |
| keep_alive                       | a matching tool call.<br>(number, optional) Optional time-to-live<br>for the cache entry (e.g., in seconds).                                           |
| Response.                        |                                                                                                                                                        |
| 200 OK                           |                                                                                                                                                        |
| {<br>"cached": <int><br/>}</int> | // number of entries accepted                                                                                                                          |

### 4.2 Testing Environment

We ran our implementations and experiments on a single compute node with four 80GB NVIDIA A100 GPUs controlled by one AMD EPYC 7763 CPU with 64 physical cores each at 2.45 GHz. Each GPU pair is connected by third generation NVLink with 25 GB/s per direction links and each GPU is connected to the CPU via PCIe 4.0. We run on a variation of SUSE Enterprise Linux Server 15 SP5 with Python 3.12 and CUDA 12.9. In our experiments, we run two vLLM inference servers: one for the main model and one for the speculative model. The main model *M* uses a single A100 for serving, while the speculative server hosts three data parallel instances of the speculative model *S* across the other three A100s.

### 4.3 Experiments

To evaluate our inference optimizations, we use the dataset of prompts and tools from the BFCL project [\[11\]](#page-12-8): a benchmark used to evaluate LMs on their tool calling capabilities. BFCL defines a large number of tools that span a wide variety of tasks. As a leaderboard, BFCL presents results comparing how well different LMs perform at calling the correct tools. This provides a great starting point for selecting small, accurate speculation models.

From the BFCL leaderboards, we selected the xLAM models [\[21\]](#page-13-5) as ideal candidates for speculative models. They are Llama 1B, 3B, and 8B fine-tunings on tool datasets to predict tool calls. During testing we found the 8B model to predict around 80% of tool calls from BFCL correctly, which is in line with the results they present in their paper. Since these models are small and accurate at predicting tools, we utilize them as speculative tool calling models.

BFCL defines pairs of input prompts and sets of tools that can be invoked for each prompt, but it does not provide concrete tool implementations (only specifications). Our goal in this work is to study the inference- and systems-level behavior of agents with tools.We therefore treat tools as black boxes with specified interfaces and latencies. For each BFCL tool, we pre-compute a representative output using a mix of LM and human authored responses, and store these outputs in an in-memory cache. During experiments, whenever the model calls a tool, the cached tool output is returned and its runtime is controlled via manually configured tool latencies.

Crucially, our algorithms and measurements depend only on *when* tools are called and *how long* they take to run, not on the content of the tool outputs themselves. Precomputing outputs in this way preserves the model's compute behavior while allowing us to systematically control tool latencies. As a consistency check, we manually inspected model generations across our experiments and found them to be reasonable and in line with typical outputs from the base model. We also ran a small experiment using simple, hand-implemented tools with real executions and observed no qualitative change in model behavior or quantitative change in the performance results, supporting the validity of our tool setup for systemlevel performance evaluation.

To turn BFCL into an agent workload, we treat each prompt and toolset pair as an independent task for an asynchronous agent. We run *M* concurrent agents with *M* ∈ {1,8,32}. Each agent samples a prompt and its associated tools. It will loop continually calling tools (when requested) and generating text until it is finished and produces a final response. Then it will receive another prompt and tools, and keep repeating this process with new prompts. We run each experiment long enough for every agent to complete 32 such tasks, and record the performance metrics reported in Section [5](#page-9-0) over the full run. During our experiments we use the gpt-oss-120b [\[10\]](#page-12-9) model as our primary agent as it is a state-of-the-art reasoning model that might be commonly used in an agent.

We systematically sweep the main experimental hyperparameters to study their impact. For tool behavior, we sample tool latencies from random normal distributions of two types: *short-latency* distributions with mean latencies ∈ {0,0.1,...,0.5}, and *long-latency* distributions with means ∈ {0,0.5,...,3}. For speculation, we vary the number of speculative samples per turn (1,3,5,7, and 9) and the speculation model (xLAM 1B, 3B, and 8B). For each point in this search space, we run the workload under three inference configurations: standard inference (no speculation), client-side speculation, and engine-side speculation. Each of these experiments is run five times to account for variation and system noise. Additionally, we run a subset of experiments with the client-side algorithm and commercial models gpt-5 and gpt-5-nano to study how well this approach works for commercial models.

#### <span id="page-9-0"></span>5 Evaluation Metrics

With our tool-calling agent experiments set up, it is important that we are able to measure the benefits of speculation in order to compare approaches. Metrics like walltime are tricky since some agents might decide to generate lots of response text to a prompt, while another gives a short response. This can happen due to diversity in sampled outputs, even for the same prompt. For example, consider agents A and B each call tools, receive outputs from those tools, and then generate a response. Agent A has its tool speculated correctly, and agent B does not. If agent A generates a 2000 token response to the tool output, while agent B generations a 50 token response, then agent A will have a larger wall time despite getting better benefits from speculation. For this reason we use *throughput* and *percent time saved* as metrics for comparison.

**Throughput** is computed as the number of tokens generated per second. We compute this for each agent and report the average throughput.

**Time Saved** is a measure of percent reduced time relative to a non-speculative baseline. We record time from agent start to finish with and without speculation for the same prompt(s). Let  $T_{\rm base}$  denote this time for the baseline agent (no speculation) and  $T_{\rm spec}$  the time for the same agent configuration using our speculative algorithm. We define

Time Saved = 
$$100 \times \frac{T_{\text{base}} - T_{\text{spec}}}{T_{\text{base}}} \%$$
.

We compute this quantity per agent and report the mean across all the async agents. Because both runs share the same prompts, tools, and tool outputs, this metric isolates the reduction in inference time attributable to speculation.

In addition to throughput and time saved, we also analyze cost for the client side algorithm. Since it can be implemented entirely using an API it can be used with commercial models. We measure the **cost** using the token usage data returned by the OpenAI API and the publicly listed per-token prices from OpenAI (costs based on November 2025 pricing). Cost is reported as the *additional cost* of speculation versus just using the main model. Since agents can vary in number of turns or tokens, we report the cost per 100 agent turns.

#### 6 Results

In this section we present the results from our evaluation of the two proposed algorithms.

#### 6.1 Client-side Speculation

Figure 5 shows the throughput of various speculative models with different acceptance rates for the client-side algorithm on a subset of the BFCL prompts. We show these results on an easier to predict subset of tools, so that we can observe the behavior for values  $\alpha > 0.8$  to validate Equation (2). In the later experiments we observed  $\alpha \approx 0.8$  for xLAM-2-8B. Results are shown for runs with an average tool latency of 1.5 seconds, but the trends are similar for other tool latency amounts. Point annotations denote how many tool calls are speculated by the speculative model each turn ( $\lambda$  in Algorithm 2)(e.g.  $1 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3 \times 0.3$ 

> **[图片提取文字 (无描述)]:**
> Throughput vs Cache Hit Rate Tool Latency: 1.5s | Total Async Clients: 32 1150 Spec. Model xLAM-2-IB Throughput (tok/s 1100 xLAM-2-3B xLAM-2-8B 1050 None 1000 (lx)950 0.2 0.4 0.8 0.0 0.6 1.0 Cache Hit Rate  $(\alpha)$
![](_page_9_Figure_12.jpeg)

<span id="page-9-1"></span>Figure 5: Throughput improvement for various speculative models and their cache hit respective tool cache hit rates. Points are annotated with the number of speculations per generation. We see a clear trend in improved throughput as the tool prediction rate increases.

We see that the trends from our performance model hold (see Section 3.1.1) hold; as speculation rate increases, we see an increase in inference throughput. The 8B model yields much better accuracy in tool prediction than the smaller models, however, we see that the accuracy of the smaller models can also be improved by increasing the number of samples. xLAM-2-1B achieves nearly 60% higher hit rate with  $9\times$  samples versus  $1\times$  samples. This flexibility is important as we may desire to run smaller speculative models due to memory constraints; taking more samples with a smaller model can get close to the performance of a larger speculative model.

Figure 6 shows the average percent time saved for each agent being run using the client side speculative algorithm. Even for shorter tools, around 0.5 seconds in duration, we still observe noticeable improvements with up to 6% of total time saved by speculating the tools. Results improve as the average tool time approaches gpt-oss-120b's average generation time; when tools average around 2.0-2.5 seconds we see the biggest gains with up to 21% saved. The percent time saved slowly

> **[图片提取文字 (无描述)]:**
> Average Percent Time Saved Per Agent (32 Async gpt-oss-120b Agents) 25 xLAM-2-IB Percent Time Saved (%) 2 0 1 0 2 xLAM-2-3B xLAM-2-8B 2.0 2.5 3.0 Average Tool Latency (sec)
![](_page_10_Figure_0.jpeg)

Figure 6: Average percent time saved across runs with various speculative models and average tool latencies. Results are shown for  $1 \times$  speculation per generation. We see that client side speculation can lead to between 6-21% end-to-end time saving per agent when average tool latencies are 0.5 to 3 seconds.

decrease as tools dominate the runtime and speculation only provides minor savings for long lasting tools. This behavior is further evidence to the performance models in Section 3.1.1 where we observe that tool calls around the generation time of the main model lead to the highest speedups.

#### **Observation 2: Client-side Speculation Benefits**

An agentic framework, without any additional changes to the inference engine or API, can achieve 6-21% time saved by speculating tools ahead of time. Savings vary depending on the ratio of model generation time and average tool call duration.

> **[图片提取文字 (无描述)]:**
> Cost vs Time Saved for End-to-End Agents Client-side speculation with gpt-5 and gpt-5-nano Time Saved By Speculation (%) Avg. Main Model Cost per 100 Turns: \$0.606 **13.5%** 13.4% 12.8% 12.4% 9.5% Num. Speculative Calls \$0.025 \$0.05 \$0.075 \$0.1 \$0.12 \$0.15 \$0.18 \$0.2 \$0.23 Increased Agent Cost per 100 Turns (\$)
![](_page_10_Figure_5.jpeg)

<span id="page-10-1"></span>Figure 7: The increased cost for running client-side speculation with gpt-5-nano alongside a gpt-5 agent. Costs are shown for 100 agent turns. For only a 4% price increase, agents can save nearly 10% time.

We can also run the client-side speculation algorithm for commercial models where we only have API access. Figure 7 shows the percent time saved by using speculation versus the added cost of doing that speculation. The results are presented for a single agent "turn", or one API then tool call pair. We see that with one speculation we can get nearly 10% time savings with only an extra API call that is an order of magnitude cheaper than the main API call. Running nine speculations increases the time saved marginally, but is around 25-30% of the cost of the main generation.

#### <span id="page-10-0"></span>**6.2** Engine-side Speculation

Figure 8 shows the percent time saved for the engine-side algorithm. We additionally show the times for average tool latencies in  $\{0.1, 0.2, 0.3, 0.4, 0.5\}$  to highlight the regime where engine-side speculation helps most. vLLM's speculative decoding infrastructure, what we build our implementation on top of, is currently in an experimental development phase [1] and we found high overheads for its spec-dec implementation when the batch size is greater than 1. For this reason we present our engine-side results with only one asynchronous agent. Note that when there is only one asynchronous agent we see much lower generation latency from vLLM due to the request scheduler and smaller batch. Our observations align with findings in other works [16]. This leads to less potential gains as there is less generation time to overlap tool calls with, so we see lower percent time saved than in Figure 6.

> **[图片提取文字 (无描述)]:**
> Average Percent Time Saved Per Agent (I Async gpt-oss-I 20b Agents) 12.5 Client-side Percent Time Saved (%) 10.0 Engine-side 7.5 5.0 2.5 Average Tool Latency (sec)
![](_page_10_Figure_10.jpeg)

<span id="page-10-2"></span>Figure 8: Average percent time saved comparison between client-side and engine-side algorithms for various average tool latencies. Results shown for one async client and xLAM-2-8B for speculation. When tools finish before the main model is done reasoning, we can achieve 2-3% better time savings than the client-side approach.

#### Observation 3: Engine-side Speculation Benefits

Posting speculated tool outputs to an inference server to keep sequences resident when tool outputs are available a priori can lead to an additional 2-3% time saved from the client's perspective. The best benefits occur where tools latencies are less than the main model *M* 's reasoning time. This is ideal as many common agent tools fall in the 0-1 second range (e.g. ls, web search, read file, ...).

We see 2-3% increased time savings in the engine-side algorithm compared to the client-side algorithm. This is on top of the already high savings from the client-side speculation. We see the best results where tool calls are between 0 and 1 second as these finish and are posted to the inference server before the reasoning phase is done. Longer tool calls do not see any benefit over client-side speculation, since their results are not posted to the inference server before decoding is ready.

### 7 Related Work

Several works have looked at accelerating LM inference using speculation. Leviathan et al. [\[8\]](#page-12-5) introduce speculative decoding to use a smaller LM to predict tokens for a larger LM. While very related to our approach, the traditional speculative decoding does not work well for speculating tool calls. As we have demonstrated in our analysis the main speedup from speculating tools comes from overlapping tool execution with generation. In traditional speculative decoding the draft model is usually only ever 10 tokens ahead of the main model meaning we could only overlap tool calls with 10 decoding steps. In the context of tool calling it is better to speculate tools as early and as many times as possible. Other works [\[3\]](#page-11-0) have looked into other variations of generation speculation, such as in sampling, but suffer from the same drawback as speculative decoding: tool calls need to be speculated as soon as possible in the generation.

Other approaches to improving agent tool use performance try to increase the accuracy of smaller models at predicting the right tool to use. By using smaller LMs for taking actions we can reduce inference time. Schick et al. [\[14\]](#page-12-1) and Patil et al. [\[11\]](#page-12-8) accomplish this through fine-tuning on large datasets of tool calling interactions. Works like SWE-Agent [\[19\]](#page-13-1) accomplish better and faster agent generation by providing the *best granularity* of tools to the agent. By giving it the right tools they reduce the number of tool calls it needs to make and ultimately the overall time for the agent to execute.

## 8 Conclusion

In this paper we showed that speculative execution of tools is an effective way to reduce the inference overheads of agents that use tools. We introduced two complementary techniques: a client-side speculative tool-calling algorithm that works with unmodified black-box APIs, and an engine-side algorithm that keeps prompts resident in the inference server by validating tool calls and ingesting tool outputs directly from a tool cache. With an analytical model we present the conditions where tool calling helps the most; we highlight the roles of tool latency, speculative model accuracy, and decode cost in determining how much speedup is achievable in practice. Our experiments show that these methods yield double-digit percent time savings per agent turn and increase throughput by hundreds of tokens per second. Speculative tools have the potential to rapidly accelerate the use of agents in tool heavy tasks, however, properly implementing speculative tool algorithms requires care to accomplish efficiently; this work provides the insights and building blocks to create effective speculative tool calling agents.

### Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy (DOE) by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 (LLNL-CONF-2014336-DRAFT). This material is based upon work supported by the DOE Office of Science, Advanced Scientific Computing Research program through solicitation DE-FOA-0003264, "Advancements in Artificial Intelligence for Science." This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility, operated under Contract No. DE-AC02-05CH11231 using NERSC award ALCC-ERCAP0034775.

## References

- <span id="page-11-3"></span>[1] Speculative Decoding - vLLM. [https://docs.vllm.](https://docs.vllm.ai/en/v0.12.0/features/spec_decode/) [ai/en/v0.12.0/features/spec\\_decode/](https://docs.vllm.ai/en/v0.12.0/features/spec_decode/) accessed on 2025-12-11.
- <span id="page-11-2"></span>[2] openai/openai-python, December 2025. original-date: 2020-10-25T23:23:54Z.
- <span id="page-11-0"></span>[3] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. *arXiv preprint arXiv:2302.01318*, 2023.
- <span id="page-11-1"></span>[4] Mingyang Chen, Haoze Sun, Tianpeng Li, Fan Yang, Hao Liang, Keer Lu, Bin Cui, Wentao Zhang, Zenan

- Zhou, and Weipeng Chen. Facilitating multi-turn function calling for llms via compositional instruction tuning. *arXiv preprint arXiv:2410.12952*, 2024.
- <span id="page-12-3"></span>[5] Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, and Pengfei Zuo. {Cost-Efficient} large language model serving for multi-turn conversations with {CachedAttention}. In *2024 USENIX Annual Technical Conference (USENIX ATC 24)*, pages 111–126, 2024.
- <span id="page-12-7"></span>[6] Yunhai Hu, Zining Liu, Zhenyuan Dong, Tianfan Peng, Bradley McDanel, and Sai Qian Zhang. Speculative Decoding and Beyond: An In-Depth Survey of Techniques, October 2025. arXiv:2502.19732 [cs].
- <span id="page-12-4"></span>[7] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- <span id="page-12-5"></span>[8] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *ICML'23*, pages 19274– 19286, Honolulu, Hawaii, USA, July 2023. JMLR.org.
- <span id="page-12-2"></span>[9] Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng, Zhenpeng Chen, Lingming Zhang, and Yiling Lou. Large language model-based agents for software engineering: A survey. *arXiv preprint arXiv:2409.02977*, 2024.
- <span id="page-12-9"></span>[10] OpenAI, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, Che Chang, Kai Chen, Mark Chen, Enoch Cheung, Aidan Clark, Dan Cook, Marat Dukhan, Casey Dvorak, Kevin Fives, Vlad Fomenko, Timur Garipov, Kristian Georgiev, Mia Glaese, Tarun Gogineni, Adam Goucher, Lukas Gross, Katia Gil Guzman, John Hallman, Jackie Hehir, Johannes Heidecke, Alec Helyar, Haitang Hu, Romain Huet, Jacob Huh, Saachi Jain, Zach Johnson, Chris Koch, Irina Kofman, Dominik Kundel, Jason Kwon, Volodymyr Kyrylov, Elaine Ya Le, Guillaume Leclerc, James Park Lennon, Scott Lessans, Mario Lezcano-Casado, Yuanzhi Li, Zhuohan Li, Ji Lin, Jordan Liss, Lily, Liu, Jiancheng Liu, Kevin Lu, Chris Lu, Zoran Martinovic, Lindsay McCallum, Josh McGrath, Scott McKinney, Aidan McLaughlin, Song Mei, Steve Mostovoy, Tong Mu, Gideon Myles, Alexander Neitz, Alex Nichol, Jakub Pachocki, Alex Paino, Dana Palmie,

- Ashley Pantuliano, Giambattista Parascandolo, Jongsoo Park, Leher Pathak, Carolina Paz, Ludovic Peran, Dmitry Pimenov, Michelle Pokrass, Elizabeth Proehl, Huida Qiu, Gaby Raila, Filippo Raso, Hongyu Ren, Kimmy Richardson, David Robinson, Bob Rotsted, Hadi Salman, Suvansh Sanjeev, Max Schwarzer, D. Sculley, Harshit Sikchi, Kendal Simon, Karan Singhal, Yang Song, Dane Stuckey, Zhiqing Sun, Philippe Tillet, Sam Toizer, Foivos Tsimpourlas, Nikhil Vyas, Eric Wallace, Xin Wang, Miles Wang, Olivia Watkins, Kevin Weil, Amy Wendling, Kevin Whinnery, Cedric Whitney, Hannah Wong, Lin Yang, Yu Yang, Michihiro Yasunaga, Kristen Ying, Wojciech Zaremba, Wenting Zhan, Cyril Zhang, Brian Zhang, Eddie Zhang, and Shengjia Zhao. gpt-oss-120b & gpt-oss-20b Model Card, August 2025. arXiv:2508.10925 [cs].
- <span id="page-12-8"></span>[11] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large Language Model Connected with Massive APIs, May 2023. arXiv:2305.15334 [cs].
- <span id="page-12-0"></span>[12] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Guoliang Li, Zhiyuan Liu, and Maosong Sun. Tool learning with foundation models. *ACM Comput. Surv.*, 57(4), December 2024.
- [13] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. *Frontiers of Computer Science*, 19(8):198343, 2025.
- <span id="page-12-1"></span>[14] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, 36:68539–68551, 2023.
- <span id="page-12-6"></span>[15] Simranjit Singh, Andreas Karatzas, Michael Fore, Iraklis Anagnostopoulos, and Dimitrios Stamoulis. An llmtool compiler for fused parallel function calling. *arXiv preprint arXiv:2405.17438*, 2024.
- <span id="page-12-10"></span>[16] Prajwal Singhania, Siddharth Singh, Lannie Dalton Hough, Akarsh Srivastava, Harshitha Menon, Charles Fredrick Jekel, and Abhinav Bhatele. LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication, November 2025. arXiv:2511.09557 [cs].

- <span id="page-13-4"></span>[17] Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, and Rui Hou. The early bird catches the leak: Unveiling timing side channels in llm serving systems. *IEEE Transactions on Information Forensics and Security*, 2025.
- <span id="page-13-2"></span>[18] vLLM. Automatic prefix caching. [https://docs.](https://docs.vllm.ai/en/stable/design/prefix_caching/) [vllm.ai/en/stable/design/prefix\\_caching/](https://docs.vllm.ai/en/stable/design/prefix_caching/).
- <span id="page-13-1"></span>[19] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. *Advances in Neural Information Processing Systems*, 37:50528–50652, 2024.
- <span id="page-13-0"></span>[20] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In *The eleventh international conference on learning representations*, 2022.
- <span id="page-13-5"></span>[21] Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Hoang, Shirley Kokane, Weiran Yao, Juntao Tan, Akshara Prabhakar, Haolin Chen, Zhiwei Liu, Yihao Feng, Tulika Awalgaonkar, Rithesh Murthy, Eric Hu, Zeyuan Chen, Ran Xu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, and Caiming Xiong. xLAM: A Family of Large Action Models to Empower AI Agent Systems, September 2024. arXiv:2409.03215 [cs].
- <span id="page-13-3"></span>[22] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. *Advances in neural information processing systems*, 37:62557–62583, 2024.