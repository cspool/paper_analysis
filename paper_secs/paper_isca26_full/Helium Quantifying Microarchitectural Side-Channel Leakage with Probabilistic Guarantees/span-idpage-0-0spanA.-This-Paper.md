# <span id="page-0-0"></span>*A. This Paper*

We present Helium, a framework for quantifying the risk of hardware side-channel leakage for arbitrary combinations of a

```
// Zero-skip optimization: MUL instruction i exhibits a fast µobs (µobs1)
if either operand is zero or a slow µobs (µobs2) otherwise.
µobs zero_skip(MUL i) :
 if(i.op1 == 0 ∨ i.op2 == 0) : return µobs1
 else : return µobs2
```

Fig. 1: Multiply instruction i can exhibit two different observable execution paths, µobs<sup>1</sup> or µobs2, as a function of its operands.

victim program, secret input, and microarchitecture, assuming an attacker that observes victim instructions' precise hardware resource usage in time and space. Helium resolves limitations of prior work by adopting: a new information-theoretic metric, called *pointwise maximal leakage* (PML) [\[79\]](#page-14-11), to express probabilistic guarantees on hardware side-channel leakage; *microarchitectural observation (*µ*obs) functions* to model how hardware side channels create instruction-level attacker observations; and *Tracer* to derive program-level attacker observation distributions using µobs functions.

*PML:* Unlike averaging metrics, PML quantifies leakage per attacker observation and enables expressing privacy guarantees as statistical properties of the leakage distribution across all attacker observations of a victim program's execution. To our knowledge, we are the first to apply PML to reason about side channels. We show how the *tail-bound guarantee* [\[79\]](#page-14-11), which expresses that high-leakage observations occur with low probability for user-selected thresholds, is especially useful for navigating side-channel security-performance trade-offs. For example, consider the zero-skip optimization, which allows arithmetic and bitwise logical instructions to "skip" execution when at least one operand is zero [\[10\]](#page-13-0), [\[52\]](#page-14-2), [\[53\]](#page-14-4), [\[97\]](#page-15-0). An attacker that observes a fast execution learns that at least one operand is zero; a slow execution reveals both operands are non-zero. Intuitively, fast executions leak substantially more information. Using PML, Helium determines whether highleakage observations occur with sufficiently low probability for a program to be deemed secure.

µ*obs functions:* Recent work shows that operanddependent hardware resource usage manifests as microarchitectural execution variability for the same (our focus, [§III-A\)](#page-2-0) or different instructions [\[48\]](#page-14-12). Based on this insight, Helium's µobs functions are a unifying formalism for succinctly encoding how different observable microarchitectural executions of an instruction, called µobs, arise as a function of one or more unsafe instruction operands due to one or more optimizations. Helium's µobs functions bear similarities to other side-channel models proposed for leakage detection [\[14\]](#page-13-22), [\[48\]](#page-14-12), [\[81\]](#page-14-0), but are more abstract, capturing minimal hardware information for leakage quantification. Fig. [1](#page-1-0) shows a µobs function for a multiply instruction impacted by the zero-skip optimization, which outputs a fast or slow µobs as a function of its operands.

*Tracer:* As a victim program executes, instructions interact with hardware side channels, exposing µobs as described by µobs functions. We call the resulting sequence of instruction-level µobs across a full program a *microarchitectural observation trace* (µtrace). We consider a strong attacker model, where the attacker can observe complete µtraces; weaker attackers may observe some non-identity function of µtraces, e.g., their execution latency ([§VIII\)](#page-11-0). Computing PML requires the distribution of program-level attacker observations given the secret input distribution. The distribution of attacker observations emerges hierarchically: the program's semantics determine the distribution of individual instruction operands; these operand distributions induce the probabilities of instruction-level µobs; and, in turn, the µobs probabilities compose to form the probabilities of the full µtraces.

Helium provides Tracer, comprised of two approaches *TracerSym* and *TracerSim*, for automatically computing the µtrace probability distributions given a victim program, secret input distribution, and set of µobs functions. TracerSym uses symbolic execution and model counting to determine exactly how many secret inputs map to each µtrace. TracerSim is a scalable alternative using Monte Carlo simulation and conservative statistical bounds to approximate µtrace probabilities. While symbolic and simulation-based techniques are standard for side-channel analysis [\[13\]](#page-13-18), [\[14\]](#page-13-22), [\[20\]](#page-13-23), [\[23\]](#page-13-24), [\[34\]](#page-13-16), [\[56\]](#page-14-13), [\[78\]](#page-14-14), [\[80\]](#page-14-15), [\[89\]](#page-15-7), [\[91\]](#page-15-8), [\[93\]](#page-15-4), [\[98\]](#page-15-5), Helium uniquely applies them to enable probabilistic side-channel leakage quantification.

Through four case studies spanning cryptographic [\[33\]](#page-13-25) and pixel-processing applications [\[38\]](#page-13-26), we assess Helium's scalability and show how Helium enables hardware and software designers to reason intuitively about the security-performance trade-offs of high-overhead side-channel mitigations.

In summary, Helium[1](#page-1-1)[2](#page-1-2) makes the following contributions:

- 1) Probabilistic Side-Channel Security: We adopt PML as a metric for quantifying hardware side-channel leakage, which enables bounding the probability of high-leakage attacker observations via the tail-bound guarantee.
- 2) µobs Functions: We introduce a formalism to encode how arbitrary hardware side channels map instruction operands to instruction-level attacker observations. Unlike prior side-channel models ([§IX\)](#page-12-0), µobs functions are designed specifically for leakage quantification.
- 3) Tracer: We provide symbolic and simulation-based analyses similar to prior work ([§IX\)](#page-12-0), but tailored to automatically compute program-level attacker observation distributions given a victim program and a set of µobs functions modeling a microarchitecture.
- 4) Case Studies: We use Helium to compute tail-bound guarantees for security-critical programs across microarchitectures modeled by different µobs functions. In one case, tolerating a 0.0003 probability of exceeding 0.0004 bits of leakage enables avoiding 2.31× overhead from a state-of-the-art software side-channel defense [\[37\]](#page-13-15).

#### II. BACKGROUND

This section gives background on side-channel attacks ([§II-A\)](#page-2-1), microarchitectural execution variability ([§II-B\)](#page-2-2), and information-theoretic leakage metrics from prior work ([§II-C\)](#page-2-3).

<span id="page-1-1"></span><sup>1</sup>Helium is used as a high-precision "tracer gas" to detect and locate leaks in underground pipes.

<span id="page-1-2"></span><sup>2</sup>https://github.com/samanthaarcher0/Helium-Artifact

#### <span id="page-2-1"></span>A. Hardware Side-Channel Attacks

Hardware side-channel attacks are often described using a telecommunications analogy [57]: a *transmitter* (an unsafe instruction in the victim program) modulates a channel (a hardware resource) in an operand-dependent manner. A *receiver* (the attacker) observes the channel modulation to infer the operand value. Channel modulations manifest as microarchitectural execution variability for one or more instructions, called *transponders* [48]. So, a receiver ultimately observes channel modulations by measuring physical aspects of a transponder's execution, e.g., its execution time [15], [44], [72], resource contention [3], [4], [49], [64], power consumption [21], [40], [58], [67], [92], and more [7], [9], [11], [39], [41], [47], [77].

Prior work classifies types of transmitters based on their microarchitectural behavior relative to transponders [48]. *Intrinsic transmitters* (our focus, §III-A) create execution variability for themselves, i.e., they are also transponders. For example, an arithmetic instruction impacted by a zero-skip optimization creates operand-dependent execution variability for itself by residing in a functional unit for either zero or nonzero consecutive cycles (Fig. 1, §I-A). Other transmitter types create execution variability for different dynamic instructions. For example, a store (transmitter) in the store buffer creates address-dependent variability for a load (transponder) that gets its data via store buffer forwarding or the L1 cache. Note, a transmitter can have multiple types.

#### <span id="page-2-2"></span>B. Characterizing Transmitter-Transponder Interactions

Recent work proposes SynthLC, an approach and tool based on formal model checking for automatically synthesizing a complete set of *leakage function* signatures from a SystemVerilog processor design [48]. Leakage functions characterize individual instances of a transponder's microarchitectural execution variability as a function of transmitter operands. Each variability instance is defined as a tuple: (*source*, {destination}), with one source control-flow state and a set of destination control-flow states, where a control-flow state is like a pipeline stage but finer-grained. During its execution, a transponder in the source control-flow state proceeds in the next cycle to exactly one of the control-flow states in the destination set.

A leakage function defines how transmitter operand values cause a transponder in the source to proceed to each of the destinations. SynthLC constructs the signature of a leakage function from RTL, identifying the source, destination set, transmitter operands, and microarchitectural state that impact the variability. SynthLC does not uncover the exact function mapping, but we expect it can be extended to do so, e.g., by leveraging symbolic simulation [55], [66] and model checking.

#### <span id="page-2-3"></span>C. Quantifying Hardware Side-Channel Leakage

Quantifying hardware side-channel leakage requires a metric to relate a victim program's secret input<sup>3</sup> distribution to an

<span id="page-2-5"></span>

| Entropy            | $H(Y) = -\sum_{y \in Y} P_Y(y) \log P_Y(y)$                                          |
|--------------------|--------------------------------------------------------------------------------------|
| Mutual information | $I(X;Y) = \sum_{x \in X, y \in Y} P_{XY}(x,y) \log \frac{P_{XY}(x,y)}{P_X(x)P_Y(y)}$ |
| Maximal leakage    | $L(X \to Y) = \log \sum_{y \in Y} \max_{x \in X} P_{Y X=x}(y)$                       |

TABLE I: Information-theoretic leakage metrics.

attacker observation distribution. Prior work uses information-theoretic metrics: Shannon entropy [68], [75], [80], [101], mutual information [13], [31], [93], [98], and maximal leakage [34], [54], [94]. Table I shows their equations, where discrete random variables X and Y represent the secret and attacker observations, respectively.

Shannon entropy denotes the average information of a random variable and is used in prior work to quantify the information across an attacker's observations. Because the observation space may depend on public or random inputs in addition to the secret, entropy cannot attribute the information in an attacker's observation to the secret input, except when a channel is deterministic with respect to the secret.

Mutual information has been used to quantify the average information leaked about a secret, X, through an attacker's observations, Y. Mutual information measures the average shared information between the secret and observation spaces for both deterministic and non-deterministic channels; in the former case, it is equal to the entropy of Y.

Maximal leakage measures the multiplicative gain of the attacker's probability of guessing a secret X given a side-channel observation Y. Specifically, maximal leakage is the average increase in the attacker's ability to guess a secret across Y for the worst-case input distribution X.

#### III. THREAT MODEL AND HELIUM OVERVIEW

This section summarizes our threat model (§III-A) and gives a broad overview of the Helium framework (§III-B).

#### <span id="page-2-0"></span>A. Threat Model

We assume that the receiver (attacker) observes each victim instruction's exact *microarchitectural execution path*, defined as a cycle-accurate partial order on the state updates the instruction induces during its execution [48]. This attacker model captures a range of attackers, including passive attackers that monitor victim instruction execution time and active attackers that contend with victim instructions for shared hardware resources [19], [34], [81], [87]. We model side-channel modulations as deterministic functions of transmitter operands to conservatively capture noise-free attacker observations; we discuss extending Helium to analyze non-deterministic (probabilistic) channels in §VIII. Although Helium conservatively considers a very powerful attacker by default, its fine-grained observation distribution can be post-processed to compute leakage for weaker attackers (§VIII).

Helium quantifies leakage due to intrinsic transmitters (§II-A), including arithmetic instructions that have been historically considered safe and thus used to process secrets in

<span id="page-2-4"></span><sup>&</sup>lt;sup>3</sup>Multiple secret inputs can be modeled as one large secret. Thus, we refer to a single secret input throughout this paper.

CT code. This focus is motivated by two considerations. First, arithmetic is ubiquitous in computations over secret data, making such optimizations both prone to leak and high cost when mitigating. Second, these transmitters are the focus of recent software mitigations that eliminate leakage for traditionally CT programs running on hardware with novel data-dependent optimizations [\[37\]](#page-13-15). These high-overhead mitigations motivate the need for principled evaluation of performance-security trade-offs. In [§VIII,](#page-11-0) we discuss how Helium can be extended to quantify leakage for other transmitter types.

We design Helium to quantify non-speculative hardware side-channel leakage. Speculative execution attacks require a distinct class of mitigations [\[24\]](#page-13-38), [\[25\]](#page-13-39), [\[96\]](#page-15-11). This scope is consistent with prior work on leakage quantification [\[13\]](#page-13-18), [\[34\]](#page-13-16), [\[36\]](#page-13-21), [\[93\]](#page-15-4), [\[98\]](#page-15-5) and recent software mitigations intended to protect against optimizations that introduce transmitters beyond the set assumed by traditional CT programs [\[37\]](#page-13-15).

#### <span id="page-3-0"></span>*B. Helium Overview*

We propose Helium, a three-part framework for quantifying the risk that a victim program leaks its secret input via hardware side channels when it runs on a given microarchitecture.

*Pointwise Maximal Leakage ([§IV\)](#page-3-1):* Helium adopts PML [\[79\]](#page-14-11), a leakage metric that enables defining probabilistic privacy guarantees of the form "observations that leak more than ϵ bits occur with probability less than δ." Prior metrics average leakage across observations ([§II-C\)](#page-2-3), masking highleakage events rather than bounding their probability. PML authors identify side-channel leakage as a potential application of the metric, but we are the first to apply it to reason about full-program leakage due to arbitrary side channels.

*Microarchitectural Observation Functions ([§V\)](#page-5-0):* Computing PML requires deriving the attacker observation distribution for a victim program running on a microarchitecture. Helium derives this distribution at the instruction-level using µobs functions, each of which defines how a transponder exhibits observably-distinct executions (µobs) as a function of one or more transmitter operands. For intrinsic transmitters (i.e., also transponders), a µobs function maps an instruction's own operands to its observable executions, such as a multiply's operands resulting in fast or slow µobs due to a zero-skip optimization in Fig. [1.](#page-1-0) Helium's µobs function abstraction is sufficient to model any timing side channel, including those that arise due to resource contention. Prior works in leakage quantification model specific channels using bespoke formalisms (e.g., for caches [\[13\]](#page-13-18), [\[27\]](#page-13-20), [\[34\]](#page-13-16), [\[36\]](#page-13-21), [\[60\]](#page-14-27), [\[68\]](#page-14-10), [\[98\]](#page-15-5) and/or control flow [\[13\]](#page-13-18), [\[36\]](#page-13-21), [\[80\]](#page-14-15), [\[98\]](#page-15-5)) that neither compose to capture multiple side channels nor generalize to new channels that we consider in [§VII.](#page-8-0)

*Program-Level Observation Analyses ([§VI\)](#page-5-1):* Computing program-level leakage requires deriving the program-level observation distribution, which is determined both by the program's semantics and the attacker's instruction-level observations. The interaction between these two factors can result in instructions ultimately exposing overlapping, disjoint, or even no portions of a secret. Prior instruction-level analyses

<span id="page-3-3"></span>

| Optimization 1       | Optimization 2       |  |  |
|----------------------|----------------------|--|--|
| if x[0] == 1:        | if x == 0:           |  |  |
| attacker observes y1 | attacker observes y1 |  |  |
| else:                | else:                |  |  |
| attacker observes y2 | attacker observes y2 |  |  |

Fig. 2: Two optimizations, each with two attacker observations, y<sup>1</sup> and y2, that leak a function of a secret input x ∈ X.

fail to capture overlapping per-instruction leakage and thus overestimate full program leakage [\[13\]](#page-13-18), [\[14\]](#page-13-22). Prior programlevel analyses avoid this pitfall but are co-designed with specific side-channel models, focusing on only cache and controlflow side channels [\[36\]](#page-13-21), [\[60\]](#page-14-27), [\[93\]](#page-15-4), [\[98\]](#page-15-5). Helium provides two complementary, general-purpose methods for determining the distribution of program-level observations.

#### <span id="page-3-1"></span>IV. POINTWISE MAXIMAL LEAKAGE: DEFINING PROBABILISTIC SIDE-CHANNEL PRIVACY GUARANTEES

We first show how existing leakage metrics may underestimate leakage ([§IV-A\)](#page-3-2). We then introduce PML ([§IV-B\)](#page-4-0), a metric that resolves this issue by capturing the leakage of individual observations, and compare it with prior metrics for a binary channel ([§IV-C\)](#page-4-1). Lastly, we discuss how PML enables probabilistic privacy guarantees for side channels ([§IV-D\)](#page-4-2).

#### <span id="page-3-2"></span>*A. Pitfalls of Averaging Leakage Metrics*

For quantitative leakage metrics to be useful, they must be both conservative and intuitive. We find that prior metrics do not fit these criteria. We illustrate this using the two most common metrics, mutual information and maximal leakage, to quantify the leakage of a single intrinsic transmitter ([§II-A\)](#page-2-1).

Assume X is a uniformly distributed 32-bit secret that is passed to an unsafe operand of an intrinsic transmitter, which exhibits observable microarchitectural execution variability as a function of this operand. The variability arises due to one of the optimizations shown in Fig. [2.](#page-3-3) The first exposes observations y<sup>1</sup> or y<sup>2</sup> depending on the operand's least significant bit. Thus, both observations leak one bit for every x ∈ X. The second exposes observation y<sup>1</sup> if the operand is zero and y<sup>2</sup> if it is non-zero. Thus, y<sup>1</sup> leaks all 32 bits of x ∈ X if x = 0, while y<sup>2</sup> leaks very little information, namely that x ̸= 0.

Table [II](#page-4-3) reports the mutual information and maximal leakage of X via each transmitter (one per optimization), where Y contains observations y<sup>1</sup> and y2, distributed according to the optimizations in Fig. [2.](#page-3-3)

Optimization 1 has mutual information of one, as exactly one bit of x ∈ X leaks for both observations y<sup>1</sup> and y2. Optimization 2 has small mutual information, because the probability of y1, i.e., the highly leaky observation, is small for a uniformly distributed X. As a result, prior work has shown that mutual information severely underestimates leakage [\[62\]](#page-14-28), [\[94\]](#page-15-6), as it masks the highly leaky observation, y1.

Both optimizations have a maximal leakage of one. Since the conditional probability is the same for both optimizations (because the channels are deterministic), the maximal leakage is log of the number of observations. Maximal leakage does

<span id="page-4-3"></span>

|                    | Optimization 1 | Optimization 2        |
|--------------------|----------------|-----------------------|
| Mutual information | 1              | $7.786 \cdot 10^{-9}$ |
| Maximal leakage    | 1              | 1                     |

TABLE II: Mutual information and maximal leakage of optimizations in Fig. 2, assuming X is a uniformly distributed 32-bit integer.

not distinguish between channels with different input distributions, so these optimizations with seemingly different leakage behavior counterintuitively have the same maximal leakage.

Both metrics provide an average leakage across all observations. Neither mutual information nor maximal leakage capture the leakage of individual observations in Optimization 2, where one observation,  $y_1$ , leaks a lot with low probability, and the other,  $y_2$ , leaks a small amount with high probability.

#### <span id="page-4-0"></span>B. A Better Leakage Metric: Pointwise Maximal Leakage

We define a leakage metric as conservative and intuitive based on what security practitioners require: the ability to distinguish between high- and low-leakage observations and characterize their probabilities. We find that pointwise maximal leakage (PML) [79], a recently proposed information-theoretic metric, satisfies this requirement.

PML quantifies the leakage of individual attacker observations and is defined as the maximum multiplicative gain of an attacker's ability to guess the secret  $x \in X$  given a single observation  $y \in Y$  [79]:

$$\ell_{P_{XY}}(X \to y) = \log \max_{x: P_X(x) > 0} \frac{P_{X|Y=y}(x)}{P_X(x)}$$

Instead of measuring the average leakage across observations like mutual information and maximal leakage, PML computes the leakage of a single observation  $y \in Y$ . We abbreviate  $\ell_{P_{XY}}(X \to y)$  to  $\ell(y)$ .

Referring back to Optimization 2 in Fig. 2, since X is uniformly distributed,  $\forall x \in X : P_X(x) = \frac{1}{2^{32}}$ . So, the PML of observation  $y_1$  is:

$$\ell(y_1) = \log \frac{1}{\frac{1}{2^{32}}} = 32$$

Intuitively, PML of 32 for  $y_1$  denotes that this observation leaks all 32 bits of the secret. The PML of  $y_2$  is:

$$\ell(y_2) = \log \frac{\frac{1}{2^{32} - 1}}{\frac{1}{2^{32}}} = 3.36 \cdot 10^{-10}$$

This small PML indicates that  $y_2$  leaks little information about the secret (only that  $x \neq 0$ ).

#### <span id="page-4-1"></span>C. Comparison of Metrics

We compare PML with mutual information and maximal leakage for a binary channel, e.g., an intrinsic transmitter that exhibits one of two observably distinct microarchitectural executions (§II-A). Assume the secret input to the transmitter, X, is a binary random variable, where P(X=0)=p and P(X=1)=1-p. The attacker observation space is a binary random variable, Y, containing two observations,  $y_1$  and  $y_2$ .

Fig. 3 plots mutual information, maximal leakage, and PML across different input distributions, i.e., for different values of p. Comparing the first two metrics, we make two observations.

<span id="page-4-4"></span>![](_page_4_Figure_17.jpeg)

Fig. 3: Comparison of leakage metrics for a binary channel with two observations,  $y_1$  and  $y_2$ , input probabilities: P(X=0)=p and P(X=1)=1-p, and arbitrary conditional probabilities:  $P(Y=y_1|X=0)=0.75$  and  $P(Y=y_1|X=1)=0.9$ .

First, maximal leakage is clearly an upper bound on mutual information, which is why prior work uses maximal leakage as a more conservative metric [34], [94]. Second, maximal leakage remains constant across different values of p. It only depends on the channel, i.e. the conditional probability distribution of observations, independent of the input distribution. Prior work cites this as a benefit, because maximal leakage may be easier to compute [34], but it prevents maximal leakage from distinguishing leakage scenarios (e.g., different instances of an intrinsic transmitter) with different input distributions.

Now, consider the PML of observations  $y_1$  and  $y_2$ . Each observation's PML can be much higher/lower than the maximal leakage and mutual information. Since both mutual information and maximal leakage fail to capture the leakage behavior of individual observations, they over- and underestimate per- observation leakage for nearly all values of p. PML robustly captures the full distribution of information leakage. Fig. 3 showcases PML's generality: it can precisely quantify leakage of individual observations under arbitrary secret input distributions and deterministic or non-deterministic side channels.

#### <span id="page-4-2"></span>D. Tail-bound Privacy Guarantee

Since the attacker observation space, Y, is a random variable with distribution  $P_Y$ , and PML is a function of y, PML can be viewed as a random variable whose distribution is induced by  $P_Y$  [79]. This critically allows us to regard privacy guarantees as statistical properties of PML. In particular, the *tail-bound guarantee* requires that the probability of PML being less than a user-defined threshold  $\epsilon$  is at least  $1 - \delta$  [79]:

$$P_Y[\ell(Y) \le \epsilon] \ge 1 - \delta$$

In other words, the likelihood of observing outcomes where PML exceeds  $\epsilon$  must be less than  $\delta$ , partitioning observations as "good" (i.e., low PML) or "bad" (i.e., high PML).

Compared to prior metrics, we expect the tail-bound guarantee to better align with how security practitioners reason about security: programs must leak very little with very high probability. Nevertheless, regarding leakage as a distribution rather than a single aggregate value (e.g., average) gives the programmer flexibility to choose from a variety of security guarantees depending on their application's requirements.

```
// Zero/All-ones-skip optimization: If either operand of a bitwise AND
instruction i is zero or all ones, \muobs<sub>1</sub> occurs, else \muobs<sub>2</sub> occurs.
\muobs zero_all_ones_skip(AND i) :
 if(i.op1 == 0 \lor i.op2 == 0 \lor i.op1 == 0 xFF...F \lor
  i.op2 == 0xFF...F) : return \muobs<sub>1</sub>
 else : return \muobs<sub>2</sub>
// Zero/One-skip optimization: If either operand of a multiply instruction
i is zero or one, \muobs<sub>1</sub> occurs, else \muobs<sub>2</sub> occurs.
\muobs zero_one_skip(MUL i) :
 if(i.op1 == 0 \lor i.op2 == 0 \lor i.op1 == 1 \lor i.op2 == 1) :
   return \muobs<sub>1</sub>
 else : return \muobs_2
// Digit-Serial optimization: Multiply instruction i exhibits 4 \muobs depend-
ing on the upper-most bytes of operand 2 being zero.
\muobs digit_serial(MUL i) :
 if(i.op2[8:31] == 0) : return \ \mu obs_1
 if (i.op2[16:31] == 0): return \muobs<sub>2</sub>
 if(i.op2[24:31] == 0) : return \mu obs_3
 else : return \muobs<sub>4</sub>
// Bit-Serial optimization: Division instruction i exhibits 65 \muobs depend-
ing on the difference in the two operands' number of leading zeros.
\muobs bit_serial(DIV i) :
 if (i.op2 == 0 \lor (i.op1 < i.op2)) : return \mu obs_1
 else:
   for d in range(64):
    if((i.op1 >> d) \ge i.op2) \land ((i.op1 >> (d+1)) < i.op2) :
      return \muobs_{d+2}
```

Fig. 4:  $\mu$ obs functions for computation simplification optimizations, including variants of zero-skip optimizations, digit-serial multiplication, and bit-serial division [99].

