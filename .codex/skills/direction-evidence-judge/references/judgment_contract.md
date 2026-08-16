# Evidence judgment contract

## 1. Establish binding and exit path

Identify the review request, contract revision/hash, cycle binding, frozen Direction, Lab result,
and artifact refs. Confirm which evidence is new. A missing or mismatched binding makes inference
invalid even when measurements look favorable.

Identify whether Lab claims:

- ordinary atomic completion;
- one explicit stop condition;
- contract conflict or forbidden weakening;
- insufficient time/resources;
- invalid implementation or measurement.

## 2. Audit early stop before completeness

For an explicit stop condition:

1. verify that it appears in the frozen contract;
2. verify its supporting artifacts, hashes, samples, mechanism trace, and statistics;
3. verify that partial shards were excluded;
4. verify that Lab stopped before prohibited downstream work;
5. apply only the conclusion boundary defined for that exit.

Do not mark a result incomplete merely because confirmation or performance was omitted after a valid
terminal stop. Do mark it invalid or inconclusive when the claimed stop is unsupported, misbound, or
observed only in an invalid carrier.

## 3. Audit implementation and causal fidelity

Determine whether Lab executed:

- a correct parent or closest-method baseline;
- the strongest simple comparator with fair legal calibration;
- the declared unique change on the same carrier;
- required component controls;
- correctness, quality, throughput, fairness, and resource guards.

Inspect versions, patches, commands, configs, raw outputs, traces, and failure logs as needed. Do not
require complete paper reproduction when the contract defines an honest substitute.

Verify that the trigger occurred and action/state/execution differed at the intended interface. A
proxy or simulator supports only the mechanism it encodes; it cannot establish real task
heterogeneity, model quality, GPU performance, or external validity unless measured.

## 4. Audit comparison and uncertainty

Check calibration/confirmation isolation, equal resources and inputs, unique-change pairing,
metrics, and statistics. Treat behavioral equivalence or dominance by a simpler calibrated baseline
as scoped negative evidence.

Report performance scope exactly:

- whether latency includes preprocessing;
- closed-loop or open-loop arrivals and concurrency;
- batch/continuous batching;
- hardware, clocks, precision, warmup, and cache state;
- actual model, data, and trace range.

Do not accept “E2E” when preprocessing is excluded or “throughput” when only serial service rate was
measured. Do not interpret zero observed differences in a small sample as zero population
uncertainty. Require suitable finite-sample bounds, power/sample rationale, and conservative handling
of multiple tasks or candidates when they affect the claim.

## 5. Classify narrowly

- `VALID_POSITIVE`: the executed atomic comparison supports incremental value within the declared
  scope and guards.
- `VALID_NEGATIVE`: a fair valid comparison establishes equivalence, dominance, a predeclared core
  failure, or absence of gain within the exact tested scope.
- `INCONCLUSIVE`: the result is interpretable but lacks power, coverage, trigger exposure, or a
  decisive comparator.
- `INVALID`: binding, implementation, carrier fidelity, baseline, attribution, measurement, or
  statistics prevent inference.

A calibrated policy collapsing to one constant can be valid negative evidence for that exact policy
definition, candidate domain, data, model, thresholds, and statistics. It does not reject every
policy in the mechanism family. Installation failure, unavailable hardware, or broken execution is
not valid negative evidence.

## 6. Select actual evidence scope

Use the narrowest demonstrated scope:

- `DESIGN_AUDIT_ONLY`;
- `WEAKENED_PROXY_MECHANISM`;
- `LOCAL_SINGLE_GPU_PERFORMANCE`;
- `SIMULATED_HARDWARE_MECHANISM`;
- `PAPER_EXTERNAL_VALIDITY`.

Do not promote scope from intended conditions or source-paper evidence.

## 7. Output

Return one JSON object using the Script template. State the main uncertainty that can still change
the Direction conclusion, or `NONE`. Do not add scheduling or global completion fields.
