# A APPENDIX

### <span id="page-13-0"></span>A.1 MORE PRELIMINARY VALIDATION CASES AND FRAMEWORK

As shown in [A.1,](#page-13-0) the performance gap between models of different scales is clearly illustrated by a case study comparing Qwen2.5-VL-32B and Qwen2.5-VL-7B. When prompted for visual grounding, the 32B model showed a solid grasp of the input: it not only captured the obvious features but also inferred hidden geometric properties, correctly identifying △AOB as a right-angled triangle—an insight that guided its reasoning effectively. In contrast, the 7B model misinterpreted the diagram, hallucinating an extra property and wrongly describing △AOB as an isosceles right-angled triangle. This flawed perception disrupted its reasoning and led to an incorrect solution.

This example provides direct evidence for our central claim: the effectiveness of visual grounding prompts depends on the model's inherent perceptual ability. For a stronger model, prompting helps unlock deeper understanding and is as effective as providing external annotations. For a weaker model, however, it can introduce errors and degrade performance.

Verification framework (Fig. [2](#page-11-0)*case visual1*)). Figure [2](#page-11-0) summarizes our verification pipeline across datasets, perception conditions, input variants, and the evaluation protocol. We benchmark two MLLMs under three visual settings—*explicit visual notes* (appending curated annotations to inputs), *structured visual grounding* (forcing a description before reasoning), and *implicit visual grounding* (light prompt to "look carefully")—and two visual–textual settings (structured vs. implicit), using full datasets except for a 600-instance, proportionally sampled subset of OlympiaBench. Accuracy is reported with ablations and comparative analyses to isolate the contribution of perception strategies. This setup operationalizes our central question: how explicit versus implicit perceptual grounding modulates downstream reasoning across tasks.

Visual perception case (Fig. [4,](#page-12-0) *case visual2*). In the semicircle–intersection construction, models must decompose the figure into isosceles sub-triangles centered at the semicircle midpoints (△AP R and △DQR) and compose ∠ARD from base angles and the given ∠P RQ. The larger model reliably enumerates the right substructures and completes the angle composition, whereas the smaller model tends to provide verbose but incomplete reasoning. This gap exemplifies our empirical finding: explicit visual notes help both models, but structured prompting can burden weaker perception and lead to degraded solutions—highlighting the capacity dependence of perception prompting observed in our broader study.

Textual perception case (Fig. [5,](#page-12-1) *case text*). This geometry example illustrates how textual perception stabilizes reasoning. The problem asks for the length of XY in an isosceles triangle ABC with AB=BC=25 and AC=30, where a circle with diameter BC intersects AB and AC at X and Y . Without textual perception, the model introduces spurious assumptions and predicts an incorrect value; with textual perception (making the diameter–right-angle property explicit and driving similarity/power-of-point reasoning), the model recovers the correct answer XY =15, demonstrating that lightweight textual cues can suppress hallucinations and guide the correct derivation.

#### A.2 DATASET CLEANING AND OPTIMIZATION PIPELINE FOR SFT

This appendix documents the automated cleaning and optimization pipeline used to produce a highquality variant of the SFT dataset. The pipeline is modular and deterministic (given fixed random seeds and LLM versions), and it emphasizes (1) converting visual inputs into a single objective representation, (2) re-deriving Chain-of-Thought (CoT) reasoning from that representation, and (3) scoring and selecting samples by multi-dimensional quality metrics.

### A.2.1 OVERVIEW

Given a raw dataset

$$\mathcal{D} = \{s_i\}_{i=1}^N$$
,  $s = (\mathrm{id}, \mathrm{image}, \mathrm{question}, \mathrm{original}, \mathrm{cot}, \mathrm{original}, \mathrm{answer})$ 

the pipeline transforms each sample into an enriched record:

s˜ = (id, formal description, cot thinking, final answer, quality metrics, metadata)

Only samples with an overall quality score above a configurable threshold τ are retained for the final cleaned dataset; an optional intelligent sampling step can further select a representative subset.

#### A.2.2 STEP-BY-STEP PIPELINE

#### Step 1: Image analysis and formal description generation

The goal is a single, objective, machine-readable representation of image content (the *formal image description*).

- 1. VLM dense captioning: call a visual-language model (e.g., gpt-4o with image input or an analogous VLM) to generate a dense caption describing layout, objects, and relationships.
- 2. Object detection: run Grounding DINO to output object classes and bounding boxes.
- 3. OCR: run EasyOCR. The OCR subroutine implements error-handling and fallback strategies to avoid pipeline interruption.
- 4. Merging the dense caption, detection outputs and OCR text into a structured canonical text format. This description is the only image-derived input used in later CoT reconstruction to ensure objectivity.

#### Step 2: Chain-of-Thought restructuring

- Input: question and formal description.
- Prompt an LLM to produce a new reasoning trace under the explicit instruction: "Use only the provided formal image description and the question. Do not consult the original CoT or external knowledge beyond the description."
- This re-derivation corrects hallucinations in original CoTs and enforces that reasoning follows observable facts.

#### Step 3: Quality assessment and data filtering

Quality checking uses multi-dimensional metrics (1) does the formal description correctly reflect the image (consistency checks, spot-check prompts to the VLM)? (2) logical clarity and stepwise correctness of reasoning trace. (3) agreement between reasoning trace and final answer. (4) absence of unsupported claims or hallucinations. And a learned/LLM-based scorer.Return per-dimension scores in [0, 1]. A weighted sum can be used:

```
overall score = wf sf + wcsc + wasa + wmsm,
```

with w<sup>f</sup> + w<sup>c</sup> + w<sup>a</sup> + w<sup>m</sup> = 1. Values of w· are experiment hyperparameters (typical: w<sup>f</sup> = 0.30, w<sup>c</sup> = 0.35, w<sup>a</sup> = 0.30, w<sup>m</sup> = 0.05).

#### A.2.3 PIPELINE PSEUDOCODE

### A.2.4 OUTPUT SCHEMA (EXAMPLE)

Each processed record is stored as JSON/JSONL with the following example structure:

```
{
  "id": "sample_0001",
  "image_path": "images/0001.jpg",
  "question": "What color is the mug on the left?",
  "formal_description": "<structured text listing objects, positions, OCR, etc.>",
  "cot_thinking": "<reconstructed chain-of-thought>",
  "final_answer": "left mug is white",
  "quality_metrics": {
    "formal_score": 0.94,
    "cot_score": 0.88,
    "answer_score": 0.97,
    "overall_score": 0.92
  },
```

#### Algorithm 1 SFT Dataset Cleaning & Optimization Pipeline

```
Require: raw dataset D, configuration C
 1: Initialize P ← ∅ (passed set), F ← ∅ (failed set)
 2: for all sample s from SFTDataLoader(D,C) do
 3: formal ← ImageFormalDescriptionGenerator(s.image, s.question, C)
 4: cot ← CoTRestructurer(s.question, formal, C)
 5: answer ← AnswerExtractor(cot, s.question, C)
 6: metrics ← QualityChecker(formal, cot, answer, C)
 7: if metrics.overall score ≥ C.min score then
 8: add enriched record s˜ (including metrics) to P
 9: else
10: add s˜ (marked failed) to F
11: end if
12: end for
13: if C.enable sampling then
14: S ← DataSampler(P, C)
15: return S
16: else
17: return P
18: end if
```

```
"passed_quality_check": true,
  "metadata": {
    "detected_objects": [
      {"class":"mug","bbox":[x1,y1,x2,y2]}
    ],
    "ocr_text": "",
    "timestamps": {...}
  }
}
```

