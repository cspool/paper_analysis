import { FileLoopStore } from "./store.ts";
import type {
  ExperimentGoalRecord,
  ExperimentGoalResult,
  NegativeExperimentIndex,
  NegativeExperimentIndexEntry,
  ReviewResult,
  StateFile,
} from "./types.ts";

export const NEGATIVE_EXPERIMENT_INDEX_REF =
  "observations/negative_experiment_index.json" as const;

export function rebuildNegativeExperimentIndex(
  store: FileLoopStore,
  state: StateFile,
): NegativeExperimentIndex {
  const index = buildNegativeExperimentIndex(store, state);
  store.writeJson(NEGATIVE_EXPERIMENT_INDEX_REF, index);
  return index;
}

export function buildNegativeExperimentIndex(
  store: FileLoopStore,
  state: StateFile,
): NegativeExperimentIndex {
  const anchorCounts = new Map<string, number>();
  const directionCounts = new Map<string, number>();
  const entries: NegativeExperimentIndexEntry[] = [];

  for (const experimentRef of store.experimentRefs().sort()) {
    const record = store.readExperiment(experimentRef);
    if (!store.exists(record.resultRef)) continue;
    const result = store.readJson<ExperimentGoalResult>(record.resultRef);
    anchorCounts.set(
      result.anchorWork,
      (anchorCounts.get(result.anchorWork) ?? 0) + 1,
    );
    if (result.directionWork) {
      directionCounts.set(
        result.directionWork,
        (directionCounts.get(result.directionWork) ?? 0) + 1,
      );
    }

    // Goal completion is not a scientific verdict. Only a completed EXP that
    // a later Reviewer rejected enters the negative navigation layer.
    if (result.goalStatus !== "complete") continue;
    const reviewRef = explicitOrInferredExperimentReview(
      store,
      record,
      result,
    );
    if (!reviewRef) continue;
    const review = store.readJson<ReviewResult>(reviewRef);
    if (review.reviewVerdict !== "REJECT") continue;
    entries.push({
      anchorWork: result.anchorWork,
      directionWork: result.directionWork,
      experimentResultRef: record.resultRef,
      reviewRef,
      reviewVerdict: "REJECT",
    });
  }

  return {
    generatedAt: new Date().toISOString(),
    sourceStateRevision: state.revision,
    counts: {
      run: store.experimentRefs().length,
      anchors: [...anchorCounts.entries()].map(([workRef, count]) => ({
        workRef,
        count,
      })),
      directions: [...directionCounts.entries()].map(([workRef, count]) => ({
        workRef,
        count,
      })),
    },
    entries,
  };
}

export function negativeExperimentHistoryForTask(
  store: FileLoopStore,
  state: StateFile,
  anchorWork: string | null,
): NegativeExperimentIndex {
  const full = buildNegativeExperimentIndex(store, state);
  if (!anchorWork) return full;
  const relatedAnchorWorks = anchorRevisionWorkRefs(store, anchorWork);
  const entries = full.entries.filter((entry) =>
    relatedAnchorWorks.has(entry.anchorWork)
  );
  const relatedDirections = new Set<string>();
  for (const experimentRef of store.experimentRefs()) {
    const record = store.readExperiment(experimentRef);
    if (!store.exists(record.resultRef)) continue;
    const result = store.readJson<ExperimentGoalResult>(record.resultRef);
    if (
      relatedAnchorWorks.has(result.anchorWork) &&
      result.directionWork
    ) {
      relatedDirections.add(result.directionWork);
    }
  }
  return {
    ...full,
    counts: {
      run: full.counts.run,
      anchors: full.counts.anchors.filter((item) =>
        relatedAnchorWorks.has(item.workRef)
      ),
      directions: full.counts.directions.filter((item) =>
        relatedDirections.has(item.workRef)
      ),
    },
    entries,
  };
}

function anchorRevisionWorkRefs(
  store: FileLoopStore,
  workRef: string,
): Set<string> {
  for (const anchor of Object.values(store.readObjects().anchors)) {
    const refs = Object.values(anchor.revisions).map((revision) =>
      revision.workRef
    );
    if (refs.includes(workRef)) return new Set(refs);
  }
  return new Set([workRef]);
}

export function experimentRecordForResult(
  store: FileLoopStore,
  resultRef: string | null,
): { ref: string; record: ExperimentGoalRecord } | null {
  if (!resultRef) return null;
  for (const ref of store.experimentRefs()) {
    const record = store.readExperiment(ref);
    if (record.resultRef === resultRef && store.exists(resultRef)) {
      return { ref, record };
    }
  }
  return null;
}

export function experimentReviewRef(
  store: FileLoopStore,
  record: ExperimentGoalRecord,
): string | null {
  if (!store.exists(record.resultRef)) return null;
  return explicitOrInferredExperimentReview(
    store,
    record,
    store.readJson<ExperimentGoalResult>(record.resultRef),
  );
}

function explicitOrInferredExperimentReview(
  store: FileLoopStore,
  record: ExperimentGoalRecord,
  result: ExperimentGoalResult,
): string | null {
  if (record.reviewRef && store.exists(record.reviewRef)) {
    return record.reviewRef;
  }
  if (!record.completedAt) return null;
  const workRef = result.directionWork ?? result.anchorWork;
  const objects = store.readObjects();
  const candidates = [
    ...Object.values(objects.anchors).flatMap((entry) =>
      Object.values(entry.revisions)
    ),
    ...Object.values(objects.directions).flatMap((entry) =>
      Object.values(entry.revisions)
    ),
  ].filter((revision) => revision.workRef === workRef);
  for (const revision of candidates) {
    const reviewTurn = store.turnRefs()
      .map((turnRef) => store.readTurn(turnRef))
      .find((turn) => turn.resultRef === revision.reviewRef);
    if (
      reviewTurn?.completedAt &&
      reviewTurn.completedAt > record.completedAt &&
      store.exists(revision.reviewRef)
    ) {
      return revision.reviewRef;
    }
  }
  return null;
}
