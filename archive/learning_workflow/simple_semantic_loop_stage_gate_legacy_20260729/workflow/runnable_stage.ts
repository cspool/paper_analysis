import type { StageNode, WorkflowPlan } from "../contracts/index.ts";

export function calculateRunnableStages(plan: WorkflowPlan): StageNode[] {
  const nodes = new Map(plan.stageNodes.map((node) => [node.stageId, node]));
  const incoming = new Map<string, typeof plan.dependencies>();
  for (const dependency of plan.dependencies) {
    incoming.set(dependency.successorStageId, [
      ...(incoming.get(dependency.successorStageId) ?? []),
      dependency,
    ]);
  }
  return plan.stageNodes
    .filter((node) => ["frozen", "runnable"].includes(node.lifecycle))
    .filter((node) =>
      (incoming.get(node.stageId) ?? []).every((edge) => {
        const predecessor = nodes.get(edge.predecessorStageId);
        if (!predecessor) return false;
        return edge.kind === "requires_consumed"
          ? predecessor.lifecycle === "consumed"
          : ["committed", "consumed"].includes(predecessor.lifecycle);
      }),
    )
    .sort(
      (left, right) =>
        left.createdAtSnapshotVersion - right.createdAtSnapshotVersion ||
        left.stageId.localeCompare(right.stageId),
    );
}

export function chooseDeterministicallyEquivalentStage(
  stages: readonly StageNode[],
  equivalenceKey: (stage: StageNode) => string,
): StageNode | null {
  if (stages.length === 0) return null;
  const keys = new Set(stages.map(equivalenceKey));
  if (keys.size !== 1) return null;
  return [...stages].sort(
    (left, right) =>
      left.createdAtSnapshotVersion - right.createdAtSnapshotVersion ||
      left.stageId.localeCompare(right.stageId),
  )[0]!;
}

