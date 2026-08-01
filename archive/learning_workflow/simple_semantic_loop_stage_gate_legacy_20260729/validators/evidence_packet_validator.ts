import type {
  EvidencePacketEnvelope,
  EvidenceReaderTaskEnvelope,
  ValidationReport,
} from "../contracts/index.ts";
import { DIMENSION_PATHS } from "./domain_validator.ts";
import {
  addError,
  emptyReport,
  mergeReports,
  validateSchema,
} from "./schema_validator.ts";
import {
  validatePayloadEnvelopeBinding,
  type ExpectedTurnBinding,
} from "./envelope_validator.ts";

export function validateEvidencePacket(
  result: EvidencePacketEnvelope,
  task: EvidenceReaderTaskEnvelope,
): ValidationReport {
  const schema = validateSchema("EVIDENCE_PACKET", result);
  if (!schema.valid) return schema;
  const expected: ExpectedTurnBinding = {
    role: "evidence_reader",
    task,
    stateBinding: task.stateBinding,
    inputHash: task.inputHash,
  };
  const packet = result.payload;
  const need = task.payload.searchNeed;
  const budget = task.payload.budget.evidenceRead!;
  const report = mergeReports(
    schema,
    validatePayloadEnvelopeBinding(result, expected),
  );

  if (packet.needId !== need.needId || packet.needRevision !== need.revision) {
    addError(
      report,
      "evidence.need_binding",
      "/payload/needId",
      "EvidencePacket must bind the supplied SearchNeed revision",
    );
  }

  if (packet.searches.length > budget.maxSearchToolCalls) {
    addError(
      report,
      "evidence.search_call_budget",
      "/payload/searches",
      "pagination and first-page calls together exceed maxSearchToolCalls",
    );
  }
  if (
    packet.searches.length + packet.contextsRead.length >
    task.payload.budget.maxToolCalls
  ) {
    addError(
      report,
      "evidence.total_tool_budget",
      "/payload",
      "search and context-read events exceed maxToolCalls",
    );
  }
  if (packet.hitsConsidered.length > budget.maxHitsConsidered) {
    addError(
      report,
      "evidence.hit_budget",
      "/payload/hitsConsidered",
      "hit budget exceeded",
    );
  }
  const selectedHits = packet.hitsConsidered.filter((hit) => hit.selected);
  if (selectedHits.length > budget.maxSelectedSources) {
    addError(
      report,
      "evidence.selected_source_budget",
      "/payload/hitsConsidered",
      "selected source budget exceeded",
    );
  }
  if (packet.contextsRead.length > budget.maxContextsRead) {
    addError(
      report,
      "evidence.context_budget",
      "/payload/contextsRead",
      "context read budget exceeded",
    );
  }

  validateSearchLedger(result, task, report);
  validateHitAndContextLedger(result, report);
  validateFindings(result, task, report);
  validateConclusion(result, task, report);
  return report;
}

function validateSearchLedger(
  result: EvidencePacketEnvelope,
  task: EvidenceReaderTaskEnvelope,
  report: ValidationReport,
): void {
  const searches = result.payload.searches;
  const need = task.payload.searchNeed;
  let lastSequence = 0;
  let lastLevel = 0;
  let lastSearch: (typeof searches)[number] | null = null;
  const distinctLevels = new Set<number>();
  const logicalQueries = new Set(
    task.payload.previousQueries.map((item) => normalizeQuery(item.query)),
  );

  searches.forEach((search, index) => {
    if (search.sequence <= lastSequence) {
      addError(
        report,
        "evidence.sequence_not_monotonic",
        `/payload/searches/${index}/sequence`,
        "search sequence must be globally increasing",
      );
    }
    lastSequence = search.sequence;
    distinctLevels.add(search.logicalQueryLevel);
    if (
      search.logicalQueryLevel < lastLevel ||
      search.logicalQueryLevel > lastLevel + 1
    ) {
      addError(
        report,
        "evidence.query_level_order",
        `/payload/searches/${index}/logicalQueryLevel`,
        "Q1-Q3 levels must be global, monotonic, and cannot skip",
      );
    }
    if (search.logicalQueryLevel > lastLevel) {
      lastLevel = search.logicalQueryLevel;
      if (search.page !== 1 || search.cursorUsed !== null) {
        addError(
          report,
          "evidence.first_page_cursor",
          `/payload/searches/${index}`,
          "a new logical query begins at page 1 without a cursor",
        );
      }
      const normalized = normalizeQuery(search.query);
      if (logicalQueries.has(normalized)) {
        addError(
          report,
          "evidence.duplicate_logical_query",
          `/payload/searches/${index}/query`,
          "a new logical query must not repeat this or a prior attempt",
        );
      }
      logicalQueries.add(normalized);
      validateQueryLevelTerms(search, task, index, report);
    } else {
      if (
        !lastSearch ||
        search.logicalQueryLevel !== lastSearch.logicalQueryLevel ||
        search.query !== lastSearch.query ||
        search.dimension !== lastSearch.dimension ||
        search.page !== lastSearch.page + 1 ||
        search.cursorUsed === null ||
        search.cursorUsed !== lastSearch.nextCursor ||
        !sameTermLedger(search.terms, lastSearch.terms)
      ) {
        addError(
          report,
          "evidence.pagination_binding",
          `/payload/searches/${index}`,
          "pagination must preserve Q-level/query/dimension and use the exact prior opaque cursor",
        );
      }
    }
    if (index === 0 && search.logicalQueryLevel !== 1) {
      addError(
        report,
        "evidence.q1_required",
        `/payload/searches/${index}/logicalQueryLevel`,
        "the first logical query must be Q1",
      );
    }
    if (
      search.logicalQueryLevel === 1 &&
      search.dimension !== need.primaryDimension
    ) {
      addError(
        report,
        "evidence.q1_primary",
        `/payload/searches/${index}/dimension`,
        "Q1 must use the primary dimension",
      );
    }
    if (
      search.dimension !== need.primaryDimension &&
      search.dimension !== need.auxiliaryDimension
    ) {
      addError(
        report,
        "evidence.dimension_not_frozen",
        `/payload/searches/${index}/dimension`,
        "search dimension is outside the frozen primary/auxiliary pair",
      );
    }
    const expectedPathFilter = `path:${DIMENSION_PATHS[search.dimension]}`;
    const pathTokens = search.query.match(/(?:^|\s)path:[^\s]+/g) ?? [];
    if (
      search.pathFilter !== expectedPathFilter ||
      pathTokens.length !== 1 ||
      !search.query.includes(expectedPathFilter)
    ) {
      addError(
        report,
        "evidence.path_filter",
        `/payload/searches/${index}/query`,
        "query must contain exactly one dimension-matching path filter",
      );
    }
    if (search.toolCallIndex !== index + 1) {
      addError(
        report,
        "evidence.tool_call_index",
        `/payload/searches/${index}/toolCallIndex`,
        "search tool-call indexes must be contiguous",
      );
    }
    const previousCumulative =
      lastSearch &&
      search.logicalQueryLevel === lastSearch.logicalQueryLevel
        ? lastSearch.cumulativeHitCount
        : 0;
    if (
      search.cumulativeHitCount !==
      previousCumulative + search.pageHitCount
    ) {
      addError(
        report,
        "evidence.cumulative_hit_count",
        `/payload/searches/${index}/cumulativeHitCount`,
        "cumulative hit count must equal prior pages plus this page",
      );
    }
    if (
      lastSearch?.stopReason === "success_criteria_met"
    ) {
      addError(
        report,
        "evidence.search_after_success",
        `/payload/searches/${index}`,
        "no search is legal after success criteria were met",
      );
    }
    lastSearch = search;
  });

  if (
    distinctLevels.size >
    task.payload.budget.evidenceRead!.maxLogicalQueries
  ) {
    addError(
      report,
      "evidence.logical_query_budget",
      "/payload/searches",
      "global Q1-Q3 logical query budget exceeded",
    );
  }
}

function validateHitAndContextLedger(
  result: EvidencePacketEnvelope,
  report: ValidationReport,
): void {
  const searches = new Set(result.payload.searches.map((search) => search.searchId));
  const hits = new Map(
    result.payload.hitsConsidered.map((hit) => [hit.hitId, hit] as const),
  );
  result.payload.hitsConsidered.forEach((hit, index) => {
    if (!searches.has(hit.searchId)) {
      addError(
        report,
        "evidence.hit_search_ref",
        `/payload/hitsConsidered/${index}/searchId`,
        "hit must reference a search in this packet",
      );
    }
    const sourceSearch = result.payload.searches.find(
      (search) => search.searchId === hit.searchId,
    );
    if (sourceSearch && hit.sequence <= sourceSearch.sequence) {
      addError(
        report,
        "evidence.hit_sequence",
        `/payload/hitsConsidered/${index}/sequence`,
        "hit consideration must occur after its search event",
      );
    }
  });

  let lastContextSequence = 0;
  result.payload.contextsRead.forEach((context, index) => {
    const hit = hits.get(context.hitId);
    if (!hit?.selected) {
      addError(
        report,
        "evidence.context_unselected_hit",
        `/payload/contextsRead/${index}/hitId`,
        "deep reads are allowed only for selected hits",
      );
    } else if (
      context.path !== hit.path ||
      context.sourceFamily !== hit.sourceFamily
    ) {
      addError(
        report,
        "evidence.context_hit_binding",
        `/payload/contextsRead/${index}`,
        "context path/source family must match its selected hit",
      );
    }
    if (
      context.sequence <= lastContextSequence ||
      (hit !== undefined && context.sequence <= hit.sequence)
    ) {
      addError(
        report,
        "evidence.context_sequence",
        `/payload/contextsRead/${index}/sequence`,
        "context reads must be ordered and occur after their selected hit",
      );
    }
    lastContextSequence = context.sequence;
    if (context.format === "section") {
      if (!context.sectionTarget || !context.heading) {
        addError(
          report,
          "evidence.section_target",
          `/payload/contextsRead/${index}`,
          "section read requires a real heading target",
        );
      }
    } else if (context.sectionTarget !== null) {
      addError(
        report,
        "evidence.section_target_for_format",
        `/payload/contextsRead/${index}/sectionTarget`,
        "sectionTarget is only legal for format=section",
      );
    }
  });
  const allSequences = [
    ...result.payload.searches.map((item) => item.sequence),
    ...result.payload.hitsConsidered.map((item) => item.sequence),
    ...result.payload.contextsRead.map((item) => item.sequence),
  ];
  if (new Set(allSequences).size !== allSequences.length) {
    addError(
      report,
      "evidence.shared_sequence_collision",
      "/payload",
      "search, hit, and context events must share one collision-free sequence",
    );
  }

  for (const hit of hits.values()) {
    if (!hit.selected) continue;
    const contexts = result.payload.contextsRead.filter(
      (context) => context.hitId === hit.hitId,
    );
    if (
      !contexts.some((context) => context.format === "document-map") ||
      !contexts.some((context) => context.format !== "document-map")
    ) {
      addError(
        report,
        "evidence.selected_hit_not_deep_read",
        "/payload/contextsRead",
        `selected hit ${hit.hitId} requires document-map and one content-bearing deep read`,
      );
    }
  }
}

function validateFindings(
  result: EvidencePacketEnvelope,
  task: EvidenceReaderTaskEnvelope,
  report: ValidationReport,
): void {
  const contextsByUnit = new Map(
    result.payload.contextsRead.map((context) => [
      `${context.path}\0${context.sourceUnitId}`,
      context,
    ]),
  );
  const findingIds = new Set(
    result.payload.findings.map((finding) => finding.evidenceId),
  );
  result.payload.findings.forEach((finding, index) => {
    const context = contextsByUnit.get(
      `${finding.sourcePath}\0${finding.sourceUnitId}`,
    );
    if (!context) {
      addError(
        report,
        "evidence.finding_context_ref",
        `/payload/findings/${index}`,
        "finding must reference an actual deep-read context from this Turn",
      );
    } else if (
      context.format === "document-map" ||
      !context.exactContext.includes(finding.quoteOrExactContext) ||
      context.sourceFamily !== finding.sourceFamily
    ) {
      addError(
        report,
        "evidence.finding_exact_context",
        `/payload/findings/${index}/quoteOrExactContext`,
        "finding context must be a continuous substring of the cited deep read",
      );
    }
    if (
      (finding.directness === "direct" &&
        finding.attribution !== "source_report") ||
      (finding.directness === "inferred" &&
        finding.attribution !== "workflow_inference")
    ) {
      addError(
        report,
        "evidence.directness_attribution",
        `/payload/findings/${index}`,
        "direct findings are source reports; inferred findings are explicit workflow inferences",
      );
    }
    if (
      task.payload.consumedSourceUnitIds.includes(finding.sourceUnitId)
    ) {
      addError(
        report,
        "evidence.consumed_source_unit",
        `/payload/findings/${index}/sourceUnitId`,
        "a finding cannot reuse a source unit frozen as consumed",
      );
    }
  });
  result.payload.contradictions.forEach((contradiction, index) => {
    contradiction.findingIds.forEach((findingId) => {
      if (!findingIds.has(findingId)) {
        addError(
          report,
          "evidence.contradiction_finding_ref",
          `/payload/contradictions/${index}/findingIds`,
          "contradiction must cite a complete finding in this packet",
        );
      }
    });
  });

  const taskTerms = new Set([
    ...task.payload.searchNeed.technicalObjects,
    ...task.payload.searchNeed.knownTerms,
    ...task.payload.searchNeed.scenarioTerms,
    ...task.payload.searchNeed.performanceRelations,
    ...task.payload.searchNeed.evidenceIntentTerms,
    ...task.payload.searchNeed.synonymGroups.flat(),
  ]);
  const contexts = new Map(
    result.payload.contextsRead.map((context) => [context.contextId, context]),
  );
  result.payload.searches.forEach((search, searchIndex) => {
    search.terms.forEach((term, termIndex) => {
      if (term.source === "task" && !taskTerms.has(term.term)) {
        addError(
          report,
          "evidence.term_not_in_task",
          `/payload/searches/${searchIndex}/terms/${termIndex}`,
          "task-sourced term is not present in the frozen task vocabulary",
        );
      }
      if (
        !search.query.toLocaleLowerCase().includes(
          term.term.toLocaleLowerCase(),
        )
      ) {
        addError(
          report,
          "evidence.term_missing_from_query",
          `/payload/searches/${searchIndex}/terms/${termIndex}/term`,
          "every term ledger entry must occur in the executed query",
        );
      }
      if (term.source === "context") {
        const source = contexts.get(term.sourceRef);
        if (
          !source ||
          source.sequence >= search.sequence ||
          !source.exactContext.includes(term.term)
        ) {
          addError(
            report,
            "evidence.term_context_provenance",
            `/payload/searches/${searchIndex}/terms/${termIndex}`,
            "context-sourced term must occur in an earlier deep-read context",
          );
        }
      }
    });
  });
}

function validateConclusion(
  result: EvidencePacketEnvelope,
  task: EvidenceReaderTaskEnvelope,
  report: ValidationReport,
): void {
  const packet = result.payload;
  const criteria = new Set(task.payload.searchNeed.successCriteria);
  const unansweredCriteria = new Set(
    packet.unanswered.map((item) => item.successCriterion),
  );
  packet.unanswered.forEach((item, index) => {
    if (!criteria.has(item.successCriterion)) {
      addError(
        report,
        "evidence.unanswered_unknown_criterion",
        `/payload/unanswered/${index}/successCriterion`,
        "unanswered item must quote a supplied success criterion",
      );
    }
  });
  if (packet.conclusion === "answered") {
    if (
      packet.findings.length === 0 ||
      packet.unanswered.length !== 0
    ) {
      addError(
        report,
        "evidence.answered_matrix",
        "/payload/conclusion",
        "answered requires findings and no unanswered criteria",
      );
    }
  } else if (packet.conclusion === "partial") {
    if (
      packet.findings.length + packet.contradictions.length === 0 ||
      packet.unanswered.length === 0
    ) {
      addError(
        report,
        "evidence.partial_matrix",
        "/payload/conclusion",
        "partial requires a finding/contradiction and at least one unanswered criterion",
      );
    }
  } else {
    if (
      packet.searches.length === 0 ||
      packet.findings.length !== 0 ||
      packet.contradictions.length !== 0 ||
      unansweredCriteria.size !== criteria.size ||
      [...criteria].some((criterion) => !unansweredCriteria.has(criterion))
    ) {
      addError(
        report,
        "evidence.not_found_matrix",
        "/payload/conclusion",
        "not_found requires a completed logical query, no findings/contradictions, and exhaustive unanswered criteria",
      );
    }
  }
}

function validateQueryLevelTerms(
  search: EvidencePacketEnvelope["payload"]["searches"][number],
  task: EvidenceReaderTaskEnvelope,
  index: number,
  report: ValidationReport,
): void {
  const terms = new Set(search.terms.map((item) => item.term));
  const need = task.payload.searchNeed;
  const hasAny = (values: readonly string[]) =>
    values.some((value) => terms.has(value));
  if (
    search.logicalQueryLevel === 1 &&
    (!hasAny(need.technicalObjects) ||
      !hasAny(need.scenarioTerms) ||
      !hasAny(need.evidenceIntentTerms))
  ) {
    addError(
      report,
      "evidence.q1_term_axes",
      `/payload/searches/${index}/terms`,
      "Q1 requires a technical object, exact scenario, and evidence-intent term",
    );
  }
  if (
    search.logicalQueryLevel === 2 &&
    !hasAny([...need.performanceRelations, ...need.evidenceIntentTerms])
  ) {
    addError(
      report,
      "evidence.q2_relation_axis",
      `/payload/searches/${index}/terms`,
      "Q2 requires a frozen performance or intent-specific relation term",
    );
  }
  if (
    search.logicalQueryLevel === 3 &&
    !search.terms.some(
      (term) =>
        need.synonymGroups.flat().includes(term.term) ||
        term.source === "context",
    )
  ) {
    addError(
      report,
      "evidence.q3_provenance_axis",
      `/payload/searches/${index}/terms`,
      "Q3 requires a frozen synonym or an earlier-context term",
    );
  }
}

function normalizeQuery(query: string): string {
  return query.trim().replace(/\s+/g, " ").toLocaleLowerCase();
}

function sameTermLedger(
  left: EvidencePacketEnvelope["payload"]["searches"][number]["terms"],
  right: EvidencePacketEnvelope["payload"]["searches"][number]["terms"],
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}
