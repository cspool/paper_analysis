import type { ParsedProtocol, Role } from "./types.ts";

function requireLoopContains(protocol: ParsedProtocol, token: string): void {
  if (!protocol.loop?.includes(token)) {
    throw new Error(`marker ${protocol.marker} requires LOOP containing ${token}`);
  }
}

function requirePayload(protocol: ParsedProtocol): void {
  if (protocol.payload === null) {
    throw new Error(`marker ${protocol.marker} requires a semantic payload`);
  }
}

export function validateProtocolTransition(role: Role, protocol: ParsedProtocol): void {
  switch (role) {
    case "anchor_stage_controller":
      if (protocol.marker === "ANCHOR_ROUND_PLAN") {
        requirePayload(protocol);
        requireLoopContains(protocol, "§EVAL_ROUND");
        if (!Array.isArray(protocol.payload)) {
          throw new Error("ANCHOR_ROUND_PLAN payload must be an array");
        }
      } else {
        requireLoopContains(protocol, "§TERMINATED");
      }
      break;
    case "anchor_evidence_worker":
      requirePayload(protocol);
      if (!Array.isArray(protocol.payload)) {
        throw new Error("ANCHOR_EVIDENCE_RESULT claims must be an array");
      }
      break;
    case "anchor_curator_worker":
      requirePayload(protocol);
      if (Array.isArray(protocol.payload) || typeof protocol.payload !== "object") {
        throw new Error("ANCHOR_DELTA payload must be an object");
      }
      break;
    case "direction_planner":
      if (protocol.marker === "DIRECTION_PROPOSAL") {
        requirePayload(protocol);
        requireLoopContains(protocol, "§EVAL_DIRECTION");
      } else {
        requireLoopContains(protocol, "§TERMINATED");
      }
      break;
    case "direction_reviewer":
      if (protocol.marker === "REVIEW_QUESTION") {
        requirePayload(protocol);
        requireLoopContains(protocol, "§EVAL_ANSWER");
      } else if (protocol.marker === "REVIEW_REFERENCE_REQUEST") {
        requirePayload(protocol);
        requireLoopContains(protocol, "§ASK");
      } else {
        requirePayload(protocol);
        requireLoopContains(protocol, "§TERMINATED");
      }
      break;
    case "review_evidence_worker":
      requirePayload(protocol);
      break;
    default: {
      const exhaustive: never = role;
      throw new Error(`unhandled role: ${exhaustive}`);
    }
  }
}

