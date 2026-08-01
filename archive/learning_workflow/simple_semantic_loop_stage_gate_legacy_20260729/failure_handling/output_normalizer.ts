export interface NormalizedOutput {
  normalizedText: string | null;
  parsed: unknown | null;
  transformations: Array<
    "removed_bom" | "normalized_newlines" | "trimmed_whitespace" | "removed_single_fence" | "extracted_unique_json"
  >;
  errorCode:
    | null
    | "empty_response"
    | "no_json_value"
    | "multiple_json_values"
    | "invalid_json";
  errorMessage: string | null;
}

export function normalizeAgentOutput(raw: string): NormalizedOutput {
  const transformations: NormalizedOutput["transformations"] = [];
  let value = raw;
  if (value.startsWith("\uFEFF")) {
    value = value.slice(1);
    transformations.push("removed_bom");
  }
  const newlineNormalized = value.replace(/\r\n?/g, "\n");
  if (newlineNormalized !== value) {
    value = newlineNormalized;
    transformations.push("normalized_newlines");
  }
  const trimmed = value.trim();
  if (trimmed !== value) {
    value = trimmed;
    transformations.push("trimmed_whitespace");
  }
  if (!value) {
    return failure(transformations, "empty_response", "response is empty");
  }
  const fence = value.match(/^```(?:json)?[ \t]*\n([\s\S]*?)\n```$/i);
  if (fence) {
    value = fence[1]!.trim();
    transformations.push("removed_single_fence");
  }
  try {
    return {
      normalizedText: value,
      parsed: JSON.parse(value),
      transformations,
      errorCode: null,
      errorMessage: null,
    };
  } catch {
    // The protocol permits extracting exactly one complete JSON value from
    // otherwise non-JSON wrapper text; the value itself remains untouched.
  }
  const candidates = findCompleteJsonContainers(value);
  if (candidates.length === 0) {
    return failure(
      transformations,
      "no_json_value",
      "no complete JSON object/array was found",
    );
  }
  const valid = candidates.flatMap((candidate) => {
    try {
      return [{ text: candidate, value: JSON.parse(candidate) }];
    } catch {
      return [];
    }
  });
  if (valid.length > 1) {
    return failure(
      transformations,
      "multiple_json_values",
      "response contains more than one complete JSON value",
    );
  }
  if (valid.length === 0) {
    return failure(
      transformations,
      "invalid_json",
      "the unique JSON-shaped value does not parse",
    );
  }
  transformations.push("extracted_unique_json");
  return {
    normalizedText: valid[0]!.text,
    parsed: valid[0]!.value,
    transformations,
    errorCode: null,
    errorMessage: null,
  };
}

function failure(
  transformations: NormalizedOutput["transformations"],
  errorCode: NonNullable<NormalizedOutput["errorCode"]>,
  errorMessage: string,
): NormalizedOutput {
  return {
    normalizedText: null,
    parsed: null,
    transformations,
    errorCode,
    errorMessage,
  };
}

function findCompleteJsonContainers(text: string): string[] {
  const results: string[] = [];
  let start = -1;
  const stack: string[] = [];
  let inString = false;
  let escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]!;
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === '"') {
        inString = false;
      }
      continue;
    }
    if (character === '"') {
      inString = true;
      continue;
    }
    if (character === "{" || character === "[") {
      if (stack.length === 0) start = index;
      stack.push(character);
      continue;
    }
    if (character === "}" || character === "]") {
      if (stack.length === 0) continue;
      const opener = stack.at(-1);
      if (
        (opener === "{" && character !== "}") ||
        (opener === "[" && character !== "]")
      ) {
        stack.length = 0;
        start = -1;
        continue;
      }
      stack.pop();
      if (stack.length === 0 && start >= 0) {
        results.push(text.slice(start, index + 1));
        start = -1;
      }
    }
  }
  return results;
}

