export type FailureClass =
  | "retryable_output_contract"
  | "transient_provider"
  | "stale_state"
  | "domain_or_gate"
  | "security_terminal"
  | "input_contract"
  | "terminal_runtime";

export interface ClassifiedFailure {
  class: FailureClass;
  code: string;
  sameRoleOutputRetry: boolean;
  workflowSemanticTrigger: boolean;
  terminal: boolean;
}

export function classifyFailure(code: string): ClassifiedFailure {
  if (
    /^(syntax|schema|binding\.identity|registry\.output_message|normalization)/.test(
      code,
    )
  ) {
    return classification(
      "retryable_output_contract",
      code,
      true,
      false,
      false,
    );
  }
  if (/^(provider\.timeout|provider\.unavailable|provider\.interrupted)/.test(code)) {
    return classification("transient_provider", code, false, false, false);
  }
  if (/^(binding\.stale|workflow\.stale|cas\.)/.test(code)) {
    return classification("stale_state", code, false, false, false);
  }
  if (/^(security\.|no_experiment\.)/.test(code)) {
    return classification("security_terminal", code, false, false, true);
  }
  if (/^(task\.|input_contract)/.test(code)) {
    return classification("input_contract", code, false, false, true);
  }
  if (/^(domain\.|gate\.|review\.|closure\.|evidence\.)/.test(code)) {
    return classification("domain_or_gate", code, false, true, false);
  }
  return classification("terminal_runtime", code, false, false, true);
}

function classification(
  failureClass: FailureClass,
  code: string,
  sameRoleOutputRetry: boolean,
  workflowSemanticTrigger: boolean,
  terminal: boolean,
): ClassifiedFailure {
  return {
    class: failureClass,
    code,
    sameRoleOutputRetry,
    workflowSemanticTrigger,
    terminal,
  };
}

