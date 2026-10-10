export type Scalar = string | number | boolean | null;
export type Page<T> = {
  items: T[];
  total: number;
  offset: number;
  limit: number;
  next_offset: number | null;
};
export type Dataset = {
  id: string;
  name: string;
  etag: number;
  state: string;
  current_version_id: string | null;
};
export type Column = {
  position: number;
  source_header: string;
  field: string;
  unit: string | null;
  transform: string;
};
export type Mapping = {
  schema_version: string;
  mapping_version: string;
  record_level: string;
  columns: Column[];
  analysis_blockers: string[];
  encoding: string;
  delimiter: string;
};
export type Schema = {
  headers: string[];
  mapping: Mapping | null;
  mapping_sha256: string | null;
  import_report: unknown;
};
export type Field = {
  name: string;
  original_value: Scalar;
  normalized_value: Scalar;
  unit: string | null;
  state: string;
  source: { row: number; column: string; table: string };
};
export type DataRecord = {
  id: string;
  revision: number;
  fields: Field[];
  business_key: Record<string, Scalar> | null;
};
export type Finding = {
  id: string;
  record_id: string;
  field_name: string | null;
  signal_kind: string;
  severity: string;
  priority: number;
  anomaly_score?: {
    value: number;
    scale: string;
    higher_is_worse: boolean;
  } | null;
  evidence: { code: string; message: string; reference_id?: string | null }[];
};
export type Review = {
  id: string;
  action: string;
  sequence: number;
  operator: string;
  review: {
    error_assessment: string;
    suggestion_decision: string;
    suggestion_id: string | null;
    note: string | null;
    reviewed_at: string;
  };
};
export type FindingItem = {
  finding: Finding;
  review_version: number;
  latest_review: Review | null;
};
export type Support = {
  source_id: string;
  version: string;
  method: string;
  status: string;
  basis: string;
  dependency_roots: string[];
  input_fields: string[];
  limitations: string[];
};
export type Suggestion = {
  id: string;
  schema_version: string;
  field_name: string;
  proposed_value: Scalar;
  candidate_range: { lower: number; upper: number } | null;
  unit: string | null;
  status: string;
  method: string | null;
  refusal_reason: string | null;
  limitations: string[];
  before: Field | null;
  evidence: Support[];
  uncertainty: { category: string; basis: string };
  explanation: {
    observed: string;
    trigger: string[];
    comparisons: string[];
    applicability: string[];
    considered_range: string;
    missing: string[];
    specialist_actions: string[];
  };
  binding: {
    dataset_revision: number;
    mapping_sha256: string;
    snapshot_sha256: string;
  };
  alternatives: {
    value: number | null;
    lower: number | null;
    upper: number | null;
    unit: string;
    support: Support;
  }[];
  unit_hypotheses: {
    observed_unit: string | null;
    candidate_unit: string;
    next_action: string;
  }[];
};
export type Recommendation = {
  suggestion: Suggestion;
  patch: unknown | null;
  validation: {
    status: string;
    limitations: string[];
    new_violations: string[];
    unresolved_violations: string[];
  };
};
export type Detail = FindingItem & { suggestions: Recommendation[] };
export type Job = {
  id: string;
  kind: string;
  status: string;
  error_code: string | null;
  request: {
    version_id?: string;
    dataset_id?: string;
    kind?: string;
    format?: string;
  };
  result: Record<string, unknown> | null;
  progress_percent: number | null;
  created_at?: number;
};
export type Profile = {
  id: string;
  name: string;
  version: string;
  source_kind: string;
  mapping_sha256: string;
  cohort_size: number | null;
  models: { id: string; target: string }[];
  limitations: string[];
};
export type Draft = {
  assessment: "confirmed_error" | "no_error" | "uncertain";
  decision: "not_considered" | "accepted" | "rejected" | "deferred";
  comment: string;
  suggestionId: string | null;
};
export type Version = { id: string; number: number; parent_id: string | null };
