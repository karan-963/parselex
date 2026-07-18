/**
 * Maps an inference artifact filename to its training-engine rerun stage.
 * Mirrors the `/runs/{slug}/rerun/{stage}` endpoints in inference_v2/routes.py.
 * Returns null for artifacts that cannot be re-run in isolation (raw extraction,
 * final aggregation).
 */
const ARTIFACT_RERUN_STAGE: Record<string, string> = {
  '2_section_headings.json': 'section_p1',
  '3_section_labels.json': 'section_p2',
  '5_education_boundaries.json': 'education_phase2_divider',
  '4_education_segments.json': 'education_phase1_segment',
  '6_education_fields.json': 'education_phase3_classify',
  '7_skills_fields.json': 'skills_classify',
  '9_experience_boundaries.json': 'experience_phase2_divider',
  '8_experience_segments.json': 'experience_phase1_segment',
  '10_experience_classification.json': 'experience_phase3_classify',
  '12_project_boundaries.json': 'project_phase2_divider',
  '11_project_segments.json': 'project_phase1_segment',
  '13_project_fields.json': 'project_phase3_classify',
  '15_personal_fields.json': 'personal_classify',
};

export function getRerunStage(filename: string): string | null {
  return ARTIFACT_RERUN_STAGE[filename] ?? null;
}
