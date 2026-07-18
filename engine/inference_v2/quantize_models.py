"""One-off generator for INT8 (dynamically quantized) variants of every stage model.

For each inference_v2 ML stage this constructs the fp32 predictor, applies dynamic
INT8 quantization to its Linear layers, and saves the resulting quantized module as
a ``*_int8.pt`` file next to the original ``best_model.pt``. At inference time
:func:`inference_v2.model_precision.apply_precision` loads these files when the
user selects INT8 precision.

Run from the training-engine directory::

    python -m inference_v2.quantize_models
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

import torch

import inference_v2
from inference_v2.model_precision import int8_path_for, quantize_dynamic_int8

BASE = os.path.dirname(os.path.abspath(inference_v2.__file__))

# (stage, module_path, class_name, fp32_filename_relative_to_stage_dir | None)
# None means the fp32 path is resolved dynamically (personal_classify).
STAGES: list[tuple[str, str, str, str | None]] = [
    ("section_p1", "inference_v2.section_p1.line_hybrid_predict", "LineHybridPredictor", "best_model_line_minilm.pt"),
    ("section_p2", "inference_v2.section_p2", "PyTorchSectionPhase2Predictor", "best_model.pt"),
    ("education_phase1_segment", "inference_v2.education_phase1_segment.predictor", "EducationPhase1Predictor", "best_model.pt"),
    ("education_phase2_divider", "inference_v2.education_phase2_divider", "PyTorchEducationPhase2DividerPredictor", "best_model.pt"),
    ("education_phase3_classify", "inference_v2.education_phase3_classify.predictor", "EducationPhase3Predictor", "best_model.pt"),
    ("skills_classify", "inference_v2.skills_classify.predictor", "SkillsPhase7Predictor", "best_model.pt"),
    ("experience_phase1_segment", "inference_v2.experience_phase1_segment", "PyTorchExperiencePhase2Predictor", "best_model.pt"),
    ("experience_phase2_divider", "inference_v2.experience_phase2_divider", "PyTorchExperiencePhase1Predictor", "best_model.pt"),
    ("experience_phase3_classify", "inference_v2.experience_phase3_classify", "PyTorchExperiencePhase2Predictor", "best_model.pt"),
    ("project_phase1_segment", "inference_v2.project_phase1_segment", "PyTorchProjectPhase1Predictor", "best_model.pt"),
    ("project_phase2_divider", "inference_v2.project_phase2_divider", "PyTorchProjectPhase2DividerPredictor", "best_model.pt"),
    ("project_phase3_classify", "inference_v2.project_phase3_classify.predictor", "ProjectPhase3Predictor", "best_model.pt"),
    ("personal_classify", "inference_v2.personal_classify.predictor", "PersonalPhase15Predictor", None),
]


def _resolve_fp32_path(stage: str, fname: str | None) -> str:
    if stage == "personal_classify":
        from inference_v2.personal_classify.predictor import _resolve_checkpoint_path

        return _resolve_checkpoint_path()
    assert fname is not None
    return os.path.join(BASE, stage, fname)


def _mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.isfile(path) else 0.0


def quantize_stage(stage: str, module_path: str, class_name: str, fname: str | None) -> None:
    fp32_path = _resolve_fp32_path(stage, fname)
    out_path = int8_path_for(fp32_path)

    module = importlib.import_module(module_path)
    predictor_cls = getattr(module, class_name)

    predictor = predictor_cls()
    model = predictor.model.to("cpu")
    quantized = quantize_dynamic_int8(model)
    quantized.eval()

    # Save the quantized state_dict (tensors + packed params). Whole-module pickles
    # break for models with monkey-patched attention forwards, so state_dict only.
    torch.save(quantized.state_dict(), out_path)
    print(f"  {stage:28s} fp32={_mb(fp32_path):7.1f}MB -> int8={_mb(out_path):7.1f}MB  ({os.path.basename(out_path)})")


def _run_single_stage(stage_name: str) -> None:
    for stage, module_path, class_name, fname in STAGES:
        if stage == stage_name:
            quantize_stage(stage, module_path, class_name, fname)
            return
    raise SystemExit(f"Unknown stage: {stage_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate INT8 model variants.")
    parser.add_argument(
        "--stage",
        help="Quantize a single stage in-process (used internally per subprocess).",
    )
    args = parser.parse_args()

    if args.stage:
        _run_single_stage(args.stage)
        return

    # Each stage runs in its own subprocess so that per-stage sys.path/sys.modules
    # mutations (training_bridge, data namespace shadowing) cannot cross-contaminate.
    print(f"Quantizing {len(STAGES)} stages (fp32 -> int8 dynamic), one subprocess each...")
    failures: list[str] = []
    for stage, _module_path, _class_name, _fname in STAGES:
        result = subprocess.run(
            [sys.executable, "-m", "inference_v2.quantize_models", "--stage", stage],
            cwd=os.path.dirname(BASE),
        )
        if result.returncode != 0:
            failures.append(stage)

    if failures:
        print(f"\n{len(failures)} stage(s) failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("\nDone. INT8 checkpoints written next to each best_model.pt.")


if __name__ == "__main__":
    main()
