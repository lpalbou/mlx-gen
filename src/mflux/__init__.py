import os

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mflux.python_runtime import (
    GeneratedOutput,
    GenerationRuntimePlan,
    LoadedGenerationModel,
    load_generation_model,
    load_generation_model_for_plan,
    resolve_generation_runtime,
    resolve_generation_runtime_for_plan,
)
from mflux.release.validation_registry import (
    FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID,
    I2I_EDIT_5X4_PROFILE_ID,
    REFRAME_OUTPAINT_PROFILE_ID,
    ModelValidation,
    ValidationProfile,
    ValidationRecord,
    get_model_validation,
    get_validation_profile,
    list_validation_profiles,
)
from mflux.task_inference import (
    GenerationCapability,
    GenerationPlan,
    ModelCapabilities,
    ResolvedTask,
    TaskInferenceError,
    get_model_capabilities,
    infer_task,
    normalize_i2i_mode,
    normalize_task,
    resolve_generation_plan,
    resolve_task,
)

__all__ = [
    "GenerationCapability",
    "GeneratedOutput",
    "FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID",
    "GenerationPlan",
    "GenerationRuntimePlan",
    "I2I_EDIT_5X4_PROFILE_ID",
    "LoadedGenerationModel",
    "ModelValidation",
    "ModelCapabilities",
    "REFRAME_OUTPAINT_PROFILE_ID",
    "ResolvedTask",
    "TaskInferenceError",
    "ValidationProfile",
    "ValidationRecord",
    "get_model_capabilities",
    "get_model_validation",
    "get_validation_profile",
    "infer_task",
    "list_validation_profiles",
    "load_generation_model",
    "load_generation_model_for_plan",
    "normalize_i2i_mode",
    "normalize_task",
    "resolve_generation_plan",
    "resolve_generation_runtime",
    "resolve_generation_runtime_for_plan",
    "resolve_task",
]
