import os
from pathlib import Path


ABSA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ABSA_ROOT
DATASET_DIR = ABSA_ROOT / "dataset"
CONFIG_DIR = ABSA_ROOT / "configs"
PROMPT_DIR = CONFIG_DIR / "prompts"
RUNS_DIR = ABSA_ROOT / "runs"
ARTIFACTS_DIR = ABSA_ROOT / "artifacts"
GENERATED_DIR = ARTIFACTS_DIR / "generated_data"
REPORTS_DIR = ABSA_ROOT / "reports"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalize_quantization(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "4bit" if value else "none"
    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if normalized in {"", "none", "no", "false", "0", "off"}:
        return "none"
    if normalized in {"4bit", "4bits", "bnb4bit", "bitsandbytes4bit", "nf4"}:
        return "4bit"
    raise ValueError(
        f"Unsupported quantization value {value!r}. Use 'none' or '4bit'."
    )

def resolve_path(path, base = None):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    roots = [base, ABSA_ROOT, REPO_ROOT, CONFIG_DIR]
    for root in roots:
        if root is None:
            continue
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (ABSA_ROOT / candidate).resolve()


def resolve_existing_or_default(path, base = None):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if base is not None:
        return (base / candidate).resolve()
    return candidate.resolve()



ASPECTS = (
    "restaurant_general",
    "restaurant_prices",
    "food_quality",
    "food_prices",
    "food_style_options",
    "drinks_quality",
    "drinks_prices",
    "drinks_style_options",
    "ambience",
    "service",
    "location",
)

POLARITIES = ("positive", "negative", "neutral", "conflict")



from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelSpec:
    name: str
    paths: tuple[str, ...]
    reportable: bool
    lora_targets: tuple[str, ...]
    local_smoke: bool = False
    trust_remote_code: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llama32B3": ModelSpec(
        name="llama32B3",
        paths=(
            "/scratch/nas/1/PDI/mml0/models/llama32B3/",
            "/scratch/nas/1/PDI/mml0/llama32B3/snapshots/snap0",
        ),
        reportable=True,
        lora_targets=("q_proj", "v_proj"),
    ),
    "qwen25B3": ModelSpec(
        name="qwen25B3",
        paths=(
            "/scratch/nas/1/PDI/mml0/models/qwen25B3/",
            "/scratch/nas/1/PDI/mml0/qwen25B3/snapshots/snap0",
        ),
        reportable=True,
        lora_targets=("q_proj", "v_proj"),
    ),
    "tiny-local": ModelSpec(
        name="tiny-local",
        paths=("sshleifer/tiny-gpt2",),
        reportable=False,
        lora_targets=("c_attn",),
        local_smoke=True,
    ),
    "qwen-small": ModelSpec(
        name="qwen-small",
        paths=("Qwen/Qwen2.5-0.5B-Instruct",),
        reportable=False,
        lora_targets=("q_proj", "v_proj"),
        local_smoke=True,
    ),
    "ensemble": ModelSpec(
        name="ensemble",
        paths=("",),
        reportable=True,
        lora_targets=(),
    ),
}


def get_model_spec(model):
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    return ModelSpec( # carrega directament de huggingface sino :)
        name=model,
        paths=(model,),
        reportable=False,
        lora_targets=("q_proj", "v_proj"),
    )


def resolve_model_path(model, override = None):
    if override:
        return override
    spec = get_model_spec(model)
    model_root = os.environ.get("ABSA_MODEL_ROOT")
    if model_root and model in MODEL_REGISTRY:
        rooted = Path(model_root) / model
        if rooted.exists():
            return str(rooted)
    for candidate in spec.paths:
        if Path(candidate).exists():
            return candidate
    return spec.paths[0]
