from pathlib import Path


ABSA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ABSA_ROOT.parent
DATASET_DIR = ABSA_ROOT / "dataset"
CONFIG_DIR = ABSA_ROOT / "configs"
PROMPT_DIR = CONFIG_DIR / "prompts"
RUNS_DIR = ABSA_ROOT / "runs"
REPORTS_DIR = ABSA_ROOT / "reports"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

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

