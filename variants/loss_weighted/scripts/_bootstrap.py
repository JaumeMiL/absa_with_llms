import sys
from pathlib import Path


# ES UNA TONTERIA PERÒ AFEGEIX EL SRC AL PATH :) PER PODER FER IMPORT common I COSES AIXI
ABSA_ROOT = Path(__file__).resolve().parents[1]
SRC = ABSA_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

