from datetime import datetime
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message, **fields) -> None:
        # **fiels pq aixi puc escriure qualsevol cosa :)
        # es una tonteria realment eh no crec ni que faci falta però per el training i aixi crec que ens
        # anirà be tenir un log com a tal i no els prints de python default de slurm
        # pero si no t'agrada es pot borrar facil
        timestamp = datetime.now().isoformat(timespec="seconds")
        suffix = ""
        if fields:
            parts = [f"{key}={value}" for key, value in fields.items()]
            suffix = " | " + " ".join(parts)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}{suffix}\n")

