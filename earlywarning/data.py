from pathlib import Path
import pandas as pd

TABLES = ["assessment", "course", "student", "student_assessment",
          "student_registration", "student_vle", "vle"]
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def load(raw=RAW):
    t = {n: pd.read_parquet(Path(raw) / f"{n}.parquet") for n in TABLES}
    v = t["vle"][["id_site", "code_module", "code_presentation", "activity_type"]]
    t["student_vle"] = t["student_vle"].merge(v, on=["id_site", "code_module", "code_presentation"])
    return t
