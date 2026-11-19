"""Download OULAD (CC BY 4.0) from the dataset authors' R package and convert it to parquet.

Source: https://github.com/jakubkuzilek/oulad (Kuzilek, Hlosta & Zdrahal), pinned commit.
Original release: https://analyse.kmi.open.ac.uk/open_dataset
"""
import tempfile
import urllib.request
from pathlib import Path

import pyreadr

COMMIT = "b51016442e8987c47edd4fe0f8dac791cc1406da"
BASE = f"https://raw.githubusercontent.com/jakubkuzilek/oulad/{COMMIT}/data/"
TABLES = ["assessment", "course", "student", "student_assessment",
          "student_registration", "student_vle", "vle"]
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"

OUT.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for name in TABLES:
        target = OUT / f"{name}.parquet"
        if target.exists():
            continue
        rda = Path(tmp) / f"{name}.rda"
        urllib.request.urlretrieve(BASE + f"{name}.rda", rda)
        for key, df in pyreadr.read_r(rda).items():
            df.to_parquet(target)
            print(f"{name}: {len(df):,} rows")
