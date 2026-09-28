"""Build the ciprofloxacin demo table from the Liverpool E. coli dataset (Gerada et al. 2026, CC BY 4.0).

    uv run --with pandas dataset/build_cip_table.py

Inputs (in liverpool_ecoli/, from https://datacat.liverpool.ac.uk/id/eprint/3008):
  meta_data_tidy.txt  wide AST table; mic_CIP holds agar-dilution MICs in mg/L, e.g. "0.008", "0.25", ">4"
  annots.zip          one AMRFinderPlus v3.11.14 TSV per isolate (acquired genes + point mutations)

Output cip_features.csv, one row per isolate with a CIP MIC and an annotation file:
  genome_id, mic_raw, censor ("=" or ">"), log2_lo, log2_hi   interval for log2 MIC on the dilution grid
  log2_mic                                                     paper's convention: ">x" -> 2x, then log2
  <feature columns>                                            0/1 presence of each AMRFinderPlus "AMR"
                                                               element (gene or point mutation) carried
                                                               by at least MIN_CARRIERS isolates
Censoring: ">4" means MIC > 4, so log2 MIC lies in (2, inf). A value on the grid, e.g. 0.25, means the
true MIC is in (0.125, 0.25], so log2 MIC in (-3, -2]. The lowest dilution tested (0.008) is an upper
bound only: log2 MIC in (-inf, log2 0.008].
"""

import io
import math
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "liverpool_ecoli"
MIN_CARRIERS = 5
LOWEST_TESTED = 0.008


def interval(mic: str) -> tuple[str, float, float, float]:
    """Return (censor, log2_lo, log2_hi, log2_convention) for one MIC string."""
    if mic.startswith(">"):
        v = float(mic[1:])
        return ">", math.log2(v), math.inf, math.log2(2 * v)
    v = float(mic.lstrip("<=≤"))
    lo = -math.inf if v <= LOWEST_TESTED else math.log2(v) - 1
    return "=", lo, math.log2(v), math.log2(v)


def main() -> None:
    meta = pd.read_csv(SRC / "meta_data_tidy.txt", sep="\t", dtype=str, keep_default_na=False)
    meta = meta[meta["mic_CIP"].str.len().gt(0) & meta["mic_CIP"].ne("NA")]

    calls = []
    with zipfile.ZipFile(SRC / "annots.zip") as zf:
        for name in zf.namelist():
            if not name.endswith(".tsv") or "__MACOSX" in name:
                continue
            t = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t", dtype=str)
            t = t[t["Element type"] == "AMR"]
            calls.append(pd.DataFrame({"genome_id": Path(name).stem, "symbol": t["Gene symbol"]}))
    calls = pd.concat(calls, ignore_index=True).drop_duplicates()

    ids = sorted(set(meta["genome_id"]) & set(calls["genome_id"]), key=lambda s: (s[0], int(s[1:])))
    calls = calls[calls["genome_id"].isin(ids)]
    carriers = calls.groupby("symbol")["genome_id"].nunique()
    keep = sorted(carriers[carriers >= MIN_CARRIERS].index)

    X = (pd.crosstab(calls["genome_id"], calls["symbol"]).reindex(index=ids, columns=keep, fill_value=0) > 0).astype(int)
    mic = meta.set_index("genome_id").loc[ids, "mic_CIP"]
    cols = pd.DataFrame([interval(m) for m in mic], index=ids, columns=["censor", "log2_lo", "log2_hi", "log2_mic"])
    out = pd.concat([mic.rename("mic_raw"), cols, X], axis=1).rename_axis("genome_id").reset_index()
    out.to_csv(HERE / "cip_features.csv", index=False)
    print(f"wrote cip_features.csv: {len(out)} isolates x {len(keep)} features "
          f"({(out['censor'] == '>').sum()} right-censored, {(out['log2_lo'] == -math.inf).sum()} at the lowest dilution)")


if __name__ == "__main__":
    main()
