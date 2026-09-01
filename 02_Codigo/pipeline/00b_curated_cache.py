# 00b_curated_cache.py
# Produces: curated_cache.parquet, the Curated Data Cut in a form the pipeline can read.
#
# Why this script exists. Six scripts read curated_cache.parquet, and until now none wrote
# it. It is where every clinical scale, every cerebrospinal fluid and blood assay and every
# striatal binding value in the analysis comes from, so the whole pipeline rested on a file
# that arrived from outside it. The transformation turns out to be nothing at all: the cache
# is the first worksheet of the PPMI Curated Data Cut, 18,821 visits by 181 columns, read
# once and stored as parquet because reading the spreadsheet takes about a minute and the
# pipeline reads it repeatedly. That is worth having written down and checkable rather than
# assumed, which is what this script is for.
#
# The spreadsheet carries three worksheets. The first is the data; the other two are the
# data dictionary and the provenance note, and neither is read by anything. The worksheet
# is taken by position rather than by name because its name is the release date and changes
# with every cut.
#
# Usage:
#   python 00b_curated_cache.py           # rebuild and compare against the stored cache
#   python 00b_curated_cache.py --write   # rebuild and overwrite it

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cohort as C
from config import PROJECT_ROOT

CACHE = os.path.join(PROJECT_ROOT, "02_Dados_Processados", "curated_cache.parquet")


def source():
    """The most recent Curated Data Cut workbook in the raw download."""
    found = sorted(glob.glob(os.path.join(C.RAW, "Curated_Data_Cut", "*.xlsx")))
    if not found:
        raise SystemExit("no Curated Data Cut workbook found under %s"
                         % os.path.join(C.RAW, "Curated_Data_Cut"))
    return found[-1]


def build(path):
    book = pd.ExcelFile(path)
    return pd.read_excel(path, sheet_name=book.sheet_names[0])


def compare(built, stored):
    """Column by column, so that a divergence names itself instead of being a bare False."""
    if list(built.columns) != list(stored.columns):
        only_built = [c for c in built.columns if c not in stored.columns]
        only_stored = [c for c in stored.columns if c not in built.columns]
        print("  columns differ: %d only in the rebuild, %d only in the cache"
              % (len(only_built), len(only_stored)))
        if only_built:
            print("    only in the rebuild: %s" % ", ".join(only_built[:10]))
        if only_stored:
            print("    only in the cache  : %s" % ", ".join(only_stored[:10]))
        return False
    if len(built) != len(stored):
        print("  row counts differ: %d rebuilt, %d stored" % (len(built), len(stored)))
        return False
    bad = []
    for c in built.columns:
        x, y = built[c], stored[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            differs = ~np.isclose(x.astype(float), y.astype(float),
                                  equal_nan=True, rtol=1e-9, atol=1e-9)
        else:
            # A text column round-tripped through parquet comes back with None where the
            # spreadsheet had NaN, and str() renders those two differently. Missing is
            # missing: compare the null masks, and the values only where both are present.
            both_null = x.isna() & y.isna()
            differs = ~(x.astype(str) == y.astype(str)) & ~both_null
        if differs.sum():
            bad.append((c, int(differs.sum())))
    if bad:
        print("  %d columns differ: %s" % (len(bad),
              ", ".join("%s (%d rows)" % b for b in bad[:10])))
        return False
    print("  all %d columns reproduce exactly" % len(built.columns))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="overwrite curated_cache.parquet with the rebuild")
    args = parser.parse_args()

    path = source()
    print("source workbook: %s" % os.path.basename(path))
    built = build(path)
    print("rebuilt %d rows and %d columns" % built.shape)

    if os.path.exists(CACHE):
        stored = pd.read_parquet(CACHE)
        print("stored cache has %d rows and %d columns" % stored.shape)
        ok = compare(built, stored)
        if not args.write:
            print("\nthe rebuild %s the stored cache"
                  % ("reproduces" if ok else "DOES NOT reproduce"))
            print("run with --write to rewrite it")
            return
    elif not args.write:
        print("no stored cache to compare against; run with --write to create it")
        return

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    built.to_parquet(CACHE, index=False)
    print("written %s" % CACHE)


if __name__ == "__main__":
    main()
