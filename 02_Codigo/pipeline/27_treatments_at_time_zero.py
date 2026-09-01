# 27_treatments_at_time_zero.py
# Produces: the treatment rows of Table 1 (TRIPOD+AI item 6c)
# Levodopa daily dose, total levodopa-equivalent daily dose, the same dose with
# amantadine removed, dopamine agonist use and amantadine use, all as recorded at
# time-zero (the date of levodopa initiation). A one-month window is used because the
# PPMI LEDD log records start and stop dates with month precision; wider windows begin
# to capture dose escalation after initiation.
#
# Rewritten on 31 August 2026. It previously carried its own copies of the three drug
# searches, written in upper case, which had drifted from the audited ones in cohort.py:
# it found 220 agonist users where the audited search finds 227, and it missed six
# levodopa records under names that are unambiguously levodopa (Dopicar, Levadopa,
# Levodop Neuraxpharm, Sinement). It also pointed at the previous project layout and
# could not run as shipped. It now imports the searches, the cohort and the paths from
# cohort.py, so the definitions cannot drift again, and it prints the amantadine-free
# dose that the Methods describe and Table 1 reports.

import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import cohort as C

TAB = C.ensure_dir(os.path.join(C.RESULTS, "Tabelas"))
D31 = pd.Timedelta(days=C.WINDOW_DAYS)

dev = C.build_cohort(with_metadata=False)[["PATNO"]].reset_index(drop=True)

L = pd.read_csv(C._find("Historia_Medica_e_Medicacao", "LEDD_Concomitant_Medication_Log_*.csv"),
                low_memory=False)
T = pd.read_parquet(os.path.join(C.PROC, "tzero.parquet"))[["PATNO", "tzero_levodopa"]]

name = L.LEDTRT.astype(str)
L["is_agonist"] = name.str.lower().str.contains(C.RE_AGONIST, regex=True, na=False)
L["is_amantadine"] = name.str.lower().str.contains(C.RE_AMANTADINE, regex=True, na=False)
L["is_levodopa"] = (name.apply(lambda s: C.is_levodopa(s, extended=True))
                    & ~L.is_agonist & ~L.is_amantadine)

L["start"] = pd.to_datetime(L.STARTDT, format="%m/%Y", errors="coerce")
L["stop"] = pd.to_datetime(L.STOPDT, format="%m/%Y", errors="coerce")
L["LEDD"] = pd.to_numeric(L.LEDD, errors="coerce")
L["PATNO"] = pd.to_numeric(L.PATNO, errors="coerce")
L = L.merge(T, on="PATNO", how="inner").dropna(subset=["tzero_levodopa"])

tz = L.tzero_levodopa
active = L.start.notna() & (L.start <= tz + D31) & (L.stop.isna() | (L.stop >= tz - D31))

g = L[active].groupby("PATNO")
r = dev.copy()
r = r.merge(g.apply(lambda x: x.loc[x.is_levodopa, "LEDD"].sum()).rename("levodopa_mg"),
            on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x.LEDD.sum()).rename("ledd_total_mg"), on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x.loc[~x.is_amantadine, "LEDD"].sum()).rename("ledd_no_amantadine_mg"),
            on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x.is_agonist.any()).rename("agonist"), on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x.is_amantadine.any()).rename("amantadine"), on="PATNO", how="left")
r[["agonist", "amantadine"]] = r[["agonist", "amantadine"]].fillna(False)


def med_iqr(s):
    s = s[s.notna() & (s > 0)]
    q = s.quantile([.25, .5, .75])
    return "%.0f [%.0f-%.0f]" % (q[.5], q[.25], q[.75]), int(len(s))


lev, n_lev = med_iqr(r.levodopa_mg)
tot, n_tot = med_iqr(r.ledd_total_mg)
noam, n_noam = med_iqr(r.ledd_no_amantadine_mg)
out = pd.DataFrame([
    dict(characteristic="Levodopa daily dose at time-zero, mg",
         value=lev, n_available=n_lev, n_total=len(r)),
    dict(characteristic="Total levodopa-equivalent daily dose at time-zero, mg",
         value=tot, n_available=n_tot, n_total=len(r)),
    dict(characteristic="Total dopaminergic daily dose at time-zero, amantadine excluded, mg",
         value=noam, n_available=n_noam, n_total=len(r)),
    dict(characteristic="Dopamine agonist at time-zero, n (%)",
         value="%d (%.0f%%)" % (r.agonist.sum(), 100 * r.agonist.mean()),
         n_available=len(r), n_total=len(r)),
    dict(characteristic="Amantadine at time-zero, n (%)",
         value="%d (%.0f%%)" % (r.amantadine.sum(), 100 * r.amantadine.mean()),
         n_available=len(r), n_total=len(r)),
])
print("PPMI development sample, n = %d" % len(r))
print(out.to_string(index=False))
print("\nagonist %d (%.1f%%), amantadine %d (%.1f%%)"
      % (r.agonist.sum(), 100 * r.agonist.mean(), r.amantadine.sum(), 100 * r.amantadine.mean()))
out.to_csv(os.path.join(TAB, "tab27_treatments_at_time_zero.csv"), index=False)
r.to_csv(os.path.join(TAB, "tab27_treatments_per_patient.csv"), index=False)
print("\nsaved tab27_treatments_at_time_zero.csv to %s" % TAB)
