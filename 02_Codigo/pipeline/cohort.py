# cohort.py
# Shared definitions used by scripts 31 to 46.
#
# Everything that more than one of the newer scripts needs lives here, so that the
# cohort, the predictor set, the drug-name searches and the cross-validation
# protocol are defined exactly once and cannot drift between analyses.

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import PROJECT_ROOT

RAW = os.path.join(PROJECT_ROOT, "01_Dados_Brutos")

# Processed files live outside the code folder so the repository holds no data.
PROC = os.environ.get(
    "LID_PROC_DIR",
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_Dados_processados")
    ),
)
RESULTS = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "03_Resultados")
)

# ----------------------------------------------------------------------------
# Model specification
# ----------------------------------------------------------------------------

# The six predictors of the development model.
TOTAL6 = [
    "updrs_totscore",   # total MDS-UPDRS, Parts I + II + III
    "ageonset",         # age at motor onset, years
    "SEX",              # 1 = male
    "BMI",              # body-mass index, kg/m2
    "td_pigd_ratio",    # tremor-dominant to postural-instability gait-difficulty ratio
    "NP2FREZ",          # MDS-UPDRS item 2.13, freezing of gait
]

# The candidate pool on which complete cases were required. Every candidate
# clinical model was compared in the same participants, and those participants
# form the development sample of 813 with 165 events.
POOL = [
    "updrs_totscore", "updrs1_score", "updrs2_score", "updrs3_score",
    "comp_bradicinesia", "td_pigd_ratio", "pigd", "MSEADLG", "hy",
    "Clinical_Stage", "NP1FATG", "NP1DPRS", "NP2FREZ", "ageonset",
    "SEX", "BMI", "scopa_gi", "EDUCYRS",
]

RIDGE = 0.05          # ridge penalty of the development model
CV_REPEATS = 20       # repetitions of five-fold cross-validation
CV_FOLDS = 5
CV_SEED = 100         # seed of repetition r is CV_SEED + r
BOOT_SEED = 20260825  # bootstrap optimism correction

# ----------------------------------------------------------------------------
# Drug-name searches
#
# The medication log records free text, so brand names and misspellings have to be
# searched for explicitly. The three levodopa brand names below (Dopicar, Levocomp,
# Sinement) were added during quality control; see 39_drug_name_quality_control.py.
# Clarium is piribedil and is therefore an agonist, not levodopa.
# ----------------------------------------------------------------------------

RE_LEVODOPA = (
    r"dopa|carb.{0,4}lev|lev.{0,3}dop|sinemet|madopar|stalevo|rytary|duopa|"
    r"prolopa|nacom|benseraz|parcopa|isicom|careldopa"
)
RE_LEVODOPA_EXTENDED = RE_LEVODOPA + r"|dopicar|levocomp|sinement"

# Both searches below were widened on 31 August 2026, after the free-text names left
# unmatched by 39_drug_name_quality_control.py were read one by one and every plausible
# amantadine or agonist spelling was checked against the log.
#
# Amantadine. PK-Merz, the European brand name of amantadine sulfate, was missing,
# although 27_treatments_at_time_zero.py had it, so PATNO 41280 ("PK MERZ" 200 mg from
# 01/2014, active at time-zero and an event at 0.42 years) was counted as a non-user by
# every script importing this module: hence 65 users rather than 66. The separator class
# replaces the earlier "pk.?merz", which matched "pk merz" and "pk-merz" but not
# "pk- merz" (PATNO 42415, whose 200 mg was therefore left inside the levodopa-equivalent
# sum that the Methods say excludes amantadine). "amandat" and "gocov" catch the
# misspellings "amandatine" (PATNO 4022) and "gocoveri" (PATNO 3436). Neither of those
# two changes the counts, because both participants are already found through another
# record, but both would otherwise distort the dose split.
RE_AMANTADINE = r"amantad|amandat|gocov|osmolex|symmetrel|pk[\W_]*merz"

# Agonists. The five misspellings added at the end are all unambiguous and all appear
# once each. Only "roprinirole" changes a reported number: PATNO 3710 is active at
# time-zero and is found by no other record, so the agonist count moves from 226 to 227.
# "Neuropatch" is left out on purpose: it is most likely the Neupro patch, but the name
# is not decisive and the participant starts after time-zero, so nothing rests on it.
# Benztropine is deliberately not matched; note that a bare "ropin" would catch it.
RE_AGONIST = (
    r"pramipex|pramiprex|ropinir|ropiner|ropirin|rotigot|mirapex|sifrol|requip|"
    r"neupro|nupro|apomorph|apokyn|piribedil|clarium|cabergol|pergolid|"
    r"bromocript|pexola|adartrel|trivastal|"
    r"roprinir|pramipl|prampex|rotigin|neurpro"
)

# A drug is active at time-zero if it started no later than one month after
# levodopa initiation and had not stopped more than one month before it.
WINDOW_DAYS = 31


def is_levodopa(name, extended=False):
    """True for a levodopa preparation. Methyldopa is explicitly excluded."""
    s = str(name).lower()
    if "methyldopa" in s:
        return False
    return bool(re.search(RE_LEVODOPA_EXTENDED if extended else RE_LEVODOPA, s))


def _find(folder, pattern):
    import glob
    hits = sorted(glob.glob(os.path.join(RAW, folder, pattern)))
    if not hits:
        raise SystemExit(f"raw table not found: {folder}/{pattern} under {RAW}")
    return hits[-1]


def load_medication_log():
    """Medication log with parsed dates and one time-zero column per participant."""
    path = _find("Historia_Medica_e_Medicacao", "LEDD_Concomitant_Medication_Log_*.csv")
    log = pd.read_csv(path, low_memory=False)
    for col in ("STARTDT", "STOPDT"):
        log[col] = pd.to_datetime(log[col], format="%m/%Y", errors="coerce")
    tzero = (
        log[log.LEDTRT.apply(is_levodopa)]
        .groupby("PATNO")
        .STARTDT.min()
        .rename("tz")
    )
    return log, tzero


def load_part_iv():
    """MDS-UPDRS Part IV, with time in years from levodopa initiation."""
    path = _find("MDS_UPDRS_e_Motor", "MDS-UPDRS_Part_IV__Motor_Complications_*.csv")
    p4 = pd.read_csv(path, low_memory=False)
    p4["INFODT"] = pd.to_datetime(p4.INFODT, format="%m/%Y", errors="coerce")
    _, tzero = load_medication_log()
    p4 = p4.merge(tzero, on="PATNO")
    p4["t"] = (p4.INFODT - p4.tz).dt.days / 365.25
    return p4


def load_curated(columns=None):
    """One row per participant from the curated data cut, taken at the first visit."""
    path = _find("Curated_Data_Cut", "PPMI_Curated_Data_Cut_Public_*.xlsx")
    cols = columns or ["PATNO", "SITE", "COHORT", "subgroup", "enroll_phase", "race", "visit_date"]
    cur = pd.read_excel(path, usecols=cols)
    if "visit_date" in cur.columns:
        cur["visit_date"] = pd.to_datetime(cur.visit_date, errors="coerce")
        cur = cur.sort_values("visit_date")
    return cur.groupby("PATNO").first().reset_index()


def build_cohort(with_metadata=True):
    """
    The development sample: 813 participants and 165 events.

    Rebuilt from the candidate matrix by requiring positive follow-up and complete
    data on the 18-variable candidate pool, which is the definition used throughout
    the manuscript.
    """
    matrix = pd.read_parquet(os.path.join(PROC, "candidate_matrix.parquet"))
    if "ageonset" not in matrix.columns and "ageonset_x" in matrix.columns:
        matrix = matrix.rename(columns={"ageonset_x": "ageonset"})
    cohort = matrix[matrix.exit > 0].dropna(subset=POOL).reset_index(drop=True)
    if with_metadata:
        cohort = cohort.merge(load_curated(), on="PATNO", how="left", suffixes=("", "_cur"))
    return cohort


# ----------------------------------------------------------------------------
# Model fitting and evaluation
# ----------------------------------------------------------------------------


def fit_total6(df, predictors=None, penalizer=RIDGE, standardise=False):
    """Ridge-penalised Cox model. Returns the fitted lifelines object."""
    from lifelines import CoxPHFitter

    cols = list(predictors or TOTAL6)
    data = df[cols + ["exit", "event"]].copy()
    if standardise:
        for c in cols:
            data[c] = (data[c] - data[c].mean()) / data[c].std()
    return CoxPHFitter(penalizer=penalizer).fit(data, "exit", "event")


def linear_predictor(model, df, predictors=None):
    cols = list(predictors or TOTAL6)
    return df[cols].values @ model.params_[cols].values


def out_of_fold_c(df, predictors=None, repeats=CV_REPEATS, folds=CV_FOLDS, return_all=False):
    """
    Out-of-fold concordance, the protocol used for every incremental-value
    comparison in the study: repeated stratified five-fold cross-validation with
    standardisation fitted inside each training fold.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    cols = list(predictors or TOTAL6)
    y = Surv.from_arrays(df.event.astype(bool).values, df.exit.values)
    x = df[cols].values
    scores = []
    for r in range(repeats):
        pred = np.zeros(len(df))
        splitter = StratifiedKFold(folds, shuffle=True, random_state=CV_SEED + r)
        for train, test in splitter.split(x, df.event):
            scaler = StandardScaler().fit(x[train])
            model = CoxPHSurvivalAnalysis(alpha=1.0).fit(scaler.transform(x[train]), y[train])
            pred[test] = model.predict(scaler.transform(x[test]))
        scores.append(
            concordance_index_censored(df.event.astype(bool).values, df.exit.values, pred)[0]
        )
    scores = np.asarray(scores)
    return scores if return_all else float(scores.mean())


def out_of_fold_linear_predictor(df, predictors=None, repeats=10, folds=CV_FOLDS):
    """
    Out-of-fold linear predictor on the original scale, averaged over repetitions.
    Used wherever an absolute risk is needed per participant, which the scaled
    scikit-survival predictions cannot give.
    """
    from sklearn.model_selection import StratifiedKFold

    cols = list(predictors or TOTAL6)
    total = np.zeros(len(df))
    count = np.zeros(len(df))
    for r in range(repeats):
        splitter = StratifiedKFold(folds, shuffle=True, random_state=CV_SEED + r)
        for train, test in splitter.split(df, df.event):
            model = fit_total6(df.iloc[train], cols)
            total[test] += linear_predictor(model, df.iloc[test], cols)
            count[test] += 1
    return total / count


def baseline_survival_at(model, horizon):
    """Baseline survival of a fitted lifelines Cox model at a given time."""
    surv = model.baseline_survival_
    idx = max(np.searchsorted(surv.index.values, horizon, side="right") - 1, 0)
    return float(surv.iloc[idx, 0])


def absolute_risk(model, lp, horizon, centre):
    """1 - S0(t) ** exp(lp - centre), the risk formula used by the calculator."""
    return 1.0 - baseline_survival_at(model, horizon) ** np.exp(lp - centre)


def km_incidence(df, horizon):
    """Observed cumulative incidence at a horizon, by Kaplan-Meier."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(df.exit, df.event)
    return 1.0 - float(km.survival_function_at_times(horizon).iloc[0])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
