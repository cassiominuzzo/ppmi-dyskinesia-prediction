# dose_sem_amantadina.R
# Produces: the treatment rows of Table 1, with the levodopa-equivalent daily dose
# recomputed after removing amantadine from the conversion.
#
# SUPERSEDED on 31 August 2026 by 27_treatments_at_time_zero.py, which was rewritten
# to import the audited drug searches from cohort.py and now prints all five rows,
# the amantadine-free dose included. This script is kept as an independent check in a
# second language: it must agree with the Python one row for row. The searches below
# are the audited ones, copied from cohort.py, and are no longer the narrower upper-case
# copies that this script and script 27 used to share.
#
# It follows script 27 in everything else: the same one-month activity window, the same
# numeric coercion of the LEDD column (entries such as "LD x 0.33" become missing), and
# the same rule of summarising only participants with a positive total.
#
# The values now produced by script 27 are printed alongside as a check.

find_root <- function() {
  p <- Sys.getenv("LID_PROJECT"); if (nzchar(p)) return(p)
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  f <- if (length(a)) sub("^--file=", "", a[1]) else NULL
  if (is.null(f)) for (i in seq_len(sys.nframe())) {
    o <- get0("ofile", envir = sys.frame(i), inherits = FALSE); if (!is.null(o)) { f <- o; break } }
  if (is.null(f)) return("C:/Users/cassi/OneDrive/Área de Trabalho/PPMI")
  normalizePath(file.path(dirname(normalizePath(f, winslash = "/")), "..", "..", ".."),
                winslash = "/", mustWork = TRUE)
}
PROJ <- find_root()
RAWROOT <- Sys.getenv("LID_RAW", unset = paste0(
  "C:/Users/cassi/OneDrive/Área de Trabalho/Cowork Cloude/IC & Meta-Análises/",
  "Discinesia Problemática - PPMI/01_Dados_Brutos"))
cat("project root:", PROJ, "\n")

if (!requireNamespace("nanoparquet", quietly = TRUE)) install.packages("nanoparquet", repos = "https://cloud.r-project.org")
suppressMessages(library(nanoparquet))
Sys.setenv(TZ = "UTC")

AGONIST    <- paste0("pramipex|pramiprex|ropinir|ropiner|ropirin|rotigot|mirapex|sifrol|requip|",
                     "neupro|nupro|apomorph|apokyn|piribedil|clarium|cabergol|pergolid|",
                     "bromocript|pexola|adartrel|trivastal|roprinir|pramipl|prampex|rotigin|neurpro")
AMANTADINE <- "amantad|amandat|gocov|osmolex|symmetrel|pk[^[:alnum:]]*merz"
LEVODOPA   <- paste0("dopa|carb.{0,4}lev|lev.{0,3}dop|sinemet|madopar|stalevo|rytary|duopa|",
                     "prolopa|nacom|benseraz|parcopa|isicom|careldopa|dopicar|levocomp|sinement")
WINDOW <- 31

# ------------------------------------------------------------------- inputs -----
L <- read.csv(file.path(RAWROOT, "Historia_Medica_e_Medicacao",
                        "LEDD_Concomitant_Medication_Log_29Apr2026.csv"), stringsAsFactors = FALSE)
tz <- read_parquet(file.path(PROJ, "04_Dados_processados", "tzero.parquet"))[, c("PATNO", "tzero_levodopa")]
tz <- tz[!is.na(tz$tzero_levodopa), ]; tz$PATNO <- as.integer(tz$PATNO)

nm <- tolower(as.character(L$LEDTRT))
L$is_agonist    <- grepl(AGONIST, nm)
L$is_amantadine <- grepl(AMANTADINE, nm)
# methyldopa is excluded by name, as cohort.py::is_levodopa does
L$is_levodopa   <- grepl(LEVODOPA, nm) & !grepl("methyldopa", nm) & !L$is_agonist & !L$is_amantadine
md <- function(x) { x <- as.character(x); as.Date(ifelse(is.na(x) | x == "", NA, paste0("01/", x)), "%d/%m/%Y") }
L$start <- md(L$STARTDT); L$stop <- md(L$STOPDT)
L$LEDD  <- suppressWarnings(as.numeric(L$LEDD))
L$PATNO <- suppressWarnings(as.integer(L$PATNO))
L <- merge(L[!is.na(L$PATNO), ], tz, by = "PATNO")
tzd <- as.Date(L$tzero_levodopa)
act <- !is.na(L$start) & L$start <= tzd + WINDOW & (is.na(L$stop) | L$stop >= tzd - WINDOW)
A <- L[act, ]

# --------------------------------------------------------- development sample ---
POOL <- c("updrs_totscore","updrs1_score","updrs2_score","updrs3_score","comp_bradicinesia",
          "td_pigd_ratio","pigd","MSEADLG","hy","Clinical_Stage","NP1FATG","NP1DPRS",
          "NP2FREZ","ageonset","SEX","BMI","scopa_gi","EDUCYRS")
M <- read_parquet(file.path(PROJ, "04_Dados_processados", "candidate_matrix.parquet"))
if (!"ageonset" %in% names(M) && "ageonset_x" %in% names(M)) names(M)[names(M) == "ageonset_x"] <- "ageonset"
M$PATNO <- as.integer(M$PATNO)
dev <- M[M$exit > 0 & stats::complete.cases(M[, POOL]), "PATNO"]
stopifnot(length(dev) == 813)

s <- function(keep) {
  v <- tapply(ifelse(keep, A$LEDD, NA), A$PATNO, function(z) sum(z, na.rm = TRUE))
  out <- v[as.character(dev)]; names(out) <- dev; out
}
lev   <- s(A$is_levodopa)
total <- s(rep(TRUE, nrow(A)))
noam  <- s(!A$is_amantadine)

report <- function(x, label, published) {
  x <- x[!is.na(x) & x > 0]
  cat(sprintf("  %-52s %3.0f [%3.0f-%3.0f]   n = %3d   (published: %s)\n",
              label, median(x), quantile(x, .25), quantile(x, .75), length(x), published))
}
cat("\n=== treatment doses at time-zero, development sample ===\n")
report(lev,   "Levodopa daily dose, mg",                          "300 [250-450], n = 805")
report(total, "Levodopa-equivalent daily dose, amantadine included", "400 [300-600], n = 809")
report(noam,  "Levodopa-equivalent daily dose, amantadine EXCLUDED", "400 [300-600], n = 808")
ag <- unique(A$PATNO[A$is_agonist]); ag <- ag[ag %in% dev]
cat(sprintf("  %-52s %d (%.0f%%)   %13s   (script 27: 227 (28%%))\n",
            "Dopamine agonist at time-zero, n (%)", length(ag), 100 * length(ag) / length(dev), ""))

am <- s(A$is_amantadine); am <- am[!is.na(am) & am > 0]
cat(sprintf("\n  participants on amantadine at time-zero: %d\n", length(am)))
cat(sprintf("  amantadine contributed a median of %.0f mg (IQR %.0f to %.0f) in those participants\n",
            median(am), quantile(am, .25), quantile(am, .75)))
sh <- 100 * median(am / total[names(am)], na.rm = TRUE)
cat(sprintf("  which is a median of %.0f%% of their converted dose\n", sh))
d <- total[names(am)] - noam[names(am)]
cat(sprintf("  removing it lowers their total by a median of %.0f mg\n", median(d, na.rm = TRUE)))
