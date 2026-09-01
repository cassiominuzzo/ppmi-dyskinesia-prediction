# amantadina_pkmerz.R
# Produces: every amantadine number that changes once PK-Merz is added to the drug
# search of cohort.py, and the proof that the method reproduces the published values.
#
# PK-Merz is the European brand name of amantadine sulfate. It was present in
# 27_treatments_at_time_zero.py but missing from cohort.py, so every script that
# imports cohort.py, including 31_amantadine_handling.py, counted 65 baseline users
# instead of 66. The participant it missed at time-zero is PATNO 41280, who reached
# the outcome, so the crude and adjusted associations both move.
#
# The script runs each analysis twice, with the old pattern and with the corrected
# one. With the old pattern it must return the numbers stored in
# 03_Resultados/Tabelas/tab31_amantadine.json; that check is printed, and it is what
# licenses the corrected values. The ridge fit reproduces lifelines exactly: lifelines
# normalises the design matrix internally and penalises the normalised coefficients
# with n * penalizer * 0.5 * sum(beta^2), which is what negll below implements.
#
# Not reproducible here, and therefore still requiring 31_amantadine_handling.py to be
# re-run in Python: the three quantities built on repeated five-fold cross-validation
# (delta C for amantadine as a seventh covariate, the refit among non-users, and the
# extreme reclassification bound), because they depend on the fold seeding of
# scikit-learn.
#
# Usage:  Rscript amantadina_pkmerz.R

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

if (!requireNamespace("nanoparquet", quietly = TRUE))
  install.packages("nanoparquet", repos = "https://cloud.r-project.org")
suppressMessages({ library(nanoparquet); library(survival) })
Sys.setenv(TZ = "UTC")

OLD <- "amantad|gocovri|osmolex|symmetrel"
NEW <- "amantad|amandat|gocov|osmolex|symmetrel|pk[^[:alnum:]]*merz"   # the audited search now in cohort.py
WINDOW <- 31
POOL <- c("updrs_totscore","updrs1_score","updrs2_score","updrs3_score","comp_bradicinesia",
          "td_pigd_ratio","pigd","MSEADLG","hy","Clinical_Stage","NP1FATG","NP1DPRS",
          "NP2FREZ","ageonset","SEX","BMI","scopa_gi","EDUCYRS")
TOTAL6 <- c("updrs_totscore","ageonset","SEX","BMI","td_pigd_ratio","NP2FREZ")
BETA <- c(0.01820067483435665, -0.030261443489475598, -0.2919818089443341,
          -0.034380328769057285, -0.0649687671095703, 0.25736201135635567)
RIDGE <- 0.05

# ------------------------------------------------------------------- inputs -----
M <- read_parquet(file.path(PROJ, "04_Dados_processados", "candidate_matrix.parquet"))
if (!"ageonset" %in% names(M) && "ageonset_x" %in% names(M))
  names(M)[names(M) == "ageonset_x"] <- "ageonset"
M$PATNO <- as.integer(M$PATNO)
coh <- M[M$exit > 0 & stats::complete.cases(M[, POOL]), ]
stopifnot(nrow(coh) == 813, sum(coh$event) == 165)

tz <- read_parquet(file.path(PROJ, "04_Dados_processados", "tzero.parquet"))[, c("PATNO", "tzero_levodopa")]
tz <- tz[!is.na(tz$tzero_levodopa), ]; tz$PATNO <- as.integer(tz$PATNO)
L <- read.csv(file.path(RAWROOT, "Historia_Medica_e_Medicacao",
                        "LEDD_Concomitant_Medication_Log_29Apr2026.csv"), stringsAsFactors = FALSE)
p4 <- read.csv(file.path(RAWROOT, "MDS_UPDRS_e_Motor",
                         "MDS-UPDRS_Part_IV__Motor_Complications_29Apr2026.csv"), stringsAsFactors = FALSE)
md <- function(x) { x <- as.character(x); as.Date(ifelse(is.na(x) | x == "", NA, paste0("01/", x)), "%d/%m/%Y") }
L$start <- md(L$STARTDT); L$stop <- md(L$STOPDT)
L$LEDD <- suppressWarnings(as.numeric(L$LEDD)); L$PATNO <- suppressWarnings(as.integer(L$PATNO))
L <- merge(L[!is.na(L$PATNO), ], tz, by = "PATNO"); L$tzd <- as.Date(L$tzero_levodopa)
L <- L[L$PATNO %in% coh$PATNO, ]
L$active <- !is.na(L$start) & L$start <= L$tzd + WINDOW & (is.na(L$stop) | L$stop >= L$tzd - WINDOW)
lo <- tolower(L$LEDTRT)
p4$PATNO <- as.integer(p4$PATNO); p4$INFODT <- md(p4$INFODT)
p4 <- merge(p4, tz, by = "PATNO"); p4$t <- as.numeric(p4$INFODT - as.Date(p4$tzero_levodopa)) / 365.25

cat("\n=== records that only the corrected pattern finds, among the 813 ===\n")
print(L[grepl(NEW, lo) & !grepl(OLD, lo), c("PATNO", "LEDTRT", "LEDD", "STARTDT", "STOPDT", "active")],
      row.names = FALSE)

# ------------------------------------------ ridge Cox, as lifelines implements it ---
mk <- function(X, time, ev) { o <- order(time)
  list(X = X[o, , drop = FALSE], time = time[o], ev = ev[o], ut = sort(unique(time[ev == 1]))) }
negll <- function(b, D, lam, w) {                       # Efron partial likelihood
  eta <- as.vector(D$X %*% b); e <- exp(eta); ll <- 0
  for (t in D$ut) {
    R <- D$time >= t; Dj <- D$time == t & D$ev == 1; m <- sum(Dj)
    S <- sum(e[R]); Sd <- sum(e[Dj])
    ll <- ll + sum(eta[Dj]) - sum(log(S - (0:(m - 1)) / m * Sd))
  }
  -(ll - lam * 0.5 * sum((b * w) ^ 2))                  # penalty on the normalised scale
}
ridge_cox <- function(X, time, ev, penalizer = RIDGE) {
  D <- mk(X, time, ev); w <- apply(X, 2, sd); lam <- penalizer * nrow(X)
  b <- optim(rep(0, ncol(X)), negll, D = D, lam = lam, w = w, method = "BFGS",
             control = list(maxit = 5000, reltol = 1e-15))$par
  h <- 1e-5; p <- length(b); H <- matrix(0, p, p)
  for (i in 1:p) for (j in i:p) {
    f <- function(di, dj) { bb <- b; bb[i] <- bb[i] + di * h; bb[j] <- bb[j] + dj * h; negll(bb, D, lam, w) }
    H[i, j] <- H[j, i] <- (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * h * h)
  }
  se <- sqrt(diag(solve(H))); names(b) <- names(se) <- colnames(X)
  list(b = b, se = se)
}
# the fit above must return the published Total-6 coefficients
chk <- ridge_cox(as.matrix(coh[, TOTAL6]), coh$exit, coh$event)
cat(sprintf("\nridge check against the published Total-6: max absolute difference = %.2e\n",
            max(abs(chk$b - BETA))))
stopifnot(max(abs(chk$b - BETA)) < 1e-4)

X <- as.matrix(coh[, TOTAL6])
lp <- as.vector(sweep(X, 2, colMeans(X)) %*% BETA)      # frozen model, centred as in Table 2

run <- function(pattern, tag) {
  hit <- grepl(pattern, lo)
  users <- unique(L$PATNO[hit & L$active])
  later_ids <- setdiff(unique(L$PATNO[hit & !is.na(L$start) & L$start > L$tzd + WINDOW]), users)
  d <- coh; d$AM <- as.integer(d$PATNO %in% users)

  cr <- summary(coxph(Surv(exit, event) ~ AM, data = d))
  Z <- cbind(scale(as.matrix(d[, TOTAL6])), AM = d$AM)
  f <- ridge_cox(Z, d$exit, d$event)
  b <- f$b[["AM"]]; s <- f$se[["AM"]]; q <- qnorm(0.975)
  cn <- function(k) survival::concordance(Surv(d$exit[k], d$event[k]) ~ lp[k], reverse = TRUE)$concordance

  st <- d[d$PATNO %in% later_ids, ]
  amst <- sapply(st$PATNO, function(p) {
    k <- L$PATNO == p
    min(as.numeric(L$start[k & hit & !is.na(L$start) & L$start > L$tzd + WINDOW] - L$tzd[k][1])) / 365.25 })
  ev_before <- sum(st$event == 1 & st$exit <= amst)
  ev_after  <- sum(st$event == 1 & st$exit >  amst)
  free <- st$PATNO[st$event == 0]; free_s <- amst[st$event == 0]
  worst <- mapply(function(p, a) {
    h <- p4[p4$PATNO == p & p4$t > 0 & p4$t <= a, ]
    if (!nrow(h)) NA else suppressWarnings(max(c(h$NP4DYSKI, h$NP4WDYSK), na.rm = TRUE)) }, free, free_s)

  A <- L[L$active, ]
  s2 <- function(keep) { v <- tapply(ifelse(keep, A$LEDD, NA), A$PATNO, function(z) sum(z, na.rm = TRUE))
                         o <- v[as.character(coh$PATNO)]; names(o) <- coh$PATNO; o }
  tot <- s2(rep(TRUE, nrow(A)))
  am  <- s2(grepl(pattern, tolower(A$LEDTRT))); am <- am[!is.na(am) & am > 0]
  nd  <- s2(!grepl(pattern, tolower(A$LEDTRT))); nd <- nd[!is.na(nd) & nd > 0]

  cat(sprintf("\n--- %s ---\n", tag))
  cat(sprintf("  baseline users                  %d (%.1f%%)\n", sum(d$AM), 100 * mean(d$AM)))
  cat(sprintf("  A. crude HR                     %.4f (%.4f to %.4f), p = %.5f\n",
              cr$conf.int["AM", 1], cr$conf.int["AM", 3], cr$conf.int["AM", 4], cr$coefficients["AM", 5]))
  cat(sprintf("  A. adjusted HR, ridge %.2f      %.4f (%.4f to %.4f), p = %.5f\n", RIDGE,
              exp(b), exp(b - q * s), exp(b + q * s), 2 * pnorm(-abs(b / s))))
  cat(sprintf("  B. frozen model, %d non-users  C %.4f | %d users C %.4f\n",
              sum(d$AM == 0), cn(d$AM == 0), sum(d$AM == 1), cn(d$AM == 1)))
  cat(sprintf("  C. started during follow-up     %d (%d event before, %d after, %d event free)\n",
              nrow(st), ev_before, ev_after, length(free)))
  cat(sprintf("     of the event free: %d scored 1 or more before starting, %d scored 0, %d unassessed\n",
              sum(worst >= 1, na.rm = TRUE), sum(worst == 0, na.rm = TRUE), sum(is.na(worst))))
  cat(sprintf("     reclassifying the plausible: %d events | reclassifying all: %d events\n",
              165 + sum(worst >= 1, na.rm = TRUE), 165 + length(free)))
  cat(sprintf("  D. amantadine median %.0f mg (IQR %.0f to %.0f) of a median total of %.0f mg, %.0f%%\n",
              median(am), quantile(am, .25), quantile(am, .75),
              median(tot[names(am)], na.rm = TRUE), median(100 * am / tot[names(am)], na.rm = TRUE)))
  cat(sprintf("  D. dopaminergic dose with amantadine removed: %.0f [%.0f-%.0f] mg, n = %d\n",
              median(nd), quantile(nd, .25), quantile(nd, .75), length(nd)))
}

cat("\npublished in tab31_amantadine.json, all produced with the old pattern:\n")
cat("  users 65 | crude 1.7570 (1.1622 to 2.6562) p 0.00753 | adjusted 1.4450 (0.9682 to 2.1567) p 0.07155\n")
cat("  frozen C 0.6933 in 748 and 0.7423 in 65 | 89 starters (22 / 31 / 36) | 10 with a prior score, 22 zero, 4 none\n")
run(OLD, "OLD PATTERN, the published analysis")
run(NEW, "CORRECTED PATTERN, with PK-Merz")
cat("\nStill to be re-run in Python, all built on repeated five-fold cross-validation:\n")
cat("  delta C for amantadine as a seventh covariate, C refit among non-users, extreme reclassification bound\n")
