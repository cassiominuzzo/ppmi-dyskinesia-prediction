# figuras_suplementares.R
# Produces: Supplementary Figure S1 (internal calibration at five years) and
#           Supplementary Figure S2 (head-to-head with published predictor sets),
#           each as PNG, SVG and PDF.
#
# Replaces 23_supplementary_figure_S1.py and 24_supplementary_figure_S2.py, which
# pointed at the previous project layout and, more importantly, were drawn from
# computations that no longer match the paper:
#
#   S1 was drawn from a five-repetition KFold of its own, and its legend quoted an
#      observed-to-expected ratio of 0.98 and a slope of 1.05. The paper reports 1.02
#      and 1.07, from the twenty-repetition stratified protocol of 32_internal_
#      performance.py. This script rebuilds the figure from that protocol's stored
#      out-of-fold linear predictor, and prints the check: mean predicted risk and O:E
#      must equal the values in tab32_internal_performance.json.
#
#   S2 was drawn from tab67_head_to_head_CI.csv, which used a different complete-case
#      sample and a different cross-validation from the table beside it, so the same
#      six predictor sets carried two different C-indices in the same supplement. It is
#      now drawn from 49_head_to_head_unified.py, the single analysis behind both.
#
# Inputs, all under 03_Resultados/Tabelas/ except the cohort:
#   tab32_out_of_fold_linear_predictor.csv   out-of-fold linear predictor per participant
#   tab32_internal_performance.json          baseline survival, slope, ICI, O:E
#   tab49_sets_unified.csv                   the seven predictor sets
#   04_Dados_processados/candidate_matrix.parquet
#
# Usage:  Rscript figuras_suplementares.R

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
TAB <- file.path(PROJ, "03_Resultados", "Tabelas")
OUT <- file.path(PROJ, "03_Resultados", "Figuras")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
cat("project root:", PROJ, "\n")

for (p in c("nanoparquet", "jsonlite"))
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
suppressMessages({ library(nanoparquet); library(jsonlite); library(survival) })
Sys.setenv(TZ = "UTC")

TEAL <- "#1B7A6B"; GOLD <- "#B08D2E"; RED <- "#B23A38"
NAVY <- "#1F4E79"; GREY <- "#6B6B6B"; INK <- "#1A1A1A"; PALE <- "#EDF3FA"

POOL <- c("updrs_totscore","updrs1_score","updrs2_score","updrs3_score","comp_bradicinesia",
          "td_pigd_ratio","pigd","MSEADLG","hy","Clinical_Stage","NP1FATG","NP1DPRS",
          "NP2FREZ","ageonset","SEX","BMI","scopa_gi","EDUCYRS")
TOTAL6 <- c("updrs_totscore","ageonset","SEX","BMI","td_pigd_ratio","NP2FREZ")
BETA <- c(0.01820067483435665, -0.030261443489475598, -0.2919818089443341,
          -0.034380328769057285, -0.0649687671095703, 0.25736201135635567)
HORIZON <- 5

# ============================================================ inputs, figure S1 =====
perf <- fromJSON(file.path(TAB, "tab32_internal_performance.json"))
S0 <- perf$calibration[["5"]]$S0
SLOPE <- perf$slope_out_of_fold
ICI <- perf$calibration[["5"]]$ICI

m <- read_parquet(file.path(PROJ, "04_Dados_processados", "candidate_matrix.parquet"))
if (!"ageonset" %in% names(m) && "ageonset_x" %in% names(m))
  names(m)[names(m) == "ageonset_x"] <- "ageonset"
m$PATNO <- as.integer(m$PATNO)
coh <- m[m$exit > 0 & stats::complete.cases(m[, POOL]), ]
lp <- read.csv(file.path(TAB, "tab32_out_of_fold_linear_predictor.csv"))
lp$PATNO <- as.integer(lp$PATNO)
d <- merge(coh[, c("PATNO", "exit", "event", TOTAL6)], lp, by = "PATNO")
stopifnot(nrow(d) == 813, sum(d$event) == 165)

# The baseline survival is centred at the covariate means, so the linear predictor has to
# be shifted by the same amount before it is exponentiated. Getting this wrong is silent:
# the curve simply sits in the wrong place. The check below is what makes it visible.
CENTRE <- sum(colMeans(as.matrix(d[, TOTAL6])) * BETA)
d$risk <- 1 - S0 ^ exp(d$lp_out_of_fold - CENTRE)
km_all <- survfit(Surv(exit, event) ~ 1, data = d, conf.type = "log-log")
OBS <- 1 - summary(km_all, times = HORIZON)$surv
OE <- OBS / mean(d$risk)
cat(sprintf("\ncalibration check against tab32_internal_performance.json\n"))
cat(sprintf("   mean predicted risk %.6f   (stored %.6f)\n", mean(d$risk),
            perf$calibration[["5"]]$predicted_mean))
cat(sprintf("   observed by KM      %.6f   (stored %.6f)\n", OBS,
            perf$calibration[["5"]]$observed))
cat(sprintf("   observed to expected %.4f    (stored %.4f)\n", OE, perf$calibration[["5"]]$OE))
stopifnot(abs(mean(d$risk) - perf$calibration[["5"]]$predicted_mean) < 1e-6,
          abs(OE - perf$calibration[["5"]]$OE) < 1e-4)

d$q <- cut(d$risk, breaks = quantile(d$risk, seq(0, 1, 0.2)), labels = FALSE, include.lowest = TRUE)
Q <- do.call(rbind, lapply(1:5, function(g) {
  s <- d[d$q == g, ]
  k <- survfit(Surv(exit, event) ~ 1, data = s, conf.type = "log-log")
  z <- summary(k, times = HORIZON)
  data.frame(q = g, n = nrow(s), pred = mean(s$risk), obs = 1 - z$surv,
             lo = 1 - z$upper, hi = 1 - z$lower)
}))
cat("\nquintiles of out-of-fold predicted risk\n")
print(within(Q, { pred <- round(pred, 4); obs <- round(obs, 4); lo <- round(lo, 4); hi <- round(hi, 4) }),
      row.names = FALSE)

# The axis is sized to the quintile points and their confidence limits, not to the highest
# predicted risk. A handful of participants are predicted far above the rest, and letting
# them set the scale would squeeze every point that carries information into one corner.
# How many fall outside is counted and stated on the figure, so nothing is hidden.
TOP <- ceiling(max(Q$hi, Q$pred) * 20) / 20 + 0.05
ABOVE <- sum(d$risk > TOP)
cat(sprintf("\naxis to %.2f; highest observed upper limit %.3f; highest predicted risk %.3f;\n",
            TOP, max(Q$hi), max(d$risk)))
cat(sprintf("   %d of %d participants (%.1f%%) are predicted above the axis, counted in the last bin\n",
            ABOVE, nrow(d), 100 * ABOVE / nrow(d)))

draw_s1 <- function() {
  layout(matrix(1:2, ncol = 1), heights = c(4.0, 1.15))
  par(mar = c(3.9, 5.2, 1.2, 1.6), mgp = c(2.9, 0.75, 0), las = 1)
  lim <- c(0, TOP)
  plot(NA, xlim = lim, ylim = lim, xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n")
  at <- seq(0, floor(TOP * 10) / 10, 0.1)
  axis(1, at = at, labels = paste0(at * 100, "%"), cex.axis = 1.02, lwd = 1.2)
  axis(2, at = at, labels = paste0(at * 100, "%"), cex.axis = 1.02, lwd = 1.2)
  mtext("Predicted five-year risk, out of fold", side = 1, line = 2.5, cex = 1.08)
  mtext("Observed five-year risk\n(Kaplan-Meier)", side = 2, line = 2.6, cex = 1.08, las = 0)
  abline(0, 1, lty = 2, lwd = 1.6, col = GREY)
  segments(Q$pred, Q$lo, Q$pred, Q$hi, lwd = 2.2, col = NAVY)
  segments(Q$pred - 0.007, Q$lo, Q$pred + 0.007, Q$lo, lwd = 2.2, col = NAVY)
  segments(Q$pred - 0.007, Q$hi, Q$pred + 0.007, Q$hi, lwd = 2.2, col = NAVY)
  lines(Q$pred, Q$obs, col = NAVY, lwd = 1.4, lty = 3)
  points(Q$pred, Q$obs, pch = 21, bg = TEAL, col = NAVY, cex = 1.7, lwd = 1.8)
  text(Q$pred, Q$hi + 0.028, sprintf("Q%d", Q$q), cex = 0.92, col = GREY)
  # near the line it names, but clear of it: the diagonal runs through any label placed on it
  text(TOP * 0.96, TOP * 0.80, "perfect calibration", adj = c(1, 0.5), font = 3,
       col = GREY, cex = 0.95)
  legend("topleft", bty = "n", cex = 1.0, text.col = INK, y.intersp = 1.35,
         legend = c(sprintf("Observed to expected  %.2f", OE),
                    sprintf("Calibration slope  %.2f", SLOPE),
                    sprintf("Integrated calibration index  %.1f percentage points", 100 * ICI)))

  # what the model actually predicted, so the reader can see where the quintiles sit
  par(mar = c(2.6, 5.2, 0.4, 1.6))
  h <- hist(pmin(d$risk, TOP), breaks = seq(0, TOP, length.out = 31), plot = FALSE)
  plot(NA, xlim = lim, ylim = c(0, max(h$counts) * 1.25), axes = FALSE, xlab = "", ylab = "")
  rect(h$breaks[-length(h$breaks)], 0, h$breaks[-1], h$counts, col = PALE, border = NAVY, lwd = 0.9)
  axis(1, at = at, labels = paste0(at * 100, "%"), cex.axis = 0.95, lwd = 1.2)
  mtext("Distribution of\npredicted risk", side = 2, line = 2.6, cex = 0.95, las = 0, col = GREY)
  abline(v = quantile(d$risk, seq(0.2, 0.8, 0.2)), lty = 3, col = GREY, lwd = 1.1)
  if (ABOVE > 0)
    text(TOP, max(h$counts) * 1.2, sprintf("%d participants above %.0f%%\nare counted in the last bin",
                                           ABOVE, 100 * TOP),
         adj = c(1, 1), cex = 0.85, font = 3, col = GREY)
}

# ============================================================ inputs, figure S2 =====
S <- read.csv(file.path(TAB, "tab49_sets_unified.csv"), stringsAsFactors = FALSE)
S <- S[order(-S$C), ]
S$label <- sprintf("%s (%d)", S$predictor_set, S$k)
S$label <- sub("Total-6 \\(this study\\)", "Total-6, this study", S$label)
cat("\nhead-to-head sets, ordered by out-of-fold C\n")
print(S[, c("predictor_set", "k", "C", "C_lo", "C_hi", "dC", "d_lo", "d_hi", "p")], row.names = FALSE)

draw_s2 <- function() {
  n <- nrow(S); y <- n:1
  is_ref <- grepl("^Total-6", S$predictor_set)
  col <- ifelse(is_ref, TEAL, NAVY)
  layout(matrix(1:2, nrow = 1), widths = c(1.34, 1))
  # ---- left: the C-index of each set
  par(mar = c(4.3, 12.4, 2.4, 0.8), mgp = c(2.7, 0.7, 0), las = 1)
  plot(NA, xlim = c(0.53, 0.78), ylim = c(0.4, n + 0.6), yaxt = "n", xaxt = "n",
       xlab = "", ylab = "", bty = "n")
  axis(1, at = seq(0.55, 0.75, 0.05), cex.axis = 1.0, lwd = 1.2)
  mtext("Out-of-fold C-index (95% CI)", side = 1, line = 2.4, cex = 1.02)
  mtext("A", side = 3, line = 0.9, at = 0.50, cex = 1.35, font = 2, col = INK)
  abline(v = 0.5, col = GREY, lty = 3)
  axis(2, at = y, labels = S$label, tick = FALSE, line = -0.4, cex.axis = 1.0,
       col.axis = INK, hadj = 1)
  segments(S$C_lo, y, S$C_hi, y, lwd = 2.4, col = col)
  segments(S$C_lo, y - 0.13, S$C_lo, y + 0.13, lwd = 2.4, col = col)
  segments(S$C_hi, y - 0.13, S$C_hi, y + 0.13, lwd = 2.4, col = col)
  points(S$C, y, pch = ifelse(is_ref, 23, 21), bg = col, col = col, cex = ifelse(is_ref, 1.7, 1.4))
  text(0.78, y, sprintf("%.3f", S$C), adj = c(1, 0.5), cex = 0.92, col = GREY)

  # ---- right: the paired difference against the Total-6
  par(mar = c(4.3, 0.8, 2.4, 5.4))
  k <- !is_ref
  plot(NA, xlim = c(-0.05, 0.16), ylim = c(0.4, n + 0.6), yaxt = "n", xaxt = "n",
       xlab = "", ylab = "", bty = "n")
  axis(1, at = seq(-0.04, 0.14, 0.04), cex.axis = 1.0, lwd = 1.2)
  mtext("Difference from the Total-6 (95% CI)", side = 1, line = 2.4, cex = 1.02)
  mtext("B", side = 3, line = 0.9, at = -0.075, cex = 1.35, font = 2, col = INK)
  abline(v = 0, lwd = 1.4, col = GREY)
  sig <- k & S$d_lo > 0
  cc <- ifelse(sig, RED, GOLD)
  segments(S$d_lo[k], y[k], S$d_hi[k], y[k], lwd = 2.4, col = cc[k])
  segments(S$d_lo[k], y[k] - 0.13, S$d_lo[k], y[k] + 0.13, lwd = 2.4, col = cc[k])
  segments(S$d_hi[k], y[k] - 0.13, S$d_hi[k], y[k] + 0.13, lwd = 2.4, col = cc[k])
  points(S$dC[k], y[k], pch = 21, bg = cc[k], col = cc[k], cex = 1.4)
  text(0.163, y[is_ref], "reference", adj = c(0, 0.5), cex = 0.95, font = 3, col = GREY, xpd = NA)
  text(0.163, y[k], sprintf("p = %.3f", S$p[k]), adj = c(0, 0.5), cex = 0.95,
       col = ifelse(sig[k], INK, GREY), xpd = NA)
  text(0, n + 0.55, "favours the Total-6", adj = c(-0.06, 0.5), cex = 0.92, font = 3, col = GREY)
}

# ------------------------------------------------------------------ output ---------
for (fig in list(list(f = draw_s1, nm = "FigureS1_internal_calibration", w = 7.6, h = 8.2),
                 list(f = draw_s2, nm = "FigureS2_head_to_head",         w = 12.4, h = 5.4))) {
  png(file.path(OUT, paste0(fig$nm, ".png")), width = fig$w, height = fig$h, units = "in", res = 200)
  fig$f(); dev.off()
  svg(file.path(OUT, paste0(fig$nm, ".svg")), width = fig$w, height = fig$h); fig$f(); dev.off()
  pdf(file.path(OUT, paste0(fig$nm, ".pdf")), width = fig$w, height = fig$h); fig$f(); dev.off()
  cat("saved", fig$nm, "as png, svg and pdf\n")
}
