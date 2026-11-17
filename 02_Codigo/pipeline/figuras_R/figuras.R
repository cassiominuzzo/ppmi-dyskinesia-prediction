# figuras.R
# Produces: Figure 2 (cumulative incidence by tertile of model-predicted risk) and
# Supplementary Figure S2 (decision curve), each as PNG at 300 dpi, SVG and PDF.
#
# Figure 1 is drawn by 38_figure1_participant_flow.py, which takes its counts from the
# analysis base; draw_fig1 below is kept only as a cross-check and is no longer written
# to disk, so the two cannot overwrite each other. Corrected in November 2026, when the
# figures moved to 300 dpi and the decision curve moved to the supplement as
# Supplementary Figure S2.
#
# Replaces 25_figure2_km_tertiles.py, which pointed at the previous project layout and
# could not run as shipped. Figure 2 also
# had the "Intermediate risk" label of the numbers-at-risk table overlapping the
# first count; here the label column is sized from the rendered text width, so the
# collision cannot recur.
#
# Only two inputs are needed:
#   04_Dados_processados/candidate_matrix.parquet   the stored cohort
#   the frozen Total-6 coefficients, restated below and identical to Table 2
#
# Usage:  Rscript figuras.R            (run from anywhere; set PROJ below if needed)

# Where is the project? The script sits in PPMI/02_Codigo/pipeline/figuras_R/, so the
# project root is three levels up. This is worked out from the script's own location,
# which keeps it working whatever the folder is called and avoids depending on how the
# console encodes accented characters in a hard-coded path.
find_root <- function() {
  p <- Sys.getenv("LID_PROJECT")
  if (nzchar(p)) return(p)
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)   # Rscript
  f <- if (length(a)) sub("^--file=", "", a[1]) else NULL
  if (is.null(f)) for (i in seq_len(sys.nframe())) {                        # source()
    o <- get0("ofile", envir = sys.frame(i), inherits = FALSE)
    if (!is.null(o)) { f <- o; break }
  }
  if (is.null(f)) return("C:/Users/cassi/OneDrive/Área de Trabalho/PPMI")
  normalizePath(file.path(dirname(normalizePath(f, winslash = "/")), "..", "..", ".."),
                winslash = "/", mustWork = TRUE)
}
PROJ <- find_root()
cat("project root:", PROJ, "\n")
PARQUET <- file.path(PROJ, "04_Dados_processados", "candidate_matrix.parquet")
OUT     <- file.path(PROJ, "03_Resultados", "Figuras")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

if (!requireNamespace("nanoparquet", quietly = TRUE)) install.packages("nanoparquet", repos = "https://cloud.r-project.org")
suppressMessages({ library(nanoparquet); library(survival) })

TEAL <- "#1B7A6B"; GOLD <- "#B08D2E"; RED <- "#B23A38"
NAVY <- "#1F4E79"; GREY <- "#6B6B6B"; INK <- "#1A1A1A"

# ---------------------------------------------------------------- the cohort ----
POOL <- c("updrs_totscore","updrs1_score","updrs2_score","updrs3_score","comp_bradicinesia",
          "td_pigd_ratio","pigd","MSEADLG","hy","Clinical_Stage","NP1FATG","NP1DPRS",
          "NP2FREZ","ageonset","SEX","BMI","scopa_gi","EDUCYRS")
TOTAL6 <- c("updrs_totscore","ageonset","SEX","BMI","td_pigd_ratio","NP2FREZ")
BETA <- c(updrs_totscore = 0.01820067483435665, ageonset = -0.030261443489475598,
          SEX = -0.2919818089443341, BMI = -0.034380328769057285,
          td_pigd_ratio = -0.0649687671095703, NP2FREZ = 0.25736201135635567)

m <- read_parquet(PARQUET)
if (!"ageonset" %in% names(m) && "ageonset_x" %in% names(m)) names(m)[names(m) == "ageonset_x"] <- "ageonset"
elig <- m[!is.na(m$exit) & m$exit > 0, ]
coh  <- elig[stats::complete.cases(elig[, POOL]), ]
stopifnot(nrow(coh) == 813, sum(coh$event) == 165, nrow(elig) == 1441)

X  <- as.matrix(coh[, TOTAL6])
lp <- as.vector(sweep(X, 2, colMeans(X)) %*% BETA[TOTAL6])
coh$grp <- cut(lp, breaks = quantile(lp, c(0, 1/3, 2/3, 1)), labels = FALSE, include.lowest = TRUE)

# ============================================================== FIGURE 1 =========
# Rounded rectangle. The radius is given in inches and converted separately for x
# and y from the panel size, so the corners are circular whatever the aspect ratio.
roundrect <- function(x0, y0, x1, y1, rin = 0.09, col, border, lwd = 1.6) {
  pin <- par("pin"); rx <- rin / pin[1]; ry <- rin / pin[2]
  a <- seq(0, pi/2, length.out = 24)
  xs <- c(x1 - rx + rx*cos(a - pi/2), x1 - rx + rx*cos(a),
          x0 + rx + rx*cos(a + pi/2), x0 + rx + rx*cos(a + pi))
  ys <- c(y0 + ry + ry*sin(a - pi/2), y1 - ry + ry*sin(a),
          y1 - ry + ry*sin(a + pi/2), y0 + ry + ry*sin(a + pi))
  polygon(xs, ys, col = col, border = border, lwd = lwd)
}

draw_fig1 <- function() {
  par(mar = c(0, 0, 0, 0), xaxs = "i", yaxs = "i")
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
  MB <- list(c(0.02, 0.745, 0.60, 0.935), c(0.02, 0.435, 0.60, 0.585), c(0.02, 0.085, 0.60, 0.275))
  main_txt <- list(
    c("Patients with Parkinson's disease in PPMI with an", "MDS-UPDRS Part IV assessment at or after levodopa initiation"),
    "Eligible participants",
    c("Development sample: complete data on all", "18 candidate clinical predictors"))
  main_n <- c("n = 1,447", "n = 1,441   (302 events)", "n = 813   (165 events)")
  LH <- 0.030; GAP <- 0.048
  for (i in 1:3) {
    b <- MB[[i]]; roundrect(b[1], b[2], b[3], b[4], col = "#EDF3FA", border = NAVY)
    cx <- (b[1] + b[3]) / 2; yc <- (b[2] + b[4]) / 2
    lab <- main_txt[[i]]; k <- length(lab)
    top <- yc + ((k - 1) * LH + GAP) / 2
    for (j in seq_len(k)) text(cx, top - (j - 1) * LH, lab[j], cex = 0.95, col = INK)
    text(cx, top - (k - 1) * LH - GAP, main_n[i], cex = 1.05, font = 2, col = NAVY)
  }
  EB <- list(c(0.65, 0.600, 0.99, 0.730), c(0.65, 0.290, 0.99, 0.420))
  exc_txt <- list(
    c("Excluded (n = 6)", "no follow-up beyond time-zero (5)", "outcome already present at time-zero (1)"),
    c("Excluded (n = 628)", "incomplete on at least one of the", "18 candidate clinical predictors"))
  for (i in 1:2) {
    b <- EB[[i]]; roundrect(b[1], b[2], b[3], b[4], col = "#F5F5F5", border = GREY, lwd = 1.2)
    cx <- (b[1] + b[3]) / 2; yc <- (b[2] + b[4]) / 2
    for (j in 1:3) text(cx, yc + 0.030 - (j - 1) * 0.030, exc_txt[[i]][j], cex = 0.85, col = INK)
  }
  arrows(0.31, 0.745, 0.31, 0.593, length = 0.11, lwd = 2.2, col = NAVY)
  arrows(0.31, 0.435, 0.31, 0.293, length = 0.11, lwd = 2.2, col = NAVY)
  arrows(0.31, 0.665, 0.645, 0.665, length = 0.09, lwd = 1.4, col = GREY)
  arrows(0.31, 0.355, 0.645, 0.355, length = 0.09, lwd = 1.4, col = GREY)
}

# ============================================================== FIGURE 2 =========
NAMES <- c("Low risk", "Intermediate risk", "High risk"); COLS <- c(TEAL, GOLD, RED)
TMAX <- 10; TICKS <- c(0, 2, 4, 6, 8, 10)
fits <- lapply(1:3, function(g) survfit(Surv(exit, event) ~ 1, data = coh[coh$grp == g, ]))
atrisk <- sapply(1:3, function(g) sapply(TICKS, function(tt) sum(coh$exit[coh$grp == g] >= tt)))
lr <- survdiff(Surv(exit, event) ~ grp, data = coh)
pval <- pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)

draw_fig2 <- function() {
  layout(matrix(1:2, ncol = 1), heights = c(3.05, 1.0))
  # the left margin is wide enough for the row labels of the numbers-at-risk table, which sit
  # in it so that every count can stand directly under its time on the axis above
  par(mar = c(4.4, 8.6, 1.0, 1.4), mgp = c(3.0, 0.8, 0), las = 1)
  plot(NA, xlim = c(0, TMAX), ylim = c(0, 85), xaxt = "n", yaxt = "n",
       xlab = "", ylab = "", bty = "n")
  axis(1, at = TICKS, cex.axis = 1.05, lwd = 1.2)
  axis(2, at = seq(0, 80, 10), cex.axis = 1.05, lwd = 1.2)
  mtext("Years since levodopa initiation", side = 1, line = 2.7, cex = 1.1)
  mtext("Cumulative incidence of\nproblematic dyskinesia (%)", side = 2, line = 2.6, cex = 1.1, las = 0)
  for (g in 1:3) {
    f <- fits[[g]]; k <- f$time <= TMAX
    lines(c(0, f$time[k]), c(0, (1 - f$surv[k]) * 100), type = "s", col = COLS[g], lwd = 2.6)
  }
  legend("topleft", bty = "n", lwd = 2.6, col = COLS, seg.len = 1.6, cex = 1.05,
         legend = sprintf("%s (n=%d)", NAMES, as.vector(table(coh$grp))))
  text(TMAX, 4, if (pval < 0.001) "log-rank p < 0.001" else sprintf("log-rank p = %.3f", pval),
       adj = c(1, 0), font = 3, col = GREY, cex = 1.05)

  # numbers at risk, each directly under its time: the panel has the horizontal scale and
  # the margins of the curves above, and the group labels stand right-aligned in the left
  # margin, clear of the first count. (Until November 2026 the counts were spread over the
  # width left by the labels, so the count at year 0 stood under year 2.)
  par(mar = c(0, 8.6, 0.4, 1.4))
  plot(NA, xlim = c(0, TMAX), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
  xr <- -(strwidth("000", cex = 0.98) / 2 + strwidth("nn", cex = 0.98))
  text(xr, 0.90, "Patients at risk", adj = c(1, 0.5), font = 2, cex = 1.02, col = INK, xpd = NA)
  for (g in 1:3) {
    yy <- 0.58 - (g - 1) * 0.24
    text(xr, yy, NAMES[g], adj = c(1, 0.5), col = COLS[g], cex = 0.98, xpd = NA)
    text(TICKS, yy, atrisk[, g], adj = c(0.5, 0.5), cex = 0.98, col = INK)
  }
  widest <- max(strwidth(NAMES, cex = 0.98), strwidth("Patients at risk", cex = 1.02, font = 2))
  stopifnot(grconvertX(xr - widest, "user", "ndc") > 0)   # the labels fit inside the figure
}

# ------------------------------------------------------------------ output ------
for (fig in list(list(f = draw_fig2, nm = "Figure2_km_tertiles",      w = 9.6,  h = 8.4))) {
  png(file.path(OUT, paste0(fig$nm, ".png")), width = fig$w, height = fig$h, units = "in", res = 300)
  fig$f(); dev.off()
  svg(file.path(OUT, paste0(fig$nm, ".svg")), width = fig$w, height = fig$h); fig$f(); dev.off()
  # cairo_pdf rather than pdf: the pdf device sets every hyphen as a minus sign
  # ("Kaplan−Meier"), which the JMD figures rasterised from these files would show
  cairo_pdf(file.path(OUT, paste0(fig$nm, ".pdf")), width = fig$w, height = fig$h); fig$f(); dev.off()
  cat("saved", fig$nm, "as png, svg and pdf\n")
}
cat("\nn per tertile:", paste(as.vector(table(coh$grp)), collapse = ", "), "\n")
cat("at risk at", paste(TICKS, collapse = ", "), "years:\n")
for (g in 1:3) cat(sprintf("  %-18s %s\n", NAMES[g], paste(atrisk[, g], collapse = "  ")))
cat(sprintf("log-rank p = %.3g\n", pval))

# =============================================== SUPPLEMENTARY FIGURE S2 =========
# Decision-curve analysis at five years. It was Figure 3 until the November 2026 revision
# returned it to the supplement, where the text now cites it in one sentence. Two panels,
# because net benefit is the right quantity but is on a scale nobody has intuition for:
# the second panel restates the same difference as interventions avoided per 1,000
# patients, which is the same number a clinician can act on.
DCA <- file.path(PROJ, "03_Resultados", "Tabelas", "tab34_decision_curve.csv")
DJ  <- file.path(PROJ, "03_Resultados", "Tabelas", "tab34_decision_curve.json")
if (file.exists(DCA)) {
  if (!requireNamespace("jsonlite", quietly = TRUE))
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
  dc <- read.csv(DCA)
  dj <- jsonlite::fromJSON(DJ)
  # net reduction in interventions per 1,000: the difference in net benefit divided by the
  # odds of the threshold, which is the standard way of putting net benefit back on a
  # scale of people rather than of utilities
  dc$net_red <- 1000 * (dc$net_benefit_model - dc$net_benefit_treat_all) /
                (dc$threshold / (1 - dc$threshold))
  INC <- dj$observed_incidence
  stopifnot(abs(dc$net_red[which.min(abs(dc$threshold - 0.20))] -
                dj$at_20pct$false_positives_avoided_per_1000) < 0.5)

  draw_s2 <- function() {
    layout(matrix(1:2, ncol = 1), heights = c(1.15, 1))
    xr <- c(0.05, 0.50)

    par(mar = c(3.2, 5.4, 4.4, 1.4), mgp = c(3.0, 0.8, 0), las = 1)
    plot(NA, xlim = xr, ylim = c(-0.03, 0.215), xaxt = "n", yaxt = "n",
         xlab = "", ylab = "", bty = "n")
    # The bands in which the model beats both defaults, shaded so that the gaps between
    # them are visible. Reporting the lowest and highest winning threshold as one range
    # hides the fact that treating everyone is better between 11% and 14%. Drawn first,
    # so that the axes and the curves sit on top of the shading rather than under it.
    # The shading is named in the key rather than in a note inside the panel, where the
    # dotted incidence line and the treat-all curve ran through the words, and it is dark
    # enough to survive conversion to CMYK and print.
    SHADE <- "#D3E9E3"
    for (i in seq_len(nrow(dj$superior_bands)))
      rect(dj$superior_bands[i, 1] - 0.005, -1, dj$superior_bands[i, 2] + 0.005, 1,
           col = SHADE, border = NA)
    at <- seq(0.05, 0.50, 0.05)
    axis(1, at = at, labels = paste0(at * 100, "%"), cex.axis = 1.0, lwd = 1.2)
    axis(2, at = seq(0, 0.20, 0.05), cex.axis = 1.0, lwd = 1.2)
    mtext("Net benefit at five years", side = 2, line = 3.0, cex = 1.06, las = 0)
    mtext("A", side = 3, line = 2.9, at = 0.036, cex = 1.3, font = 2, col = INK)
    abline(h = 0, col = GREY, lwd = 1.4)
    abline(v = INC, lty = 3, col = GREY, lwd = 1.2)
    lines(dc$threshold, dc$net_benefit_treat_all, lwd = 2.2, col = GOLD, lty = 2)
    lines(dc$threshold, dc$net_benefit_model, lwd = 3.0, col = TEAL)
    # the incidence label to the left of its line. The key stands above the panel, on white:
    # inside it, over a shaded band, its shading sample looked like an empty square
    text(INC, 0.205, sprintf("observed incidence, %.0f%%", 100 * INC),
         adj = c(1.04, 0.5), cex = 0.9, font = 3, col = GREY)
    stopifnot(all(dc$net_benefit_model[dc$threshold >= INC - strwidth("observed incidence, 24%", cex = 0.9) &
                                       dc$threshold <= INC] < 0.205 - strheight("o", cex = 0.9)))
    legend(mean(xr), par("usr")[4] + 0.004, xjust = 0.5, yjust = 0, ncol = 2, xpd = NA,
           bty = "n", cex = 1.0, seg.len = 1.8, y.intersp = 1.25, x.intersp = 0.8,
           text.width = max(strwidth(c("Treat every patient", "Model better than both"), cex = 1.0)),
           lwd = c(3.0, 2.2, 1.4, NA), lty = c(1, 2, 1, NA), col = c(TEAL, GOLD, GREY, GREY),
           pch = c(NA, NA, NA, 22), pt.bg = c(NA, NA, NA, SHADE), pt.cex = 2.6, pt.lwd = 1.0,
           legend = c("The model", "Treat every patient", "Treat no patient",
                      "Model better than both"))

    par(mar = c(4.4, 5.4, 1.2, 1.4))
    plot(NA, xlim = xr, ylim = c(-110, 560), xaxt = "n", yaxt = "n",
         xlab = "", ylab = "", bty = "n")
    axis(1, at = at, labels = paste0(at * 100, "%"), cex.axis = 1.0, lwd = 1.2)
    yat <- seq(-100, 500, 100)
    axis(2, at = yat, labels = sub("^-", "−", yat), cex.axis = 1.0, lwd = 1.2)
    abline(h = 0, col = GREY, lwd = 1.4)
    mtext("Threshold probability at which a clinician would act", side = 1, line = 2.8, cex = 1.06)
    mtext("Unnecessary interventions\navoided per 1,000 patients", side = 2, line = 2.7,
          cex = 1.06, las = 0)
    mtext("B", side = 3, line = 0.2, at = 0.036, cex = 1.3, font = 2, col = INK)
    abline(v = INC, lty = 3, col = GREY, lwd = 1.2)
    lines(dc$threshold, dc$net_red, lwd = 3.0, col = NAVY)
    k <- which.min(abs(dc$threshold - 0.20))
    # the label sits above and to the left of its point, where the curve is lower, with a
    # short leader, so that neither the curve nor the dotted incidence line crosses it
    xk <- dc$threshold[k]; yk <- dc$net_red[k]
    segments(xk - 0.002, yk + 16, xk - 0.010, yk + 58, col = GREY, lwd = 1)
    points(xk, yk, pch = 21, bg = RED, col = RED, cex = 1.5)
    text(xk - 0.011, yk + 74, sprintf("%.0f at a threshold of 20%%", yk),
         adj = c(1, 0.5), cex = 0.95, col = INK)
    stopifnot(all(dc$net_red[dc$threshold <= xk] < yk + 74 - strheight("0", cex = 0.95)))
  }

  for (nm in c("png", "svg", "pdf")) {
    f <- file.path(OUT, paste0("FigureS2_decision_curve.", nm))
    if (nm == "png") png(f, width = 8.6, height = 8.4, units = "in", res = 300)
    if (nm == "svg") svg(f, width = 8.6, height = 8.4)
    if (nm == "pdf") cairo_pdf(f, width = 8.6, height = 8.4)
    draw_s2(); dev.off()
  }
  cat("saved FigureS2_decision_curve as png, svg and pdf\n")
  cat("  model superior to both defaults at ",
      paste(sprintf("%.0f%% to %.0f%%", 100 * dj$superior_bands[, 1],
                    100 * dj$superior_bands[, 2]), collapse = ", "), "\n", sep = "")
  cat(sprintf("  at 20%%: net benefit %.3f against %.3f, %.0f avoided per 1,000\n",
              dj$at_20pct$model, dj$at_20pct$treat_all,
              dj$at_20pct$false_positives_avoided_per_1000))
}
