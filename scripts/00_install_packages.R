# ============================================================
# 00_install_packages.R
# mlr3proba Deney — Paket Kurulumu
# Önce bu scripti çalıştırın, sonra ana deneye geçin.
# ============================================================

cat("=== PAKET KURULUMU BAŞLIYOR ===\n")

# CRAN paketleri
cran_pkgs <- c(
  # mlr3 ekosistemi
  "mlr3", "mlr3proba", "mlr3tuning", "mlr3mbo", "mlr3pipelines",
  "mlr3learners", "mlr3misc", "paradox", "bbotk",
  # Model paketleri
  "survival", "ranger", "glmnet", "xgboost", "rpart",
  "partykit", "mboost", "flexsurv", "survivalsvm",
  "penalized", "randomForestSRC", "aorsf",
  # XAI
  "survex",
  # Veri
  "data.table", "dplyr", "lubridate"
)

for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Kuruluyor:", pkg, "\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  } else {
    cat(paste("Mevcut:", pkg, "\n"))
  }
}

# mlr3extralearners (CRAN'da yok, r-universe'den kur)
if (!requireNamespace("mlr3extralearners", quietly = TRUE)) {
  cat("Kuruluyor: mlr3extralearners (r-universe)\n")
  install.packages("mlr3extralearners",
                   repos = "https://mlr-org.r-universe.dev",
                   quiet = TRUE)
} else {
  cat("Mevcut: mlr3extralearners\n")
}

# CoxBoost (CRAN'dan kaldırılmış olabilir, r-universe dene)
if (!requireNamespace("CoxBoost", quietly = TRUE)) {
  cat("Kuruluyor: CoxBoost (r-universe)\n")
  tryCatch({
    install.packages("CoxBoost",
                     repos = c("https://binderh.r-universe.dev",
                               "https://cloud.r-project.org"),
                     quiet = TRUE)
  }, error = function(e) {
    cat(paste("CoxBoost kurulamadı:", e$message, "\n"))
  })
} else {
  cat("Mevcut: CoxBoost\n")
}

# Doğrulama
cat("\n=== KURULUM DOĞRULAMA ===\n")
all_pkgs <- c(cran_pkgs, "mlr3extralearners", "CoxBoost")
for (pkg in all_pkgs) {
  ok <- requireNamespace(pkg, quietly = TRUE)
  cat(paste(ifelse(ok, "✓", "✗"), pkg, "\n"))
}

cat("\n=== PAKET KURULUMU TAMAMLANDI ===\n")
