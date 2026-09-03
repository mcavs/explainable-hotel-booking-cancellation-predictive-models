# ============================================================
# 04_survpfi_survshap.R
# Explainer nesneleri üzerinden:
#   - SurvPFI (Permutation Feature Importance) hesaplama
#   - SurvSHAP (Global SHAP) hesaplama
#   - Değişken bazlı önem derecesi (integral alan) çıkarımı ve CSV kaydı
# FIX: ggplot2 S7 plot hatasını engellemek için grafik çizimi kaldırıldı,
#      sadece CSV olarak önemlilik değerleri kaydedilecek.
# ============================================================

SEED <- 42
set.seed(SEED)

BASE_DIR    <- "/home/mustafacavus"
EXPL_DIR    <- file.path(BASE_DIR, "explainers")
RESULT_DIR  <- file.path(BASE_DIR, "results")
LOG_FILE    <- file.path(BASE_DIR, "pfi_shap_log.txt")

# =================================================================
# LIBRARIES
# =================================================================
suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3proba)
  library(mlr3extralearners)
  library(mlr3pipelines)
  library(mlr3learners)
  library(survex)
  library(survival)
  library(data.table)
})

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("=== SURVPFI & SURVSHAP TÜM MODELLER İÇİN (SADECE CSV) BAŞLIYOR ===")

explainer_files <- list.files(EXPL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Toplam explainer dosyası:", length(explainer_files)))

for (ef in explainer_files) {
  ml <- gsub("^explainer_", "", gsub("\\.rds$", "", ef))
  log_msg(paste("İşleniyor:", ml))

  expl_path <- file.path(EXPL_DIR, ef)
  expl <- readRDS(expl_path)

  # predict fonksiyonlarının closure environment'ına lrn_obj değişkenini bağla
  env_pred <- environment(expl$predict_function)
  env_surv <- environment(expl$predict_survival_function)
  assign("lrn_obj", expl$model, envir = env_pred)
  assign("lrn_obj", expl$model, envir = env_surv)

  # 1. SurvPFI (Permutation Feature Importance) Hesaplama
  log_msg(paste("  [PFI] hesaplanıyor..."))
  pfi <- tryCatch({
    model_parts(expl, loss_function = survex::loss_brier_score)
  }, error = function(e) {
    log_msg(paste("    [PFI HATASI]:", e$message))
    NULL
  })

  if (!is.null(pfi)) {
    pfi_df <- as.data.frame(pfi$result)
    write.csv(pfi_df, file.path(RESULT_DIR, paste0("pfi_values_", ml, ".csv")), row.names = FALSE)
    log_msg(paste("    [PFI Tamamlandı]"))
  }

  # 2. SurvSHAP (Global SHAP) Hesaplama
  log_msg(paste("  [SurvSHAP] hesaplanıyor..."))
  shap <- tryCatch({
    set.seed(SEED)
    test_x <- expl$data
    test_y <- expl$y
    sample_size <- min(30, nrow(test_x)) # KernelSHAP yükünü optimize etmek için 30 örnek seçildi
    idx <- sample(seq_len(nrow(test_x)), sample_size)
    
    new_obs <- test_x[idx, , drop = FALSE]
    y_true  <- test_y[idx]

    model_survshap(
      explainer          = expl,
      new_observation    = new_obs,
      y_true             = y_true,
      calculation_method = "kernelshap",
      aggregation_method = "integral"
    )
  }, error = function(e) {
    log_msg(paste("    [SHAP HATASI]:", e$message))
    NULL
  })

  if (!is.null(shap)) {
    shap_df <- as.data.frame(shap$result)
    write.csv(shap_df, file.path(RESULT_DIR, paste0("shap_values_", ml, ".csv")), row.names = FALSE)
    log_msg(paste("    [SHAP Tamamlandı]"))
  }

  rm(expl, pfi, shap, env_pred, env_surv)
  gc()
}

log_msg("=== TÜM HESAPLAMALAR TAMAMLANDI ===")
