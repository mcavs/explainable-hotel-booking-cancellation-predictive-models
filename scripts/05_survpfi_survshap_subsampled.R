# ============================================================
# 05_survpfi_survshap_subsampled.R
# OOM Hatasını Önlemek İçin 1.000 Satırlık Test Kümesi ile:
#   - Explainer nesnelerini çalışma anında yeniden oluşturma
#   - SurvPFI (Brier kaybı tabanlı) hesaplama
#   - SurvSHAP (KernelSHAP tabanlı global integral) hesaplama
#   - Sonuçları CSV olarak results/ altına kaydetme
# ============================================================

SEED <- 42
set.seed(SEED)

BASE_DIR    <- "/home/mustafacavus"
MODEL_DIR   <- file.path(BASE_DIR, "models")
RESULT_DIR  <- file.path(BASE_DIR, "results")
SPLIT_DIR   <- file.path(BASE_DIR, "datasets", "splits")
LOG_FILE    <- file.path(BASE_DIR, "pfi_shap_subsampled_log.txt")

dir.create(RESULT_DIR, recursive = TRUE, showWarnings = FALSE)

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

log_msg("=== SUBSAMPLED PFI & SHAP TÜM MODELLER İÇİN BAŞLIYOR ===")

# =================================================================
# VERİ HAZIRLAMA (1.000 Satırlık Subsample Test Seti)
# =================================================================
log_msg("--- VERİ HAZIRLAMA (1.000 Satır Subsample) ---")

mk_task <- function(dt, id) {
  TaskSurv$new(id = id, backend = as.data.frame(dt), time = "time", event = "status")
}

tasks <- list()
test_data_subsampled <- list()

for (ds in 1:2) {
  pre <- if (ds == 1) "h" else "r"
  
  # Orijinal test setini yükle
  test_dt <- readRDS(file.path(SPLIT_DIR, paste0("test_ds", ds, ".rds")))
  
  # 1.000 satıra subsample et (Veri boyutu yetersizse maksimumunu al)
  set.seed(SEED)
  sample_n <- min(1000, nrow(test_dt))
  idx <- sample(seq_len(nrow(test_dt)), sample_n)
  test_dt_sub <- test_dt[idx, ]
  
  tasks[[paste0("test", ds)]] <- mk_task(test_dt_sub, paste0(pre, "_test_sub"))
  
  test_data_subsampled[[ds]] <- list(
    x = as.data.frame(test_dt_sub[, !c("time", "status"), with = FALSE]),
    y = Surv(test_dt_sub$time, test_dt_sub$status)
  )
  
  rm(test_dt, test_dt_sub)
  gc()
}

log_msg(paste("DS1 Subsampled Test:", nrow(test_data_subsampled[[1]]$x), "satır"))
log_msg(paste("DS2 Subsampled Test:", nrow(test_data_subsampled[[2]]$x), "satır"))

# =================================================================
# CUSTOM PREDICT FUNCTIONS FOR SURVEX
# =================================================================
make_predict_function <- function(lrn_obj) {
  function(model, newdata, times) {
    newdata_with_dummy <- as.data.frame(newdata)
    newdata_with_dummy$time <- 1
    newdata_with_dummy$status <- 0

    new_task <- tryCatch({
      TaskSurv$new(id = "pred_task", backend = newdata_with_dummy, time = "time", event = "status")
    }, error = function(e) NULL)

    if (is.null(new_task)) return(rep(NA, nrow(newdata)))

    pred <- tryCatch(lrn_obj$predict(new_task), error = function(e) NULL)
    if (is.null(pred)) return(rep(NA, nrow(newdata)))

    tryCatch(as.numeric(pred$lp), error = function(e) {
      tryCatch(as.numeric(pred$crank), error = function(e2) rep(NA, nrow(newdata)))
    })
  }
}

make_predict_survival_function <- function(lrn_obj) {
  function(model, newdata, times) {
    newdata_with_dummy <- as.data.frame(newdata)
    newdata_with_dummy$time <- 1
    newdata_with_dummy$status <- 0

    new_task <- tryCatch({
      TaskSurv$new(id = "pred_task_surv", backend = newdata_with_dummy, time = "time", event = "status")
    }, error = function(e) NULL)

    if (is.null(new_task)) return(matrix(NA, nrow = nrow(newdata), ncol = length(times)))

    pred <- tryCatch(lrn_obj$predict(new_task), error = function(e) NULL)
    if (is.null(pred) || is.null(pred$distr)) {
      return(matrix(NA, nrow = nrow(newdata), ncol = length(times)))
    }

    surv_matrix <- tryCatch({
      s <- pred$distr$survival(times)
      if (is.matrix(s)) {
        if (nrow(s) == length(times)) t(s) else s
      } else {
        matrix(s, nrow = nrow(newdata), ncol = length(times))
      }
    }, error = function(e) matrix(NA, nrow = nrow(newdata), ncol = length(times)))
    
    surv_matrix
  }
}

# =================================================================
# MODELLERİ YÜKLE VE AÇIKLAYICILARI YENİDEN OLUŞTURARAK ÇALIŞTIR
# =================================================================
model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Toplam model sayısı:", length(model_files)))

for (mf in model_files) {
  ds <- as.integer(gsub(".*_ds(\\d+)\\.rds$", "\\1", mf))
  if (is.na(ds) || !(ds %in% 1:2)) next

  ml <- gsub("\\.rds$", "", mf)
  log_msg(paste("İşleniyor:", ml))

  tryCatch({
    # Modeli oku
    obj <- readRDS(file.path(MODEL_DIR, mf))
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    # 1.000 satırlık verilerle explainer'ı çalışma anında kur
    x_test <- test_data_subsampled[[ds]]$x
    y_test <- test_data_subsampled[[ds]]$y

    pred_fn <- make_predict_function(lrn_obj)
    pred_surv_fn <- make_predict_survival_function(lrn_obj)

    # R6 closure environment ataması
    assign("lrn_obj", lrn_obj, envir = environment(pred_fn))
    assign("lrn_obj", lrn_obj, envir = environment(pred_surv_fn))

    expl <- explain_survival(
      model                     = lrn_obj,
      data                      = x_test,
      y                         = y_test,
      predict_function          = pred_fn,
      predict_survival_function = pred_surv_fn,
      label                     = ml,
      verbose                   = FALSE
    )

    # 1. SurvPFI (Permutation Feature Importance)
    log_msg(paste("  [PFI] hesaplanıyor..."))
    pfi <- tryCatch({
      model_parts(expl, loss_function = survex::loss_brier_score)
    }, error = function(e) {
      log_msg(paste("    [PFI HATASI]:", e$message))
      NULL
    })

    if (!is.null(pfi)) {
      write.csv(as.data.frame(pfi$result), file.path(RESULT_DIR, paste0("pfi_values_", ml, ".csv")), row.names = FALSE)
      log_msg(paste("    [PFI Tamamlandı]"))
    }

    # 2. SurvSHAP (Global SHAP)
    log_msg(paste("  [SurvSHAP] hesaplanıyor..."))
    shap <- tryCatch({
      set.seed(SEED)
      sample_size <- min(30, nrow(x_test))
      idx <- sample(seq_len(nrow(x_test)), sample_size)
      
      new_obs <- x_test[idx, , drop = FALSE]
      y_true  <- y_test[idx]

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
      write.csv(as.data.frame(shap$result), file.path(RESULT_DIR, paste0("shap_values_", ml, ".csv")), row.names = FALSE)
      log_msg(paste("    [SHAP Tamamlandı]"))
    }

    rm(obj, lrn_obj, pred_fn, pred_surv_fn, expl, pfi, shap)
  }, error = function(e) {
    log_msg(paste("  [MODEL HATASI]:", ml, "-", e$message))
  })

  gc()
}

log_msg("=== SUBSAMPLED TÜM MODEL HESAPLAMALARI TAMAMLANDI ===")
