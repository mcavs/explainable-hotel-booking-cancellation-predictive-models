# ============================================================
# 05_survshap_only.R
# OOM ve zaman aşımını önlemek amacıyla:
#   - Tüm 32 model için sırayla YALNIZCA SurvSHAP hesaplar.
#   - 1.000 satırlık test subsample içinden 30 satır KernelSHAP örneklemi alır.
#   - Sonuçları CSV olarak results/ altına kaydeder.
# ============================================================

SEED <- 42
set.seed(SEED)

BASE_DIR    <- "/home/mustafacavus"
MODEL_DIR   <- file.path(BASE_DIR, "models")
RESULT_DIR  <- file.path(BASE_DIR, "results")
SPLIT_DIR   <- file.path(BASE_DIR, "datasets", "splits")
LOG_FILE    <- file.path(BASE_DIR, "shap_only_log.txt")

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

log_msg("=== YALNIZCA SURVSHAP HESAPLAMALARI BAŞLIYOR ===")

# =================================================================
# VERİ HAZIRLAMA (1.000 Satır Subsample)
# =================================================================
test_data_subsampled <- list()

for (ds in 1:2) {
  test_dt <- readRDS(file.path(SPLIT_DIR, paste0("test_ds", ds, ".rds")))
  
  set.seed(SEED)
  sample_n <- min(1000, nrow(test_dt))
  idx <- sample(seq_len(nrow(test_dt)), sample_n)
  test_dt_sub <- test_dt[idx, ]
  
  test_data_subsampled[[ds]] <- list(
    x = as.data.frame(test_dt_sub[, !c("time", "status"), with = FALSE]),
    y = Surv(test_dt_sub$time, test_dt_sub$status)
  )
  rm(test_dt, test_dt_sub)
  gc()
}

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
# SURVSHAP HESAPLAMA DÖNGÜSÜ
# =================================================================
model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Toplam model sayısı:", length(model_files)))

for (mf in model_files) {
  ds <- as.integer(gsub(".*_ds(\\d+)\\.rds$", "\\1", mf))
  if (is.na(ds) || !(ds %in% 1:2)) next

  ml <- gsub("\\.rds$", "", mf)
  output_csv <- file.path(RESULT_DIR, paste0("shap_values_", ml, ".csv"))

  # Eğer SHAP çıktısı zaten varsa atla
  if (file.exists(output_csv)) {
    log_msg(paste("  Atlanıyor (mevcut):", ml))
    next
  }

  log_msg(paste("İşleniyor:", ml))

  tryCatch({
    obj <- readRDS(file.path(MODEL_DIR, mf))
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    x_test <- test_data_subsampled[[ds]]$x
    y_test <- test_data_subsampled[[ds]]$y

    pred_fn <- make_predict_function(lrn_obj)
    pred_surv_fn <- make_predict_survival_function(lrn_obj)

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

    # OOM engellemek amacıyla 30 gözlem üzerinde KernelSHAP çalıştırıyoruz
    set.seed(SEED)
    shap_idx <- sample(seq_len(nrow(x_test)), 30)
    x_test_shap <- x_test[shap_idx, , drop = FALSE]

    shap <- model_survshap(expl, new_observation = x_test_shap, calculation_method = "kernelshap")

    if (!is.null(shap)) {
      # SurvSHAP integral alan/değişken önemlilik çıktılarını kaydet
      write.csv(as.data.frame(shap$result), output_csv, row.names = FALSE)
      log_msg(paste("    ✓ SurvSHAP Kaydedildi:", ml))
    }

    rm(obj, lrn_obj, pred_fn, pred_surv_fn, expl, shap)
  }, error = function(e) {
    log_msg(paste("  [SHAP HATASI]:", ml, "-", e$message))
  })

  gc()
}

log_msg("=== SURVSHAP TÜM MODELLER İÇİN TAMAMLANDI ===")
