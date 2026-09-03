# ============================================================
# 05_survpfi_only.R
# OOM ve zaman aşımını önlemek amacıyla:
#   - Tüm 32 model için sırayla YALNIZCA SurvPFI hesaplar.
#   - Sonuçları CSV olarak results/ altına kaydeder.
# ============================================================

SEED <- 42
set.seed(SEED)

BASE_DIR    <- "/home/mustafacavus"
MODEL_DIR   <- file.path(BASE_DIR, "models")
RESULT_DIR  <- file.path(BASE_DIR, "results")
SPLIT_DIR   <- file.path(BASE_DIR, "datasets", "splits")
LOG_FILE    <- file.path(BASE_DIR, "pfi_only_log.txt")

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

log_msg("=== YALNIZCA SURVPFI HESAPLAMALARI BAŞLIYOR ===")

# =================================================================
# VERİ HAZIRLAMA (1.000 Satır Subsample)
# =================================================================
mk_task <- function(dt, id) {
  TaskSurv$new(id = id, backend = as.data.frame(dt), time = "time", event = "status")
}

tasks <- list()
test_data_subsampled <- list()

for (ds in 1:2) {
  pre <- if (ds == 1) "h" else "r"
  test_dt <- readRDS(file.path(SPLIT_DIR, paste0("test_ds", ds, ".rds")))
  
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
# SURVPFI HESAPLAMA DÖNGÜSÜ
# =================================================================
model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Toplam model sayısı:", length(model_files)))

for (mf in model_files) {
  ds <- as.integer(gsub(".*_ds(\\d+)\\.rds$", "\\1", mf))
  if (is.na(ds) || !(ds %in% 1:2)) next

  ml <- gsub("\\.rds$", "", mf)
  output_csv <- file.path(RESULT_DIR, paste0("pfi_values_", ml, ".csv"))

  # Eğer PFI çıktısı zaten varsa atla
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

    pfi <- model_parts(expl, loss_function = survex::loss_brier_score)

    if (!is.null(pfi)) {
      write.csv(as.data.frame(pfi$result), output_csv, row.names = FALSE)
      log_msg(paste("    ✓ PFI Kaydedildi:", ml))
    }

    rm(obj, lrn_obj, pred_fn, pred_surv_fn, expl, pfi)
  }, error = function(e) {
    log_msg(paste("  [PFI HATASI]:", ml, "-", e$message))
  })

  gc()
}

log_msg("=== SURVPFI TÜM MODELLER İÇİN TAMAMLANDI ===")
