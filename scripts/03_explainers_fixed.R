# ============================================================
# 03_explainers_fixed.R
# Mevcut eğitilmiş modeller üzerinde survex explainer oluşturma
# FIX: mlr3 learner nesnelerini survex'e uyumlu hale getirme
# ============================================================

SEED <- 42
set.seed(SEED)

BASE_DIR   <- "/home/mustafacavus"
MODEL_DIR  <- file.path(BASE_DIR, "models")
EXPL_DIR   <- file.path(BASE_DIR, "explainers")
SPLIT_DIR  <- file.path(BASE_DIR, "datasets", "splits")
LOG_FILE   <- file.path(BASE_DIR, "explainer_log.txt")

dir.create(EXPL_DIR, recursive = TRUE, showWarnings = FALSE)

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
  library(data.table)
  library(survival)
})

lgr::get_logger("mlr3")$set_threshold("warn")

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("=== SURVEX EXPLAINER (DÜZELTME) BAŞLIYOR ===")

# =================================================================
# DATA PREPARATION (from saved splits)
# =================================================================
log_msg("--- VERİ HAZIRLAMA ---")

mk_task <- function(dt, id) {
  TaskSurv$new(id = id, backend = as.data.frame(dt), time = "time", event = "status")
}

tasks <- list()
test_data <- list()

for (ds in 1:2) {
  pre <- if (ds == 1) "h" else "r"

  train_dt <- readRDS(file.path(SPLIT_DIR, paste0("train_ds", ds, ".rds")))
  test_dt  <- readRDS(file.path(SPLIT_DIR, paste0("test_ds",  ds, ".rds")))

  tasks[[paste0("train", ds)]] <- mk_task(train_dt, paste0(pre, "_train"))
  tasks[[paste0("test",  ds)]] <- mk_task(test_dt,  paste0(pre, "_test"))

  # Explainer için test verisini hazırla
  test_data[[ds]] <- list(
    x = as.data.frame(test_dt[, !c("time", "status"), with = FALSE]),
    y = Surv(test_dt$time, test_dt$status)
  )

  rm(train_dt, test_dt)
  gc()
}

log_msg(paste("DS1 test:", nrow(test_data[[1]]$x), "| DS2 test:", nrow(test_data[[2]]$x)))

# =================================================================
# CUSTOM PREDICT FUNCTIONS FOR SURVEX
# =================================================================

# mlr3 learner -> risk score tahmin fonksiyonu
make_predict_function <- function(lrn_obj, task_template) {
  function(model, newdata, times) {
    # mlr3 task oluştur
    newdata_with_dummy <- as.data.frame(newdata)
    newdata_with_dummy$time <- 1
    newdata_with_dummy$status <- 0

    new_task <- tryCatch({
      TaskSurv$new(
        id = "pred_task",
        backend = newdata_with_dummy,
        time = "time",
        event = "status"
      )
    }, error = function(e) NULL)

    if (is.null(new_task)) return(rep(NA, nrow(newdata)))

    pred <- tryCatch(
      lrn_obj$predict(new_task),
      error = function(e) NULL
    )

    if (is.null(pred)) return(rep(NA, nrow(newdata)))

    # Risk skoru döndür
    tryCatch(as.numeric(pred$lp), error = function(e) {
      tryCatch(as.numeric(pred$crank), error = function(e2) {
        rep(NA, nrow(newdata))
      })
    })
  }
}

# mlr3 learner -> survival function tahmin fonksiyonu
make_predict_survival_function <- function(lrn_obj) {
  function(model, newdata, times) {
    # mlr3 task oluştur
    newdata_with_dummy <- as.data.frame(newdata)
    newdata_with_dummy$time <- 1
    newdata_with_dummy$status <- 0

    new_task <- tryCatch({
      TaskSurv$new(
        id = "pred_task_surv",
        backend = newdata_with_dummy,
        time = "time",
        event = "status"
      )
    }, error = function(e) NULL)

    if (is.null(new_task)) {
      return(matrix(NA, nrow = nrow(newdata), ncol = length(times)))
    }

    pred <- tryCatch(
      lrn_obj$predict(new_task),
      error = function(e) NULL
    )

    if (is.null(pred) || is.null(pred$distr)) {
      return(matrix(NA, nrow = nrow(newdata), ncol = length(times)))
    }

    # distr nesnesinden survival olasılıklarını çıkar
    surv_matrix <- tryCatch({
      # mlr3proba distr -> survival function
      s <- pred$distr$survival(times)
      if (is.matrix(s)) {
        if (nrow(s) == length(times)) t(s) else s
      } else {
        matrix(s, nrow = nrow(newdata), ncol = length(times))
      }
    }, error = function(e) {
      matrix(NA, nrow = nrow(newdata), ncol = length(times))
    })

    surv_matrix
  }
}

# =================================================================
# EXPLAINER OLUŞTURMA
# =================================================================
log_msg("--- SURVEX EXPLAINERS ---")

model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Toplam model dosyası:", length(model_files)))

success_count <- 0
fail_count <- 0

for (mf in model_files) {
  ds <- as.integer(gsub(".*_ds(\\d+)\\.rds$", "\\1", mf))
  if (is.na(ds) || !(ds %in% 1:2)) next

  ml <- gsub("\\.rds$", "", mf)
  expl_file <- file.path(EXPL_DIR, paste0("explainer_", ml, ".rds"))

  if (file.exists(expl_file)) {
    log_msg(paste("  Atlanıyor (mevcut):", ml))
    next
  }

  log_msg(paste("  Explainer oluşturuluyor:", ml))

  tryCatch({
    obj <- readRDS(file.path(MODEL_DIR, mf))

    # Learner nesnesini çıkar
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    x_test <- test_data[[ds]]$x
    y_test <- test_data[[ds]]$y

    # Custom predict fonksiyonlarını oluştur
    pred_fn <- make_predict_function(lrn_obj, tasks[[paste0("test", ds)]])
    pred_surv_fn <- make_predict_survival_function(lrn_obj)

    # survex explainer oluştur - custom predict fonksiyonlarıyla
    expl <- explain_survival(
      model                    = lrn_obj,
      data                     = x_test,
      y                        = y_test,
      predict_function         = pred_fn,
      predict_survival_function = pred_surv_fn,
      label                    = ml,
      verbose                  = FALSE
    )

    saveRDS(expl, expl_file)
    log_msg(paste("  ✓ Explainer kaydedildi:", ml))
    success_count <- success_count + 1

    rm(obj, lrn_obj, expl)

  }, error = function(e) {
    log_msg(paste("  ✗ Explainer hatası:", ml, "-", e$message))
    fail_count <<- fail_count + 1
  })

  gc()
}

log_msg(paste("=== EXPLAINER TAMAMLANDI === Başarılı:", success_count, "| Başarısız:", fail_count))
log_msg(paste("Bitiş zamanı:", Sys.time()))
