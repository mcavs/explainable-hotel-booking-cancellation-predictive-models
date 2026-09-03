# ============================================================
# 02_eval_explain.R
# Mevcut eğitilmiş modeller üzerinde:
#   - Performans değerlendirmesi (C-Index, IBS, RCLL, UNO AUC)
#   - Survex explainer nesneleri oluşturma
# ============================================================

# =================================================================
# SECTION 0: CONFIGURATION
# =================================================================
SEED <- 42
set.seed(SEED)

BASE_DIR   <- "/home/mustafacavus"
MODEL_DIR  <- file.path(BASE_DIR, "models")
EXPL_DIR   <- file.path(BASE_DIR, "explainers")
RESULT_DIR <- file.path(BASE_DIR, "results")
SPLIT_DIR  <- file.path(BASE_DIR, "datasets", "splits")
DATA_DIR   <- file.path(BASE_DIR, "datasets")
LOG_FILE   <- file.path(BASE_DIR, "eval_log.txt")

for (d in c(EXPL_DIR, RESULT_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# =================================================================
# SECTION 1: LIBRARIES
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
  library(dplyr)
  library(lubridate)
})

lgr::get_logger("mlr3")$set_threshold("warn")

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("=== EVAL & EXPLAINER BAŞLIYOR ===")

# =================================================================
# SECTION 2: DATA PREPARATION (from saved splits)
# =================================================================
log_msg("--- VERİ HAZIRLAMA ---")

mk_task <- function(dt, id) {
  TaskSurv$new(id = id, backend = as.data.frame(dt), time = "time", event = "status")
}

tasks <- list()

# Önce kaydedilmiş split'lerden okumayı dene
split_exists <- file.exists(file.path(SPLIT_DIR, "train_ds1.rds"))

if (split_exists) {
  log_msg("Kaydedilmiş split dosyaları kullanılıyor...")
  for (ds in 1:2) {
    pre <- if (ds == 1) "h" else "r"
    tasks[[paste0("train", ds)]] <- mk_task(readRDS(file.path(SPLIT_DIR, paste0("train_ds", ds, ".rds"))), paste0(pre, "_train"))
    tasks[[paste0("test",  ds)]] <- mk_task(readRDS(file.path(SPLIT_DIR, paste0("test_ds",  ds, ".rds"))), paste0(pre, "_test"))
    gc()
  }
} else {
  log_msg("Split dosyaları bulunamadı, veriyi yeniden bölüyoruz...")

  # Dataset 1
  h <- fread(file.path(DATA_DIR, "hotels.csv"))
  month_map <- c(January=1, February=2, March=3, April=4, May=5, June=6,
                 July=7, August=8, September=9, October=10, November=11, December=12)
  h[, arrival_month_num := month_map[arrival_date_month]]
  h[, arrival_date := as.Date(paste(arrival_date_year, arrival_month_num, arrival_date_day_of_month, sep = "-"))]
  h[, booking_date := arrival_date - lead_time]
  h[, status_date  := as.Date(reservation_status_date)]
  h[, status := as.integer(is_canceled)]
  h[, time   := fifelse(status == 1L,
                         as.numeric(difftime(status_date, booking_date, units = "days")),
                         as.numeric(lead_time))]
  h[time <= 0, time := 1]
  h[is.na(children), children := 0]
  drop1 <- c("is_canceled", "reservation_status", "reservation_status_date",
             "arrival_date", "booking_date", "status_date",
             "country", "agent", "company", "arrival_date_month")
  h[, (drop1) := NULL]
  chr1 <- names(h)[vapply(h, is.character, logical(1))]
  h[, (chr1) := lapply(.SD, as.factor), .SDcols = chr1]
  ds1 <- h[!is.na(time) & !is.na(status)]
  rm(h); gc()

  # Dataset 2
  r <- fread(file.path(DATA_DIR, "hotel_reservations.csv"))
  r[, status := fifelse(booking_status == "Canceled", 1L, 0L)]
  r[, time   := fifelse(status == 1L, lead_time / 2, as.numeric(lead_time))]
  r[time <= 0, time := 1]
  drop2 <- c("Booking_ID", "booking_status")
  r[, (drop2) := NULL]
  chr2 <- names(r)[vapply(r, is.character, logical(1))]
  r[, (chr2) := lapply(.SD, as.factor), .SDcols = chr2]
  ds2 <- r[!is.na(time) & !is.na(status)]
  rm(r); gc()

  split_data <- function(dt, seed = 42) {
    set.seed(seed)
    n <- nrow(dt); idx <- sample(n)
    t1 <- floor(0.6 * n); t2 <- floor(0.8 * n)
    list(train = dt[idx[1:t1]], val = dt[idx[(t1+1):t2]], test = dt[idx[(t2+1):n]])
  }

  sp1 <- split_data(ds1); sp2 <- split_data(ds2)
  rm(ds1, ds2); gc()

  # Subsample DS1 train
  set.seed(SEED)
  if (nrow(sp1$train) > 21765) {
    keep_idx <- sample(seq_len(nrow(sp1$train)), 21765)
    sp1$train <- sp1$train[keep_idx, ]
  }

  for (ds in 1:2) {
    sp  <- if (ds == 1) sp1 else sp2
    pre <- if (ds == 1) "h" else "r"
    tasks[[paste0("train", ds)]] <- mk_task(sp$train, paste0(pre, "_train"))
    tasks[[paste0("test",  ds)]] <- mk_task(sp$test,  paste0(pre, "_test"))
  }
  rm(sp1, sp2); gc()
}

log_msg(paste("DS1 train:", tasks[["train1"]]$nrow, "| test:", tasks[["test1"]]$nrow))
log_msg(paste("DS2 train:", tasks[["train2"]]$nrow, "| test:", tasks[["test2"]]$nrow))

# =================================================================
# SECTION 7: PERFORMANCE EVALUATION
# =================================================================
log_msg("--- SECTION 7: PERFORMANCE EVALUATION ---")

base_measures <- msrs(c("surv.cindex", "surv.graf", "surv.rcll"))

eval_model <- function(model_file, task_eval, label) {
  tryCatch({
    obj <- readRDS(file.path(MODEL_DIR, model_file))
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    pred <- lrn_obj$predict(task_eval)

    scores <- tryCatch(pred$score(base_measures), error = function(e) {
      c(surv.cindex = NA_real_, surv.graf = NA_real_, surv.rcll = NA_real_)
    })

    iauc <- tryCatch(pred$score(msr("surv.uno_auc")),
                     error = function(e) c(surv.uno_auc = NA_real_))

    c(scores, iauc)
  }, error = function(e) {
    log_msg(paste("  Eval hatası:", label, "-", e$message))
    c(surv.cindex = NA, surv.graf = NA, surv.rcll = NA, surv.uno_auc = NA)
  })
}

model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Değerlendirilecek model sayısı:", length(model_files)))
perf_rows <- list()

for (mf in model_files) {
  ds <- as.integer(gsub(".*_ds(\\d+)\\.rds$", "\\1", mf))
  if (is.na(ds) || !(ds %in% 1:2)) next

  ml <- gsub("\\.rds$", "", mf)
  log_msg(paste("Eval başlıyor:", ml))

  train_sc <- eval_model(mf, tasks[[paste0("train", ds)]], paste0(ml, "_train"))
  test_sc  <- eval_model(mf, tasks[[paste0("test", ds)]],  paste0(ml, "_test"))

  perf_rows[[ml]] <- data.frame(
    model         = ml,
    dataset       = ds,
    train_cindex  = train_sc["surv.cindex"],
    train_ibs     = train_sc["surv.graf"],
    train_rcll    = train_sc["surv.rcll"],
    train_iauc    = train_sc["surv.uno_auc"],
    test_cindex   = test_sc["surv.cindex"],
    test_ibs      = test_sc["surv.graf"],
    test_rcll     = test_sc["surv.rcll"],
    test_iauc     = test_sc["surv.uno_auc"],
    stringsAsFactors = FALSE
  )
  log_msg(paste("  ✓ Eval:", ml, "| Test C:", round(test_sc["surv.cindex"], 4)))

  # Model nesnesini bellekten temizle
  rm(train_sc, test_sc)
  gc()
}

perf_df <- do.call(rbind, perf_rows)
rownames(perf_df) <- NULL

write.csv(perf_df[perf_df$dataset == 1, ],
          file.path(RESULT_DIR, "performance_ds1.csv"), row.names = FALSE)
write.csv(perf_df[perf_df$dataset == 2, ],
          file.path(RESULT_DIR, "performance_ds2.csv"), row.names = FALSE)
write.csv(perf_df, file.path(RESULT_DIR, "performance_all.csv"), row.names = FALSE)
log_msg("Performans sonuçları kaydedildi.")

# =================================================================
# SECTION 8: SURVEX EXPLAINER OBJECTS
# =================================================================
log_msg("--- SECTION 8: SURVEX EXPLAINERS ---")

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
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    task_tst <- tasks[[paste0("test", ds)]]
    test_dt  <- as.data.table(task_tst$data())
    x_test   <- test_dt[, !c("time", "status"), with = FALSE]
    y_test   <- Surv(test_dt$time, test_dt$status)

    expl <- explain(
      model   = lrn_obj,
      data    = as.data.frame(x_test),
      y       = y_test,
      label   = ml,
      verbose = FALSE
    )

    saveRDS(expl, expl_file)
    log_msg(paste("  ✓ Explainer kaydedildi:", ml))

    rm(obj, lrn_obj, test_dt, x_test, y_test, expl)
  }, error = function(e) {
    log_msg(paste("  ✗ Explainer hatası:", ml, "-", e$message))
  })

  gc()
}

log_msg("=== TÜM DEĞERLENDİRME İŞLEMLERİ TAMAMLANDI ===")
log_msg(paste("Bitiş zamanı:", Sys.time()))
