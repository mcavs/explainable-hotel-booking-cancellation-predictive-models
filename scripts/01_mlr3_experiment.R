# ============================================================
# 01_mlr3_experiment.R
# mlr3proba Survival Analysis — Ana Deney Scripti
# Hotel Booking Cancellation Prediction
# ============================================================
# Akış:
#   1. Veri hazırlama (2 veri seti)
#   2. Veri bölme (%60 train / %20 val / %20 test)
#   3. Vanilla model eğitimi (10-fold CV + final model)
#   4. Tuned model eğitimi (3 yöntem × tunable modeller)
#   5. Performans değerlendirmesi (train + test)
#   6. Survex explainer nesneleri
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
LOG_FILE   <- file.path(BASE_DIR, "experiment_log.txt")

for (d in c(MODEL_DIR, EXPL_DIR, RESULT_DIR, SPLIT_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# =================================================================
# SECTION 1: LIBRARIES
# =================================================================
suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3proba)
  library(mlr3extralearners)
  library(mlr3tuning)
  library(mlr3mbo)
  library(mlr3pipelines)
  library(mlr3learners)
  library(paradox)
  library(survex)
  library(data.table)
  library(survival)
  library(dplyr)
  library(lubridate)
})

lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("=== MLR3PROBA DENEY BAŞLIYOR ===")
log_msg(paste("Seed:", SEED))

# =================================================================
# SECTION 2: DATA PREPARATION
# =================================================================
log_msg("--- SECTION 2: DATA PREPARATION ---")

# --- Dataset 1: hotels.csv ---
log_msg("Dataset 1 yükleniyor...")
h <- fread(file.path(DATA_DIR, "hotels.csv"))

month_map <- c(January=1, February=2, March=3, April=4, May=5, June=6,
               July=7, August=8, September=9, October=10, November=11, December=12)

h[, arrival_month_num := month_map[arrival_date_month]]
h[, arrival_date := as.Date(paste(arrival_date_year, arrival_month_num,
                                   arrival_date_day_of_month, sep = "-"))]
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
log_msg(paste("Dataset 1 hazır:", nrow(ds1), "satır,", ncol(ds1), "kolon"))

# --- Dataset 2: hotel_reservations.csv ---
log_msg("Dataset 2 yükleniyor...")
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
log_msg(paste("Dataset 2 hazır:", nrow(ds2), "satır,", ncol(ds2), "kolon"))

# =================================================================
# SECTION 3: DATA SPLITTING & TASK CREATION
# =================================================================
log_msg("--- SECTION 3: DATA SPLITTING ---")

split_data <- function(dt, seed = 42) {
  set.seed(seed)
  n <- nrow(dt)
  idx <- sample(n)
  t1 <- floor(0.6 * n); t2 <- floor(0.8 * n)
  list(train = dt[idx[1:t1]], val = dt[idx[(t1+1):t2]], test = dt[idx[(t2+1):n]])
}

sp1 <- split_data(ds1); sp2 <- split_data(ds2)
rm(ds1, ds2); gc()

# Subsample Dataset 1 Train to exactly 21,765 rows (matching Dataset 2 train size) to prevent OOM
set.seed(SEED)
if (nrow(sp1$train) > 21765) {
  log_msg(paste("Dataset 1 train seti", nrow(sp1$train), "satırdan 21765 satıra subsample ediliyor..."))
  keep_idx <- sample(seq_len(nrow(sp1$train)), 21765)
  sp1$train <- sp1$train[keep_idx, ]
}

# Save splits
for (p in c("train", "val", "test")) {
  saveRDS(sp1[[p]], file.path(SPLIT_DIR, paste0(p, "_ds1.rds")))
  saveRDS(sp2[[p]], file.path(SPLIT_DIR, paste0(p, "_ds2.rds")))
}
log_msg("Veri bölmeleri kaydedildi.")
log_msg(paste("DS1 train:", nrow(sp1$train), "| val:", nrow(sp1$val), "| test:", nrow(sp1$test)))
log_msg(paste("DS2 train:", nrow(sp2$train), "| val:", nrow(sp2$val), "| test:", nrow(sp2$test)))

# --- mlr3 Task nesneleri ---
mk_task <- function(dt, id) {
  TaskSurv$new(id = id, backend = as.data.frame(dt), time = "time", event = "status")
}

tasks <- list()
for (ds in 1:2) {
  sp  <- if (ds == 1) sp1 else sp2
  pre <- if (ds == 1) "h" else "r"

  tasks[[paste0("train", ds)]]    <- mk_task(sp$train, paste0(pre, "_train"))
  tasks[[paste0("val", ds)]]      <- mk_task(sp$val,   paste0(pre, "_val"))
  tasks[[paste0("test", ds)]]     <- mk_task(sp$test,  paste0(pre, "_test"))

  tv <- rbind(sp$train, sp$val)
  tasks[[paste0("trainval", ds)]] <- mk_task(tv, paste0(pre, "_trainval"))

  # Tuning resampling: train üzerinde eğit, val üzerinde değerlendir
  n_tr <- nrow(sp$train); n_va <- nrow(sp$val)
  rs <- rsmp("custom")
  rs$instantiate(tasks[[paste0("trainval", ds)]],
                 train = list(1:n_tr), test = list((n_tr + 1):(n_tr + n_va)))
  tasks[[paste0("rsmp_tune", ds)]] <- rs

  # SVM subsampled task (5000 satır)
  set.seed(42)
  svm_n   <- min(5000, n_tr)
  svm_ids <- sample(tasks[[paste0("train", ds)]]$row_ids, svm_n)
  tasks[[paste0("train_svm", ds)]] <- tasks[[paste0("train", ds)]]$clone()$filter(svm_ids)

  svm_tv <- rbind(sp$train[sample(.N, svm_n)], sp$val)
  tasks[[paste0("trainval_svm", ds)]] <- mk_task(svm_tv, paste0(pre, "_tv_svm"))
  rs_svm <- rsmp("custom")
  rs_svm$instantiate(tasks[[paste0("trainval_svm", ds)]],
                     train = list(1:svm_n), test = list((svm_n + 1):(svm_n + n_va)))
  tasks[[paste0("rsmp_tune_svm", ds)]] <- rs_svm
}
rm(sp1, sp2); gc()
log_msg("Task nesneleri oluşturuldu.")

# =================================================================
# SECTION 4: MODEL SPECIFICATIONS
# =================================================================
log_msg("--- SECTION 4: MODEL SPECIFICATIONS ---")

# Learner oluşturma yardımcısı
mk_lrn <- function(lid, params = list(), encode = FALSE) {
  lrn_obj <- tryCatch(lrn(lid), error = function(e) {
    log_msg(paste("  Learner bulunamadı:", lid, "-", e$message))
    NULL
  })
  if (is.null(lrn_obj)) return(NULL)

  for (nm in names(params))
    lrn_obj$param_set$values[[nm]] <- params[[nm]]

  if (encode) {
    lrn_obj <- tryCatch(
      as_learner(po("encode", method = "treatment") %>>% lrn_obj),
      error = function(e) {
        log_msg(paste("  Pipeline hatası:", lid, "-", e$message))
        NULL
      })
  }
  lrn_obj
}

# --- Model tanımları ---
# enc: encoding pipeline gerekli mi
# svm: subsample gerekli mi
# ss: search space (tunable modeller için)
specs <- list(
  # ===== VANILLA ONLY =====
  list(id = "kaplan", lid = "surv.kaplan",
       tune = FALSE, enc = FALSE, par = list()),

  list(id = "coxph", lid = "surv.coxph",
       tune = FALSE, enc = FALSE, par = list()),

  list(id = "parametric_weibull", lid = "surv.parametric",
       tune = FALSE, enc = FALSE, par = list(dist = "weibull", type = "aft")),

  list(id = "parametric_lognormal", lid = "surv.parametric",
       tune = FALSE, enc = FALSE, par = list(dist = "lognormal", type = "aft")),

  list(id = "parametric_loglogistic", lid = "surv.parametric",
       tune = FALSE, enc = FALSE, par = list(dist = "loglogistic", type = "aft")),

  # ===== TUNABLE MODELS =====
  list(id = "cv_glmnet", lid = "surv.cv_glmnet",
       tune = TRUE, enc = TRUE, par = list(),
       ss = ps(surv.cv_glmnet.alpha = p_dbl(lower = 0, upper = 1))),

  list(id = "ranger", lid = "surv.ranger",
       tune = TRUE, enc = FALSE, par = list(num.threads = 2),
       ss = ps(num.trees    = p_int(lower = 100, upper = 500), # Lower tree count upper limit to save memory
               mtry         = p_int(lower = 2,   upper = 15),
               min.node.size = p_int(lower = 3,  upper = 50))),

  list(id = "rfsrc", lid = "surv.rfsrc",
       tune = TRUE, enc = FALSE, par = list(ntree = 250), # Set custom limit
       ss = ps(ntree    = p_int(lower = 100, upper = 500), # Lower upper limit
               mtry     = p_int(lower = 2,   upper = 15),
               nodesize = p_int(lower = 3,   upper = 50))),

  list(id = "aorsf", lid = "surv.aorsf",
       tune = TRUE, enc = FALSE, par = list(n_thread = 2),
       ss = ps(n_tree        = p_int(lower = 100, upper = 500), # Lower upper limit
               mtry_ratio    = p_dbl(lower = 0.1, upper = 0.9),
               split_min_obs = p_int(lower = 5,   upper = 50))),

  list(id = "xgboost", lid = "surv.xgboost",
       tune = TRUE, enc = TRUE, par = list(),
       ss = ps(surv.xgboost.nrounds          = p_int(lower = 50,   upper = 500),
               surv.xgboost.max_depth         = p_int(lower = 2,    upper = 8),
               surv.xgboost.eta               = p_dbl(lower = 0.01, upper = 0.3),
               surv.xgboost.min_child_weight  = p_int(lower = 1,    upper = 10))),

  list(id = "cforest", lid = "surv.cforest",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(ntree        = p_int(lower = 50,   upper = 500),
               mtry         = p_int(lower = 2,    upper = 15),
               mincriterion = p_dbl(lower = 0.01, upper = 0.99))),

  list(id = "ctree", lid = "surv.ctree",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(mincriterion = p_dbl(lower = 0.01, upper = 0.99),
               minsplit     = p_int(lower = 10,   upper = 100))),

  list(id = "rpart", lid = "surv.rpart",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(cp       = p_dbl(lower = 0.001, upper = 0.1),
               minsplit = p_int(lower = 10,    upper = 100),
               maxdepth = p_int(lower = 3,     upper = 15))),

  list(id = "coxboost", lid = "surv.coxboost",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(maxstepno = p_int(lower = 50, upper = 500),
               penalty   = p_dbl(lower = 1,  upper = 1000))),

  list(id = "glmboost", lid = "surv.glmboost",
       tune = TRUE, enc = TRUE, par = list(),
       ss = ps(surv.glmboost.mstop = p_int(lower = 50,   upper = 500),
               surv.glmboost.nu    = p_dbl(lower = 0.01, upper = 0.5))),

  list(id = "blackboost", lid = "surv.blackboost",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(mstop    = p_int(lower = 50,   upper = 500),
               nu       = p_dbl(lower = 0.01, upper = 0.5))),

  list(id = "gamboost", lid = "surv.gamboost",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(mstop = p_int(lower = 50,   upper = 500),
               nu    = p_dbl(lower = 0.01, upper = 0.5))),

  list(id = "penalized", lid = "surv.penalized",
       tune = TRUE, enc = TRUE, par = list(),
       ss = ps(surv.penalized.lambda1 = p_dbl(lower = 0.01, upper = 100),
               surv.penalized.lambda2 = p_dbl(lower = 0.01, upper = 100))),

  list(id = "flexible", lid = "surv.flexible",
       tune = TRUE, enc = FALSE, par = list(),
       ss = ps(k = p_int(lower = 1, upper = 5))),

  list(id = "svm", lid = "surv.svm",
       tune = TRUE, enc = TRUE, par = list(),
       ss = ps(surv.svm.gamma.mu = p_dbl(lower = 0.001, upper = 10)),
       svm = TRUE)
)

log_msg(paste("Toplam model tipi:", length(specs)))

# =================================================================
# SECTION 5: VANILLA TRAINING (10-fold CV + final model)
# =================================================================
log_msg("--- SECTION 5: VANILLA TRAINING ---")

for (spec in specs) {
  for (ds in 1:2) {
    label <- paste0("vanilla_", spec$id, "_ds", ds)

    # Daha önce eğitilmiş mi kontrol et
    fpath <- file.path(MODEL_DIR, paste0(label, ".rds"))
    if (file.exists(fpath)) {
      log_msg(paste("Atlanıyor (mevcut):", label))
      next
    }

    log_msg(paste("Eğitim:", label))

    # SVM subsample kontrolü
    task_key <- if (isTRUE(spec$svm)) paste0("train_svm", ds) else paste0("train", ds)
    task_tr  <- tasks[[task_key]]

    tryCatch({
      # Learner oluştur
      lrn_obj <- mk_lrn(spec$lid, spec$par, spec$enc)
      if (is.null(lrn_obj)) next

      # 10-fold CV
      tryCatch({
        rr <- resample(task_tr, lrn_obj$clone(), rsmp("cv", folds = 10), store_models = FALSE)
        cv_c <- rr$aggregate(msr("surv.cindex"))
        log_msg(paste("  CV C-Index:", round(cv_c, 4)))
        rm(rr); gc()
      }, error = function(e) {
        log_msg(paste("  CV hatası:", e$message))
      })

      # Final model eğit
      lrn_final <- mk_lrn(spec$lid, spec$par, spec$enc)
      lrn_final$train(task_tr)

      # Kaydet
      saveRDS(lrn_final, fpath)
      log_msg(paste("  ✓ Kaydedildi:", label))
      rm(lrn_final, lrn_obj); gc()

    }, error = function(e) {
      log_msg(paste("  ✗ Hata:", label, "-", e$message))
    })

    gc()
  }
}

log_msg("=== VANILLA TRAINING TAMAMLANDI ===")

# =================================================================
# SECTION 6: TUNED TRAINING (3 yöntem)
# =================================================================
log_msg("--- SECTION 6: TUNED TRAINING ---")

tuning_methods <- list(
  list(id = "random",   mk_tuner = function() tnr("random_search"),              evals = 50),
  list(id = "bayesian", mk_tuner = function() tnr("mbo"),                         evals = 30),
  list(id = "grid",     mk_tuner = function() tnr("grid_search", resolution = 5), evals = NULL)
)

for (spec in specs) {
  if (!isTRUE(spec$tune)) next

  for (ds in 1:2) {
    for (tm in tuning_methods) {
      label <- paste0("tuned_", tm$id, "_", spec$id, "_ds", ds)

      fpath <- file.path(MODEL_DIR, paste0(label, ".rds"))
      if (file.exists(fpath)) {
        log_msg(paste("Atlanıyor (mevcut):", label))
        next
      }

      log_msg(paste("Tuning:", label))

      tryCatch({
        # Task ve resampling seç
        if (isTRUE(spec$svm)) {
          task_tv  <- tasks[[paste0("trainval_svm", ds)]]
          task_tr  <- tasks[[paste0("train_svm", ds)]]
          rs_tune  <- tasks[[paste0("rsmp_tune_svm", ds)]]
        } else {
          task_tv  <- tasks[[paste0("trainval", ds)]]
          task_tr  <- tasks[[paste0("train", ds)]]
          rs_tune  <- tasks[[paste0("rsmp_tune", ds)]]
        }

        # Learner oluştur
        lrn_obj <- mk_lrn(spec$lid, spec$par, spec$enc)
        if (is.null(lrn_obj)) next

        # Tuning çalıştır
        instance <- tune(
          tuner       = tm$mk_tuner(),
          task        = task_tv,
          learner     = lrn_obj,
          resampling  = rs_tune,
          measure     = msr("surv.cindex"),
          term_evals  = tm$evals,
          search_space = spec$ss
        )

        best <- instance$result_learner_param_vals
        log_msg(paste("  Best C-Index:", round(instance$result_y[["surv.cindex"]], 4)))

        # Best params ile train setinde final model eğit
        lrn_best <- mk_lrn(spec$lid, spec$par, spec$enc)
        lrn_best$param_set$values <- modifyList(lrn_best$param_set$values, best)
        lrn_best$train(task_tr)

         # Model + tuning bilgileri kaydet
         obj <- list(
           learner     = lrn_best,
           best_params = best,
           best_score  = instance$result_y,
           archive     = tryCatch(as.data.table(instance$archive), error = function(e) NULL)
         )
         saveRDS(obj, fpath)
         log_msg(paste("  ✓ Kaydedildi:", label))
         rm(lrn_best, lrn_obj, instance, obj); gc()

       }, error = function(e) {
         log_msg(paste("  ✗ Tuning hatası:", label, "-", e$message))
       })

       gc()
    }
  }
}

log_msg("=== TUNED TRAINING TAMAMLANDI ===")

# =================================================================
# SECTION 7: PERFORMANCE EVALUATION
# =================================================================
log_msg("--- SECTION 7: PERFORMANCE EVALUATION ---")

# Metrik listesi
base_measures <- msrs(c("surv.cindex", "surv.graf", "surv.rcll"))

eval_model <- function(model_file, task_eval, label) {
  tryCatch({
    obj <- readRDS(file.path(MODEL_DIR, model_file))
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    pred <- lrn_obj$predict(task_eval)

    # Temel metrikler
    scores <- tryCatch(pred$score(base_measures), error = function(e) {
      c(surv.cindex = NA_real_, surv.graf = NA_real_, surv.rcll = NA_real_)
    })

    # IAUC (ayrı dene, bazı modellerde distr olmayabilir)
    iauc <- tryCatch(pred$score(msr("surv.uno_auc")),
                     error = function(e) c(surv.uno_auc = NA_real_))

    c(scores, iauc)
  }, error = function(e) {
    log_msg(paste("  Eval hatası:", label, "-", e$message))
    c(surv.cindex = NA, surv.graf = NA, surv.rcll = NA, surv.uno_auc = NA)
  })
}

model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
perf_rows <- list()

for (mf in model_files) {
  ds <- as.integer(gsub(".*_ds(\\d+)\\.rds$", "\\1", mf))
  if (is.na(ds) || !(ds %in% 1:2)) next

  ml <- gsub("\\.rds$", "", mf)

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
  log_msg(paste("Eval:", ml, "| Test C:", round(test_sc["surv.cindex"], 4)))
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

  tryCatch({
    obj <- readRDS(file.path(MODEL_DIR, mf))
    lrn_obj <- if (is.list(obj) && "learner" %in% names(obj)) obj$learner else obj

    task_tst <- tasks[[paste0("test", ds)]]
    test_dt  <- as.data.table(task_tst$data())
    x_test   <- test_dt[, !c("time", "status"), with = FALSE]
    y_test   <- Surv(test_dt$time, test_dt$status)

    # survex explain (mlr3 learner desteği)
    expl <- explain(
      model   = lrn_obj,
      data    = as.data.frame(x_test),
      y       = y_test,
      label   = ml,
      verbose = FALSE
    )

    saveRDS(expl, expl_file)
    log_msg(paste("  ✓ Explainer kaydedildi:", ml))

  }, error = function(e) {
    log_msg(paste("  ✗ Explainer hatası:", ml, "-", e$message))
  })

  gc()
}

log_msg("=== TÜM İŞLEMLER TAMAMLANDI ===")
log_msg(paste("Bitiş zamanı:", Sys.time()))
