# ============================================================
# 07_local_evaluation_fixed.R
# Yerel bilgisayardaki tüm modeller üzerinde performans değerlendirmesi:
#   - FIX: Test setinde, eğitim setinde olmayan kategori seviyelerine
#     sahip SATIRLAR veri setinden çıkarılır. (cforest/ctree hatasını kesin çözer)
#   - Metrikler: C-Index, IBS, RCLL ve Uno's AUC (iauc)
# ============================================================

BASE_DIR   <- "/Users/mustafacavus/Documents/HotelBook"
MODEL_DIR  <- file.path(BASE_DIR, "models", "models")
RESULT_DIR <- file.path(BASE_DIR, "results")
SPLIT_DIR  <- file.path(BASE_DIR, "datasets", "splits")
LOG_FILE   <- file.path(BASE_DIR, "local_eval_log.txt")

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
  library(data.table)
  library(survival)
})

lgr::get_logger("mlr3")$set_threshold("warn")

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("=== YEREL PERFORMANS DEĞERLENDİRMESİ BAŞLIYOR ===")

# =================================================================
# VERİ YÜKLEME VE YENİ SEVİYELİ SATIRLARIN SİLİNMESİ
# =================================================================
log_msg("Eğitim ve test setleri yükleniyor...")
train_dt <- list(
  readRDS(file.path(SPLIT_DIR, "train_ds1.rds")),
  readRDS(file.path(SPLIT_DIR, "train_ds2.rds"))
)
test_dt <- list(
  readRDS(file.path(SPLIT_DIR, "test_ds1.rds")),
  readRDS(file.path(SPLIT_DIR, "test_ds2.rds"))
)

# Test setinde, eğitim setinde bulunmayan yeni kategori seviyelerine sahip satırları drop et
for (ds in 1:2) {
  tr <- train_dt[[ds]]
  ts <- test_dt[[ds]]
  
  factor_cols <- names(tr)[vapply(tr, is.factor, logical(1))]
  initial_rows <- nrow(ts)
  
  for (col in factor_cols) {
    tr_levels <- levels(tr[[col]])
    
    # Test setindeki değerlerin karakter karşılıkları
    ts_vals <- as.character(ts[[col]])
    
    # Sadece eğitim setinde tanımlı seviyeleri içeren satırları tut
    valid_idx <- ts_vals %in% tr_levels
    ts <- ts[valid_idx, ]
    
    # Faktör seviyelerini eğitim setiyle eşitle
    ts[[col]] <- factor(as.character(ts[[col]]), levels = tr_levels)
  }
  
  test_dt[[ds]] <- ts
  removed_rows <- initial_rows - nrow(ts)
  if (removed_rows > 0) {
    log_msg(paste("Dataset", ds, "üzerinden yeni seviyeli", removed_rows, "satır temizlendi. Kalan satır:", nrow(ts)))
  }
}

# Task oluşturma fonksiyonu
mk_task <- function(dt, id) {
  TaskSurv$new(id = id, backend = as.data.frame(dt), time = "time", event = "status")
}

tasks <- list()
for (ds in 1:2) {
  pre <- if (ds == 1) "h" else "r"
  tasks[[paste0("train", ds)]] <- mk_task(train_dt[[ds]], paste0(pre, "_train"))
  tasks[[paste0("test",  ds)]] <- mk_task(test_dt[[ds]],  paste0(pre, "_test"))
}

# =================================================================
# MODELLERİN DEĞERLENDİRİLMESİ
# =================================================================
base_measures <- msrs(c("surv.cindex", "surv.graf", "surv.rcll"))
model_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = FALSE)
log_msg(paste("Toplam değerlendirilecek model sayısı:", length(model_files)))

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
  log_msg(paste("  ✓ Eval:", ml, "| Test C-Index:", round(test_sc["surv.cindex"], 4), "| Test IBS:", round(test_sc["surv.graf"], 4)))
  
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

log_msg("=== YEREL DEĞERLENDİRME TAMAMLANDI ===")
