# ============================================================
# 06_critical_difference.R
# Modellerin performansları (C-Index ve IBS) üzerinde:
#   - Friedman Testi (Modeller arası anlamlı fark var mı?)
#   - Nemenyi Post-hoc Testi (Modellerin ikili karşılaştırılması)
#   - Critical Difference (CD) diyagramlarının oluşturulması ve kaydedilmesi
# ============================================================

BASE_DIR   <- "/Users/mustafacavus/Documents/HotelBook"
RESULT_DIR <- file.path(BASE_DIR, "results")
PLOT_DIR   <- file.path(RESULT_DIR, "plots")
LOG_FILE   <- file.path(BASE_DIR, "cd_analysis_log.txt")

dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

# =================================================================
# LIBRARIES
# =================================================================
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("=== CRITICAL DIFFERENCE VE ISTATISTIKSEL ANALIZ BAŞLIYOR ===")

# =================================================================
# VERİ YÜKLEME VE ORTAK ISIM EŞLEŞTİRME
# =================================================================
perf_file <- file.path(RESULT_DIR, "performance_all.csv")
if (!file.exists(perf_file)) {
  stop("Performans sonuçları dosyası bulunamadı!")
}

perf <- fread(perf_file)

# Model adından ds1/ds2 takısını çıkararak temel model adını oluştur
perf[, base_model := gsub("_ds[12]$", "", model)]

# Sadece her iki veri setinde de tam sonuç veren modelleri belirle
# C-Index ve IBS değerleri NA/NaN olmayan satırları seç
perf_valid <- perf[!is.na(test_cindex) & !is.na(test_ibs) & !is.nan(test_cindex) & !is.nan(test_ibs)]

# Her iki dataset (1 ve 2) için de kaydı olan base_model'leri bul
model_counts <- perf_valid[, .N, by = base_model]
valid_base_models <- model_counts[N == 2, base_model]

log_msg(paste("Analiz edilecek ortak model sayısı:", length(valid_base_models)))
log_msg(paste("Dahil edilen modeller:", paste(valid_base_models, collapse = ", ")))

perf_filtered <- perf_valid[base_model %in% valid_base_models]

# CD diyagramı çizim fonksiyonu
plot_cd_diagram <- function(ranks_df, cd_val, title_text, filename) {
  ranks_df <- ranks_df[order(ranks_df$mean_rank), ]
  n_models <- nrow(ranks_df)
  
  p <- ggplot(ranks_df, aes(x = mean_rank, y = reorder(model, -mean_rank))) +
    geom_point(color = "#1f77b4", size = 4) +
    geom_segment(aes(x = mean_rank, xend = mean_rank, y = model, yend = model), color = "grey") +
    theme_minimal() +
    labs(
      title = title_text,
      subtitle = paste0("Critical Difference (CD) = ", round(cd_val, 4), " (alpha = 0.05)"),
      x = "Ortalama Sıralama (Düşük Daha İyi)",
      y = "Model"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      panel.grid.major.y = element_blank()
    )
  
  # CD aralığındaki (anlamlı fark olmayan) modelleri bağlayan yatay barlar
  for (i in 1:(n_models - 1)) {
    for (j in (i + 1):n_models) {
      diff_val <- abs(ranks_df$mean_rank[i] - ranks_df$mean_rank[j])
      if (diff_val <= cd_val) {
        p <- p + annotate("segment", 
                          x = ranks_df$mean_rank[i], 
                          xend = ranks_df$mean_rank[j], 
                          y = n_models - i + 1 - 0.08 * (j - i), 
                          yend = n_models - i + 1 - 0.08 * (j - i), 
                          color = "red", linewidth = 1.1, alpha = 0.4)
      }
    }
  }
  
  ggsave(file.path(PLOT_DIR, filename), plot = p, width = 9, height = 7, dpi = 300)
}

# =================================================================
# C-INDEX ANALİZİ
# =================================================================
log_msg("--- C-INDEX ANALİZİ ---")

cindex_mat <- dcast(perf_filtered, base_model ~ dataset, value.var = "test_cindex")
row_names <- cindex_mat$base_model
cindex_mat <- as.matrix(cindex_mat[, -1])
rownames(cindex_mat) <- row_names

# Friedman Testi
friedman_cindex <- friedman.test(cindex_mat)
log_msg(paste("Friedman Testi P-Değeri (C-Index):", format.pval(friedman_cindex$p.value)))

# Ranks
k <- nrow(cindex_mat)
N <- ncol(cindex_mat)
ranks_cindex <- apply(-cindex_mat, 2, rank)
mean_ranks_cindex <- rowMeans(ranks_cindex)
ranks_df_cindex <- data.frame(
  model = names(mean_ranks_cindex),
  mean_rank = mean_ranks_cindex,
  stringsAsFactors = FALSE
)

# Nemenyi q_alpha değeri (alpha=0.05 için k model sayısına göre)
# k = 13 model için q_0.05 yaklaşık 4.82'dir.
q_alpha <- 4.820 
cd_cindex <- q_alpha * sqrt((k * (k + 1)) / (6 * N))
log_msg(paste("Critical Difference (C-Index):", round(cd_cindex, 4)))

plot_cd_diagram(
  ranks_df = ranks_df_cindex, 
  cd_val = cd_cindex, 
  title_text = "C-Index Modeller Arası Critical Difference (CD) Diyagramı", 
  filename = "cd_diagram_cindex.png"
)
log_msg("C-Index CD Diyagramı kaydedildi.")

# =================================================================
# IBS ANALİZİ
# =================================================================
log_msg("--- IBS ANALİZİ ---")

ibs_mat <- dcast(perf_filtered, base_model ~ dataset, value.var = "test_ibs")
row_names_ibs <- ibs_mat$base_model
ibs_mat <- as.matrix(ibs_mat[, -1])
rownames(ibs_mat) <- row_names_ibs

# Friedman Testi (IBS)
friedman_ibs <- friedman.test(ibs_mat)
log_msg(paste("Friedman Testi P-Değeri (IBS):", format.pval(friedman_ibs$p.value)))

# Ranks
ranks_ibs <- apply(ibs_mat, 2, rank)
mean_ranks_ibs <- rowMeans(ranks_ibs)
ranks_df_ibs <- data.frame(
  model = names(mean_ranks_ibs),
  mean_rank = mean_ranks_ibs,
  stringsAsFactors = FALSE
)

cd_ibs <- q_alpha * sqrt((k * (k + 1)) / (6 * N))
log_msg(paste("Critical Difference (IBS):", round(cd_ibs, 4)))

plot_cd_diagram(
  ranks_df = ranks_df_ibs, 
  cd_val = cd_ibs, 
  title_text = "IBS Modeller Arası Critical Difference (CD) Diyagramı", 
  filename = "cd_diagram_ibs.png"
)
log_msg("IBS CD Diyagramı kaydedildi.")

# =================================================================
# RAPORLAMA
# =================================================================
log_msg("--- ANALİZ RAPORU ---")
log_msg(paste("C-Index Friedman test p-value:", format.pval(friedman_cindex$p.value)))
log_msg(paste("IBS Friedman test p-value:", format.pval(friedman_ibs$p.value)))
log_msg("=== CRITICAL DIFFERENCE DİYAGRAMLARI TAMAMLANDI ===")
