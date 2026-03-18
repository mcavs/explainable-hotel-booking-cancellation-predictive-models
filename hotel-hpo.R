# mtry ve nodesize için otomatik arama
# stepFactor: Her adımda mtry'nin ne kadar değişeceği
# improve: Hata oranındaki minimum iyileşme eşiği
tuned_params <- tune.rfsrc(Surv(lead_time, is_canceled) ~ ., 
                           data = model_data_fixed,
                           ntreeTry = 100,      # Arama sırasında kullanılacak ağaç sayısı
                           stepFactor = 1.5, 
                           improve = 0.01, 
                           trace = TRUE, 
                           doBest = TRUE)       # En iyi modeli otomatik döndürür

# En iyi parametreleri görelim
print(tuned_params$optimal)

# Parametre tablosu oluşturalım
param_grid <- expand.grid(
  ntree = c(100, 300, 500),
  nodesize = c(5, 10, 20),
  mtry = c(2, 4)
)

# Sonuçları saklamak için bir liste
results <- list()

for(i in 1:nrow(param_grid)) {
  model <- rfsrc(Surv(lead_time, is_canceled) ~ ., 
                 data = model_data_fixed,
                 ntree = param_grid$ntree[i],
                 nodesize = param_grid$nodesize[i],
                 mtry = param_grid$mtry[i])
  
  # C-index (Concordance Index) başarısını kaydedelim
  results[[i]] <- 1 - get.cindex(model$yvar[,1], model$yvar[,2], model$predicted.oob)
}

# En yüksek C-index değerine sahip parametreyi seçebilirsiniz
param_grid$c_index <- unlist(results)
best_params <- param_grid[which.max(param_grid$c_index), ]