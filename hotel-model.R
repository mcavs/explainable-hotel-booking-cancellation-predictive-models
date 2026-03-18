# Modeli eğit
# rsf_fixed <- rfsrc(Surv(lead_time, is_canceled) ~ ., data = model_data_fixed)
rsf_fixed <- rfsrc(Surv(lead_time, is_canceled) ~ ., 
                   data     = model_data_fixed,
                   ntree    = 300,
                   nodesize = 20,
                   mtry     = 2) 

# Tahmin fonksiyonu (Daha önceki ile aynı)
predict_survival_rsf <- function(model, newdata, times) {
  pred <- predict(model, newdata = newdata)
  surv <- pred$survival
  time_idx <- findInterval(times, pred$time.interest)
  time_idx[time_idx == 0] <- 1
  return(surv[, time_idx, drop = FALSE])
}

# Explainer oluştur
rsf_explainer_fixed <- explain_survival(rsf_fixed, 
                                        data = model_data_fixed %>% select(-is_canceled, -lead_time),
                                        y = Surv(model_data_fixed$lead_time, model_data_fixed$is_canceled),
                                        predict_survival_function = predict_survival_rsf)