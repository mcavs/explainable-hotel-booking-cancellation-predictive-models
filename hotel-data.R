library(tidyverse)
library(survival)
library(randomForestSRC)
library(survex)

# Veriyi çekelim
url <- "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
hotels <- read_csv(url)

# Sayısal değişkenleri anlamlı kategorilere bölelim
model_data_fixed <- hotels %>%
  select(is_canceled, lead_time, hotel, market_segment, deposit_type, 
         customer_type, adr, total_of_special_requests) %>%
  mutate(
    # ADR (Fiyat): Çeyrekliklere veya belirli eşiklere göre bölelim
    adr = cut(adr, 
              breaks = c(-Inf, 70, 120, Inf), 
              labels = c("Low Price", "Medium Price", "High Price")),
    
    # Özel İstek Sayısı: 0, 1 ve 2+ şeklinde gruplayalım
    total_of_special_requests = as.factor(ifelse(total_of_special_requests >= 2, "2+", 
                                                 as.character(total_of_special_requests))),
    
    # Diğer karakterleri faktör yapalım
    across(where(is.character), as.factor)
  ) %>%
  drop_na() %>%
  sample_n(5000) # Hız için 5000 satır

# Lead time 0 hatasını engelleyelim
model_data_fixed$lead_time[model_data_fixed$lead_time == 0] <- 0.5