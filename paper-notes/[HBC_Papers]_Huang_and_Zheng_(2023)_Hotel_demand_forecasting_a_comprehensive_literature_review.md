# [HBC Papers] Huang & Zheng (2023) Hotel demand forecasting: a comprehensive literature review

* This study aims to provide a comprehensive review of hotel demand forecasting to identify its key fundamentals and evolution and future research directions and trends to advance the field.

* Articles on hotel demand modeling and forecasting were identified and rigorously selected using transparent inclusion and exclusion criteria, obtaining a final sample of 85 empirical studies for comprehensive content analysis.

* Synthesis of the literature highlights that hotel forecasting based on historical demand data dominates the research, while reservation/cancellation data and combined data gradually attracted research attention in recent years.

* In terms of model evolution, time series and AI-based models are the most popular models for hotel demand forecasting, with numerous recent studies focusing on hybrid models and AI-based models.

* This study represents the first systematic review of the literature on hotel demand forecasting from the dual perspective of data sources and methodological development.

* Unlike tourism demand forecasting which relies primarily on aggregated historical demand data, hotel demand forecasting deals with higher complexity due to advance booking/cancellation data, competitive behavior, online reviews, and granular spatial effects.

* The development of hotel demand forecasting can be divided into two distinct phases: a slow growth phase (1990–2015) dominated by time series models in non-hospitality journals, and a rapid growth phase (2016–present) characterized by big data integration and AI-based models.

* Historical demand data in hotel forecasting is analyzed across various frequencies (monthly, weekly, daily), addressing key issues such as double seasonality, non-integer period patterns, and regional spatial agglomeration effects.

* Reservation demand forecasting operates under two primary assumptions regarding known and future bookings: independence (additive pickup models) and proportional relationship (multiplicative pickup models).

* Reservation cancellation forecasting has emerged as a critical subfield, managed either as a time-series prediction or as a classification problem estimating the cancellation probability of individual orders using machine learning (e.g., XGBoost).

* Combined data approaches integrate hybrid demand data (historical + advance bookings) or multisource data (economic factors, weather, search query indices like Google Trends, website traffic, and competitor pricing) to enhance predictive accuracy.

* Time series models (e.g., ARIMA/SARIMA, Holt-Winters, TBATS, booking curve models, pickup models) remain popular for short-term daily and monthly forecasting due to ease of data collection.

* Econometric models (e.g., ADLM, MIDAS, VAR) are unique in quantifying causal relationships and explanatory variable impacts, though they are less frequently applied at the individual hotel level due to data limitations and frequency mismatches.

* AI-based models (e.g., ANN, LSTM, XGBoost, Support Vector Regression) excel at handling non-linear, highly volatile, and high-frequency data without requiring prior assumptions about data distribution.

* Single-hotel demand forecasting accounts for the majority of studies (45 out of 85) and relies heavily on daily data using time series and AI-based models (94%), whereas destination or hotel-group level forecasting leans toward monthly/yearly data and econometric/hybrid models.

* Future research should focus on developing hotel-level econometric models that incorporate explanatory variables such as price, online word-of-mouth, and competitor dynamics.

* Exploring complex spatial dependencies among hotels, such as simultaneous spillover and competition effects, presents a major challenge and opportunity for future forecasting models.

* Integrating unstructured data from social media (textual reviews, images, sentiment analysis) while balancing prediction accuracy and model error terms represents a vital future research path.

* Combining AI-based models (high predictive power) with econometric models (high interpretability) into hybrid frameworks offers a promising direction to overcome the black-box nature of machine learning.

* Model robustness against external shocks and structural shifts (e.g., COVID-19, economic crises) needs enhancement by combining quantitative models with qualitative judgment methods such as Delphi adjustments and scenario planning.
