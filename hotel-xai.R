# Profil oluştur (Artık hepsi kategorik olduğu için 'type = partial' sorunsuz çalışacaktır)
rsf_pdp_fixed <- model_profile(rsf_explainer_fixed)

# Görselleştir
plot(rsf_pdp_fixed)