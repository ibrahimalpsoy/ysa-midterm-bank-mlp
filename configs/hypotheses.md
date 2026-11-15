# Önceden belirlenmiş deney hipotezleri

Bu hipotezler development EDA'sından sonra, nested CV çalıştırılmadan ve dış kat sonuçları görülmeden önce yazılmıştır.
Her hipotez, belirtilen karşılaştırmanın **iç CV mean AP** değerleriyle (her dış katta ve tam development'ta) test edilir.

**H1. Optimizer:** Aynı başlangıç öğrenme oranında (1e-3) SGD+Momentum (M3), AdamW'ye (M1) göre daha yavaş yakınsar.
Daha geç en iyi epoch'a ulaşır ya da 150 epoch sınırına takılır ve iç CV AP'si M1'den düşük veya eşit olur.
Adam (M2) ile AdamW (M1) arasındaki fark, weight decay = 1e-4 düzeyinde ihmal edilebilir kalır (|ΔAP| < 0.01).

**H2. Ölçekleme:** `campaign`, `previous` ve `age` sağa çarpık ve aykırı değerlidir; makroekonomik değişkenler ise az sayıda
değerde kümelenir. Bu nedenle MinMaxScaler (M4), gözlemlerin çoğunu dar bir aralığa sıkıştırarak StandardScaler'a (M1)
göre eşit veya daha düşük AP verir. Kat-içi kırpma (clipping) AP'yi belirgin biçimde düşürmez.

**H3. Kapasite:** Development'ta yaklaşık 3.300 gözlem ve ~360 pozitif örnek vardır. Bu nedenle daha büyük (64,32) mimari (M5),
(32,) mimarisine (M1) göre AP'yi anlamlı biçimde artırmaz (|ΔAP| < 0.01 veya düşüş). Daha erken en iyi epoch ile aşırı uyum eğilimi gösterir.

**H4. Learning-rate düşürme:** Early stopping açıkken ReduceLROnPlateau'nun kapatılması (M6), AP'yi M1'e göre belirgin değiştirmez.
Etkisi esas olarak en iyi epoch ve eğitim süresinde görülür.

**H5. Doğrusal referans:** Hedefle ilişkili sinyalin önemli kısmı güçlü korelasyonlu makroekonomik değişkenlerde ve
`poutcome` / `pdays` bilgisindedir. Bu yüzden en iyi MLP, dış CV'de LogisticRegression'a göre AP'de küçük bir fark (< 0.03) yaratır.
Her iki model de Dummy (AP ≈ pozitif oran ≈ 0.11) referansını açıkça geçer.
