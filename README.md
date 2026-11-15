# Yapay Sinir Ağları Ara Sınav Projesi

İbrahim Alpsoy, 254329040
Ders: Yapay Sinir Ağları (Doç. Dr. Gökalp Tulum)

Bu projede UCI Bank Marketing veri setindeki bank-additional.csv dosyasını kullanarak bir müşterinin vadeli mevduata abone olup olmayacağını PyTorch ile yazdığım bir MLP modeliyle tahmin etmeye çalıştım. Hiperparametreleri nested cross validation ile seçtim. 6 manuel ve 6 random search denemesi yaptım, sonra en iyi modeli ayırdığım test verisinde denedim.

Raporun tamamı report.pdf dosyasında.

## Veri

Veri seti repoda yok, buradan indirilebilir:
https://archive.ics.uci.edu/dataset/222/bank+marketing (DOI: 10.24432/C5K306)

Zip dosyasının içindeki bank-additional.zip içinde bulunan bank-additional.csv dosyasını data klasörüne koymak yeterli. 01 notebook dosyayı bulamazsa kendisi indirmeye çalışıyor.

Not: duration sütununu baştan çıkardım çünkü görüşme bitmeden bu değer bilinmiyor.

## Seed

Seed değerini öğrenci numaramın son 5 hanesine 2026 ekleyerek buldum:
29040 + 2026 = 31066

Diğer seedleri bu sayıya 0, 1, 2 gibi sabit sayılar ekleyerek ürettim (bölünme için 31069, dış CV için 31070, iç CV için 31071 gibi). Hepsi src/config.py ve configs/training.json içinde yazıyor. Final modeli 3 farklı seed ile eğittim: 31068, 31166 ve 31266. Ana seed olarak 31068 kullandım.

## Nasıl çalıştırılır

```bash
git clone https://github.com/ibrahimalpsoy/ysa-midterm-bank-mlp.git
cd ysa-midterm-bank-mlp
pip install -r requirements.txt
```

Sonra notebookları sırayla çalıştırmak gerekiyor:

1. 01_eda.ipynb: veriyi okuyup kontrol ediyor, %80 development ve %20 test olarak bölüyor, sadece development üzerinde EDA yapıyor
2. 02_nested_cv.ipynb: manuel ve random search adaylarını nested CV ile deniyor, final modeli eğitiyor ve test tahminlerini etiketlere bakmadan results/test_predictions_locked.csv dosyasına yazıyor
3. 03_final_test.ipynb: kilitli tahminlerle test sonuçlarını, bootstrap güven aralıklarını ve grafikleri çıkarıyor

Colab'da açınca ilk hücre repoyu klonlayıp gerekli paketleri kuruyor. Her şey CPU üzerinde çalışıyor, bende toplam birkaç dakika sürdü.

## Kullandığım sürümler

Python 3.13.18, torch 2.15.0, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2
Bilgisayar: 2 çekirdekli Linux, GPU yok

## Klasörler

- src: kodların asıl kısmı (veri okuma, ön işleme, MLP modeli, nested CV, metrikler)
- configs: arama uzayı, M1-M6 ve R1-R6 adayları, eğitim ayarları ve hipotezlerim. Bunları nested CV çalıştırmadan önce commitledim.
- results: bölünme dosyası, arama sonuçları, dış CV sonuçları, final model ayarları, kilitli test tahminleri ve test sonuçları
- figures: raporda kullandığım grafikler
- report: raporun LaTeX kaynağı ve tabloları sonuç dosyalarından üreten make_tables.py

## Test kilidi hakkında

Test tahminlerini etiketlere bakmadan ayrı bir commit ile kaydettim (d463f75). Test sonuçlarını bu committen sonra hesapladım. Veri seti herkese açık olduğu için bu tam bir kör test sayılmaz ama hangi adımı hangi sırayla yaptığımı gösteriyor.

## Kısaca sonuçlar

- Dış CV AP: manuel arama 0.422 ± 0.061, random search 0.405 ± 0.044, lojistik regresyon 0.414 ± 0.061
- Seçilen model: R5 (AdamW, 16 nöron, tanh, dropout 0.2)
- Test AP: 0.440 (%95 güven aralığı 0.336 - 0.561), ROC-AUC: 0.784
- MLP, lojistik regresyondan belirgin bir şekilde daha iyi çıkmadı. Bu veride duration olmadan çok fazla bilgi kalmıyor gibi görünüyor.

Rapor tekrar üretilmek istenirse:
```bash
python report/make_tables.py
cd report && xelatex report.tex && xelatex report.tex
```
