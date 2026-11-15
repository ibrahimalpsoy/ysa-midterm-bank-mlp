# MLP ile Sınıflandırma: EDA, Ön İşleme, Hiperparametre Optimizasyonu ve Nested Cross-Validation

Yapay Sinir Ağları dersi, yüksek lisans ara sınav projesi. İkili sınıflandırma problemi UCI Bank Marketing veri setinin
`bank-additional.csv` dosyası (4.119 gözlem, 20 girdi) üzerinde ele alınır. Model, PyTorch ile yazılmış bir MLP'dir.

- **Hedef:** `y` (yes = 1, no = 0). **`duration` tüm modellerden çıkarılmıştır** (görüşme öncesinde bilinmez).
- **Birincil metrik:** Average Precision (AP, `average_precision_score`).
- **Protokol:** %80 development / %20 bağımsız test. Development içinde 3×3 nested stratified CV kullanılır.
  Manuel (M1–M6) ve otomatik (R1–R6, random search) adaylar aynı iç katlarda değerlendirilir. Test tahminleri ayrı bir commit ile kilitlenir.

## Seed politikası
Temel seed = öğrenci numarasının son 5 hanesi + 2026 = **29040 + 2026 = 31066**. Alt seed'ler sabit ofsetlerle türetilir
(`src/config.py`, `configs/training.json`):

| Amaç | Seed |
|---|---|
| Python `random` | 31066 |
| NumPy | 31067 |
| PyTorch (ağırlık başlangıcı, birincil) | 31068 |
| Dev/test bölünmesi | 31069 |
| Outer CV | 31070 |
| Inner CV | 31071 |
| Random search (ParameterSampler) | 31072 |
| Early-stop alt bölünmesi | 31073 |
| Bootstrap | 31074 |
| Seed kararlılığı (3 ağırlık seed'i) | 31068 (birincil), 31166, 31266 |

## Veri
Ham veri repoya eklenmemiştir. Resmî kaynak: UCI Machine Learning Repository, *Bank Marketing*
(Moro, Rita ve Cortez, 2014), https://archive.ics.uci.edu/dataset/222/bank+marketing, DOI: 10.24432/C5K306.
`01_eda.ipynb` dosyayı bulamazsa resmî zip'i indirir ve yalnızca `bank-additional/bank-additional.csv` dosyasını `data/` klasörüne çıkarır.
Manuel kullanım için bu dosyayı `data/bank-additional.csv` yoluna koymanız yeterlidir. Kullanılan dosyanın MD5 değeri: `ee09b855dd692f74099ac5cba74f3e54`.

## Kurulum ve çalıştırma
```bash
git clone https://github.com/ibrahimalpsoy/ysa-midterm-bank-mlp.git
cd ysa-midterm-bank-mlp
pip install -r requirements.txt
python -m src.make_configs          # (zaten commit edilmiş) adayları yeniden üretir; 02 notebook birebir eşitliği doğrular
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_nested_cv.ipynb
# -> results/test_predictions_locked.csv ayrı commit ile kilitlenir
jupyter nbconvert --to notebook --execute --inplace 03_final_test.ipynb
```
**Google Colab:** Notebook'u açın. İlk hücredeki `REPO_URL` repoyu klonlar ve bağımlılıkları kurar. Notebook'ları 01 → 02 → 03 sırasıyla çalıştırın.
Tüm eğitimler CPU'da yapılır (`torch.set_num_threads(1)`, deterministik algoritmalar), toplam süre birkaç dakikadır.

## Klasör yapısı
| Yol | İçerik |
|---|---|
| `01_eda.ipynb` | Veri doğrulama, kopya kontrolü, stratified split, yalnızca development üzerinde EDA |
| `02_nested_cv.ipynb` | Fold-local ön işleme, manuel ve otomatik arama, nested CV, nihai seçim, 3 seed ile eğitim, kilitli tahminler |
| `03_final_test.ipynb` | Kilitli tahminlerle test metrikleri, bootstrap %95 CI, seed kararlılığı, hata analizi, ROC/PR grafikleri |
| `src/` | `config` (seed/sabitler), `data`, `preprocessing`, `model` (MLP + eğitim), `search` (nested CV), `metrics`, `make_configs` |
| `configs/` | `training.json`, `search_space.json`, `manual_configs.json`, `random_configs.json`, `hypotheses.md` |
| `results/` | `dataset_split.csv`, `manual_search_results.csv`, `random_search_results.csv`, `outer_cv_results.csv`, `final_hyperparameters.json`, `test_predictions_locked.csv`, `test_metrics.csv`, `seed_stability_results.csv`, eğitim logları |
| `figures/` | Raporda kullanılan tüm grafikler (kodla üretilmiştir) |
| `report.pdf` | Bilimsel teknik rapor |

## Zaman sıralı deney kaydı (commit sırası)
1. Kod iskeleti, sabitlenmiş adaylar (`configs/`) ve hipotezler. Nested CV'den önce commit edilir.
2. EDA çıktıları ve `dataset_split.csv`.
3. Nested CV sonuçları, nihai konfigürasyon ve **etiketsiz** `test_predictions_locked.csv` (ayrı commit).
4. Test metrikleri, bootstrap CI, grafikler ve rapor.

Test etiketleri kamuya açık bir veri setinde bulunduğu için bu sıralama kör test garantisi vermez. Yalnızca zaman sıralı bir deney kaydıdır.
