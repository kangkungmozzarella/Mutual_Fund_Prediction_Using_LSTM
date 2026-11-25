# Dashboard Prediksi Reksa Dana LSTM

Web dashboard bergaya Power BI untuk memprediksi harga reksa dana/ETF dengan LSTM.
Data harian diambil langsung dari Yahoo Finance sampai hari ini, dan semua variabel model
bisa diatur dari panel kiri lalu dilatih ulang langsung dari browser.

Dikembangkan dari skripsi [Mutual_Fund_Prediction_Using_LSTM](https://github.com/kangkungmozzarella/Mutual_Fund_Prediction_Using_LSTM).

## Fitur

- **Data Yahoo Finance** lewat `yfinance`, di-cache 15 menit. Tombol *Segarkan data* mengambil ulang.
- **Harga terbaru** di pojok kanan atas, diperbarui otomatis tiap 60 detik.
- **Variabel model**: fitur input, window, porsi data latih, periode RSI, jumlah lapis LSTM,
  unit, dropout, epoch, batch size, learning rate, early stopping, seed, dan horizon forecast.
- **Training di background** dengan progress per epoch dan kurva loss live.
- **Visual**: KPI, grafik harga + prediksi + forecast + rentang 95% (bisa di-zoom),
  kurva loss, MAPE per bulan, distribusi error, scatter aktual vs prediksi, tabel forecast.
- **Riwayat percobaan**: setiap training tersimpan di browser, jadi beberapa konfigurasi bisa dibandingkan.
  Klik baris untuk memuat hasil dan variabelnya.
- **Pembanding naive**: model dibandingkan dengan tebakan "harga besok = harga hari ini".
- **Fallback CSV**: kalau Yahoo Finance tidak bisa diakses, dashboard memakai `data/<TICKER>.csv`.

## Menjalankan

Butuh Python 3.10 atau lebih baru (diuji dengan Python 3.13 + TensorFlow 2.21).

```bash
cd lstm-fund-dashboard
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000. Saat pertama dibuka, model langsung dilatih dengan parameter default
(sekitar 30 detik di CPU).

Pengaturan opsional lewat environment variable:

| Variabel | Default | Keterangan |
|---|---|---|
| `DEFAULT_TICKER` | `XSSI.JK` | Ticker Yahoo Finance awal |
| `DEFAULT_START` | `2022-03-01` | Tanggal awal data |
| `PORT` | `5000` | Port server |

## Struktur

```
app.py           Server Flask: API + menyajikan dashboard
data_source.py   Ambil data Yahoo Finance, cache, fallback CSV, harga terbaru
model.py         Fitur (Average, RSI), LSTM, evaluasi, forecast rekursif
static/          index.html, style.css, app.js (ECharts)
data/            CSV cadangan
```

### API

| Endpoint | Keterangan |
|---|---|
| `GET /api/config` | Ticker, parameter default, batas nilai |
| `GET /api/history?ticker=&start=&refresh=1` | Data OHLCV harian |
| `GET /api/quote?ticker=` | Harga terakhir |
| `POST /api/train` | Mulai training. Body: `{ticker, start, params: {...}}` → `{job_id}` |
| `GET /api/jobs/<id>` | Status, progress epoch, dan hasil (metrik, prediksi uji, forecast) |

## Perbedaan dengan notebook skripsi

1. **Target = Close.** Di notebook, target diambil dari kolom pertama fitur (`High`), padahal dibaca sebagai Close.
2. **Scaler di-fit di data latih saja**, supaya informasi data uji tidak bocor ke model.
3. **Early stopping memakai 10% terakhir data latih** sebagai validasi, bukan data uji.
4. **Forecast rekursif mengisi semua fitur.** Di notebook, fitur selain target diisi 0 sehingga forecast
   turun lalu mendatar. Sekarang setiap prediksi dijadikan baris OHLC baru, lalu Average dan RSI dihitung ulang.
5. **Rentang perkiraan 95%** = ±1,96 × sebaran error data uji (MAD) × √hari ke depan.

## Catatan

- Data CSV bawaan berasal dari skripsi dan berisi baris akhir pekan yang nilainya diulang.
  Data dari Yahoo Finance hanya berisi hari bursa.
- Hanya satu training berjalan sekaligus. Request lain saat training berjalan akan diarahkan ke job yang sama.
- Ini alat analisis/pembelajaran, bukan saran investasi.
