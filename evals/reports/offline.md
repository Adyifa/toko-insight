# Offline eval report

Dataset: `python -m tokoinsight generate` (seed 7, 2023-01-01 to 2026-10-31).
Reproduce: `python -m tokoinsight eval --suite analytics` and `--suite forecast`.

## analytics suite
**34/34 checks passed**

| check | result | detail |
|---|---|---|
| top product by revenue | PASS | got Beras Premium 5kg, expected Beras Premium 5kg |
| top product by units | PASS | got Mi Instan Goreng, expected Mi Instan Goreng |
| fastest growing product | PASS | got Kopi Susu Botol 1L, expected one of ['Kopi Susu Botol 1L'] |
| decliner found: Voucher Pulsa Fisik 10rb | PASS | top 3 declining: ['Voucher Pulsa Fisik 10rb', 'Sabun Batang', 'Pulpen Gel'] |
| decliner found: Sabun Batang | PASS | top 3 declining: ['Voucher Pulsa Fisik 10rb', 'Sabun Batang', 'Pulpen Gel'] |
| delisted found: Wafer Stik Keju | PASS | status: stopped selling (last sale 2025-08-31) |
| new product found: Mi Pedas Korea | PASS | status: new (not sold for the whole previous period) |
| seasonal (ramadan): Tepung Terigu 1kg | PASS | detected: ramadan |
| seasonal (ramadan): Santan Kemasan 200ml | PASS | detected: ramadan |
| seasonal (ramadan): Sirup Melon 460ml | PASS | detected: ramadan |
| seasonal (ramadan): Biskuit Kaleng Assorted | PASS | detected: ramadan |
| seasonal (ramadan): Kurma 500g | PASS | detected: ramadan |
| seasonal (year_end): Minuman Soda 1.5L | PASS | detected: year_end |
| seasonal (year_end): Kacang Kulit 250g | PASS | detected: year_end |
| seasonal (dry): Minuman Isotonik 500ml | PASS | detected: dry |
| seasonal (dry): Es Krim Cup | PASS | detected: dry |
| seasonal (rainy): Obat Masuk Angin Sachet | PASS | detected: rainy |
| seasonal (rainy): Jas Hujan Plastik | PASS | detected: rainy |
| seasonal (rainy): Payung Lipat | PASS | detected: rainy |
| seasonal (school): Buku Tulis 38 Lembar (10) | PASS | detected: school |
| seasonal (school): Pulpen Gel | PASS | detected: school |
| bought together: Roti Tawar + Selai Cokelat | PASS | top 5 pairs by lift: [('Popok Bayi M 20s', 'Tisu Basah 50s', 16.71), ('Pasta Gigi 190g', 'Sikat Gigi', 15.84), ('Roti Ta |
| bought together: Mi Instan Goreng + Telur Ayam 1kg | PASS | top 5 pairs by lift: [('Popok Bayi M 20s', 'Tisu Basah 50s', 16.71), ('Pasta Gigi 190g', 'Sikat Gigi', 15.84), ('Roti Ta |
| bought together: Pasta Gigi 190g + Sikat Gigi | PASS | top 5 pairs by lift: [('Popok Bayi M 20s', 'Tisu Basah 50s', 16.71), ('Pasta Gigi 190g', 'Sikat Gigi', 15.84), ('Roti Ta |
| bought together: Gula Pasir 1kg + Kopi Sachet Renceng | PASS | top 5 pairs by lift: [('Popok Bayi M 20s', 'Tisu Basah 50s', 16.71), ('Pasta Gigi 190g', 'Sikat Gigi', 15.84), ('Roti Ta |
| bought together: Popok Bayi M 20s + Tisu Basah 50s | PASS | top 5 pairs by lift: [('Popok Bayi M 20s', 'Tisu Basah 50s', 16.71), ('Pasta Gigi 190g', 'Sikat Gigi', 15.84), ('Roti Ta |
| rarely bought: Payung Lipat | PASS | 5 least frequent: ['Payung Lipat', 'Lampu LED 9W', 'Baterai AA (2)', 'Kurma 500g', 'Biskuit Kaleng Assorted'] |
| rarely bought: Lampu LED 9W | PASS | 5 least frequent: ['Payung Lipat', 'Lampu LED 9W', 'Baterai AA (2)', 'Kurma 500g', 'Biskuit Kaleng Assorted'] |
| stock less: Sabun Batang | PASS | action: reduce / stop ordering |
| stock less: Voucher Pulsa Fisik 10rb | PASS | action: reduce / stop ordering |
| stock more: Kopi Susu Botol 1L | PASS | action: order now |
| stock more: Mi Pedas Korea | PASS | action: order now |
| forecast peaks in the Ramadan month | PASS | Kurma 500g: peak 2027-02, Ramadan month 2027-02 |
| QRIS share rising every year | PASS | 2023: 14%, 2024: 26%, 2025: 37%, 2026: 48% |

## forecast suite
Rolling-origin backtest: 4 origins x 3 months (last 12 months), 44 products. hw_lebaran beats seasonal-naive on 28/44 products.

| series | hw_lebaran | holt_winters | seasonal_naive | naive |
|---|---|---|---|---|
| all products (sum of product errors) (WAPE) | 5.5% | 6.0% | 7.6% | 11.8% |
| Ramadan products (WAPE) | 8.9% | 18.3% | 17.0% | 19.6% |
| store total (one series) (WAPE) | 2.2% | 2.6% | 4.0% | 7.8% |
