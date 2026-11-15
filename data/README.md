# Data

This directory is the single location where all of the book's
experiment data lives. Every experiment script locates this directory via its own `paths.py`, so it runs from any working directory -- there's no longer any need to
`cd` into the script's own directory.

```python
from paths import data_file, stock_file

data = pd.read_csv(data_file("data_ml_2017_11.csv"), index_col="stock_id")
df   = pd.read_csv(stock_file("688256.SH.csv"), index_col="trade_date")
```

---

## File Manifest

| File | Size | Content | Used by |
|---|---|---|---|
| `data_ml.csv` | 133 MB | An international monthly stock panel, 283,380 rows x 99 columns: `stock_id`, `date`, 93 features, 4 forward-return labels (`R1M_Usd`/`R3M_Usd`/`R6M_Usd`/`R12M_Usd`) | 5.5, 7.5 |
| `data_ml.RData` | 22 MB | The same dataset, in R's native format | 3.6, 3.7, 4.6 |
| `data_ml_2017_11.csv` | 524 KB | A single 2017-11-30 cross section extracted from `data_ml.csv`, 1,148 stocks x 94 columns. A **derived file**, regenerable via `chap5/5.5/get_data.py` | 5.5, 6.4 |
| `analysts_stock_price_synchronicity.dta` | 2.0 MB | An analyst-coverage / stock-price-synchronicity panel (Stata format) | 4.7 |
| `CSI100_stock_data_with_factors.pkl` | 37 MB | A CSI 100 daily panel, 437,656 rows x 12 columns, 190 stocks, 2015-01-23 through 2025-08-15. Columns include `open`/`high`/`low`/`close`/`vol`/`ret`/`momentum_14`/`volatility_14`/`RSI_14`/`industry` | 8.5, 8.6, 11.6 |
| `CSI100_Top30.pkl` | 30 MB | A daily panel for 30 stocks from the CSI 100, 120,472 rows x 34 columns (a richer feature set) | 10.6 |
| `stocks/000001SZ.xlsx` | 173 KB | Daily quotes for Ping An Bank | 5.6, 6.5 |
| `stocks/688256.SH.csv` | 297 KB | Daily quotes for Cambricon, 1,222 rows x 23 columns (22 technical indicators + daily return) | 7.6 |
| `stocks/000538.SZ.csv` | 1.4 MB | Daily quotes for Yunnan Baiyao, same structure | 7.6 |
| `stocks/600111.SH.csv` | 1.4 MB | Daily quotes for Northern Rare Earth, same structure | 7.6 |



