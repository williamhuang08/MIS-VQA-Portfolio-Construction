import pandas as pd
import numpy as np
import yfinance as yf
import time

def get_dow30_tickers() -> list[str]:
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    )
    for idx, table in enumerate(tables):
        cols = table.columns.astype(str).tolist()
        if any("Symbol" in c or "Ticker" in c for c in cols):
            sym_col = next(c for c in cols if "Symbol" in c or "Ticker" in c)
            return table[sym_col].astype(str).str.replace(".", "-", regex=False).tolist()
    

def compute_correlation_matrix(prices: pd.DataFrame, T: int = None) -> pd.DataFrame:
    log_prices = np.log(prices)
    log_returns = log_prices.diff().dropna()

    if T is not None:
        log_returns = log_returns.iloc[-T:]

    mean_returns = log_returns.mean(axis=0)  

    centered = log_returns.subtract(mean_returns, axis=1)

    cov_num = centered.T.dot(centered) 

    var = (centered ** 2).sum(axis=0)      
    denom = np.sqrt(np.outer(var, var))    

    C = cov_num / denom

    C = C.clip(-1.0, 1.0)

    return pd.DataFrame(C, index=prices.columns, columns=prices.columns)



def download_prices(tickers: list[str],
                    start: str,
                    end: str,
                    chunk_size: int = 100,
                    pause: float = 1.0) -> pd.DataFrame:

    all_data = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        df = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )["Close"]
        all_data.append(df)
        time.sleep(pause)  
    combined = pd.concat(all_data, axis=1)
    combined = combined.dropna(axis=1, how="all")
    return combined

if __name__ == "__main__":
    # Parameters
    YEARS         = 3
    BUSINESS_DAYS = int(252 * YEARS)  # ~756 days
    END_DATE      = pd.Timestamp.today().normalize()
    START_DATE    = END_DATE - pd.DateOffset(years=YEARS)

    tickers = get_dow30_tickers()
    print(f"Found {len(tickers)} DOW30 tickers.")

    print("Downloading price data (this may take a few minutes)…")
    prices = download_prices(
        tickers,
        start=START_DATE,
        end=END_DATE,
        chunk_size=30,
        pause=0.5,
    )
    print(f"Downloaded data for {prices.shape[1]} tickers over {prices.shape[0]} days.")

    print("Computing correlation matrix…")
    C = compute_correlation_matrix(prices, T=BUSINESS_DAYS)

    out_path = "dow30_correlation_matrix.csv"
    C.to_csv(out_path)
    print(f"Saved correlation matrix to {out_path}")
