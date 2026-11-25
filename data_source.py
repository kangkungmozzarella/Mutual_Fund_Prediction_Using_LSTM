"""Ambil data harga harian dari Yahoo Finance, dengan cache dan fallback CSV."""
import threading
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

FALLBACK_DIR = Path(__file__).parent / "data"
CACHE_TTL = 15 * 60  # detik
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

_cache: dict = {}
_lock = threading.Lock()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rapikan kolom (yfinance kadang memakai MultiIndex), tanggal, dan duplikat."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip().title())
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _from_yahoo(ticker: str, start: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, interval="1d", auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance tidak mengembalikan data untuk {ticker}")
    return _clean(raw)


def _from_csv(ticker: str, start: str) -> pd.DataFrame:
    path = FALLBACK_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Tidak ada data cadangan untuk {ticker}")
    df = _clean(pd.read_csv(path, index_col="Date", parse_dates=True))
    return df[df.index >= pd.Timestamp(start)]


def get_history(ticker: str, start: str = "2022-03-01", force: bool = False) -> dict:
    """Kembalikan {'df', 'source', 'error', 'fetched_at'}.

    source = 'yahoo' bila berhasil dari Yahoo Finance, 'csv' bila memakai cadangan.
    """
    key = (ticker.upper(), start)
    with _lock:
        hit = _cache.get(key)
    if hit and not force and time.time() - hit["fetched_at"] < CACHE_TTL:
        return hit

    try:
        out = {"df": _from_yahoo(ticker, start), "source": "yahoo", "error": None}
    except Exception as exc:  # jaringan putus, ticker salah, diblokir, dll.
        out = {"df": _from_csv(ticker, start), "source": "csv", "error": str(exc)}
    out["fetched_at"] = time.time()
    with _lock:
        _cache[key] = out
    return out


def get_quote(ticker: str) -> dict:
    """Harga terakhir (intraday bila bursa buka) dari Yahoo Finance."""
    try:
        fi = yf.Ticker(ticker).fast_info
        price = float(fi.last_price)
        prev = float(fi.previous_close)
        return {
            "ticker": ticker, "price": price, "prev_close": prev,
            "change_pct": (price / prev - 1) * 100 if prev else None,
            "currency": getattr(fi, "currency", None),
            "source": "yahoo", "at": time.time(),
        }
    except Exception as exc:
        df = get_history(ticker)["df"]
        price, prev = float(df.Close.iloc[-1]), float(df.Close.iloc[-2])
        return {
            "ticker": ticker, "price": price, "prev_close": prev,
            "change_pct": (price / prev - 1) * 100, "currency": "IDR",
            "source": "csv", "error": str(exc), "at": time.time(),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
        }
