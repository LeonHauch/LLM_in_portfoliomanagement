# (You write this code in VS Code inside src/data/download_prices.py)

import yfinance as yf
import os
import pandas as pd
from itertools import product

# CAC 40 tickers (represent French blue-chip stocks)
TICKERS = [
    "AIR.PA",  # Airbus
    "OR.PA",   # L'Oréal
    "MC.PA",   # LVMH
    "BNP.PA",  # BNP Paribas
    "DG.PA",   # Vinci
    "EN.PA",   # Bouygues
    "SAN.PA",  # Sanofi
    "KER.PA",  # Kering
    "SU.PA",   # Schneider Electric
    "ACA.PA",  # Crédit Agricole
    "VIE.PA",  # Veolia
    "VIV.PA",  # Vivendi
    "CA.PA",   # Carrefour
    "GLE.PA",  # Société Générale
    "DSY.PA",  # Dassault Systèmes
    "RI.PA",   # Pernod Ricard
    "ATO.PA",  # Atos
    "ENGI.PA", # Engie
    "ML.PA",   # Michelin
    "CAP.PA",  # Capgemini
    "EL.PA",   # EssilorLuxottica
    "RNO.PA",  # Renault
    "FR.PA",   # Valeo
    "SW.PA",   # Sodexo
    "AI.PA",   # Air Liquide
    "HO.PA",   # Thales
    "WLN.PA",  # Worldline
]

def download_and_store(tickers, start="2018-01-01", end="2024-12-31", folder="raw"):
    os.makedirs(folder, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)

        if not df.empty:
            # Ensure df.columns is a flat Index before applying MultiIndex
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product([df.columns, [ticker]], names=["Price", "Ticker"])

            df.to_csv(f"{folder}/{ticker}.csv")
        else:
            print(f"Warning: {ticker} returned empty data.")

    print("Download complete.")

if __name__ == "__main__":
    download_and_store(TICKERS)
