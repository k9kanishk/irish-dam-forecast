#!/usr/bin/env python
from __future__ import annotations
import argparse
import pandas as pd
from dotenv import load_dotenv
from src.data.entsoe_api import Entsoe

if __name__ == '__main__':
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    ent = Entsoe()
    dam = ent.day_ahead_prices(args.start, args.end)
    dam.to_csv('data/raw/dam_prices_ie.csv', header=True)

    load_fc = ent.load_forecast(args.start, args.end)
    load_fc.to_csv('data/raw/load_forecast_ie.csv', header=True)

    ws = ent.wind_solar_forecast(args.start, args.end)
    ws.to_csv('data/raw/wind_solar_forecast_ie.csv')
