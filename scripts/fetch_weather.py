#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml
from src.data.weather import fetch_hourly

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open('config.yaml'))
    lat = cfg['weather']['lat']; lon = cfg['weather']['lon']
    df = fetch_hourly(lat, lon, args.start, args.end)
    df.to_csv('data/raw/weather_hourly.csv')
