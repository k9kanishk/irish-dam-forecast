#!/usr/bin/env python
from __future__ import annotations
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from zoneinfo import ZoneInfo

IE_ZONE = 'IE'
TZ = ZoneInfo('Europe/Dublin')

def main():
    load_dotenv()
    token = os.getenv('ENTSOE_TOKEN')
    if not token:
        print('FAIL: ENTSOE_TOKEN not set in environment (.env).', file=sys.stderr)
        sys.exit(2)
    try:
        client = EntsoePandasClient(api_key=token)
        start = pd.Timestamp.utcnow().tz_convert(TZ) - pd.Timedelta(days=2)
        end = pd.Timestamp.utcnow().tz_convert(TZ) - pd.Timedelta(days=1)
        s = client.query_day_ahead_prices(IE_ZONE, start, end)
        if len(s):
            print(f'OK: token works. Received {len(s)} hourly prices for IE between {start} and {end}.')
            sys.exit(0)
        else:
            print('WARN: request succeeded but returned empty series (check date range).')
            sys.exit(1)
    except Exception as e:
        print(f'FAIL: API call error: {e}', file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
