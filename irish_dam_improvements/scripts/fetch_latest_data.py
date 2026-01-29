#!/usr/bin/env python
# scripts/fetch_latest_data.py
"""
One-command script to fetch or generate latest Irish DAM data.

Usage:
    python scripts/fetch_latest_data.py           # Try all sources
    python scripts/fetch_latest_data.py --entsoe  # ENTSO-E only
    python scripts/fetch_latest_data.py --synthetic  # Generate synthetic
    python scripts/fetch_latest_data.py --days 90    # Custom days
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_generator import generate_for_dashboard


def try_entsoe(days: int) -> bool:
    """Try fetching from ENTSO-E."""
    print("\n📡 Trying ENTSO-E Transparency Platform...")
    
    try:
        from src.data.entsoe_fixed import EntsoeClientFixed, check_entsoe_connection
        
        status = check_entsoe_connection()
        
        if not status["token_set"]:
            print("  ❌ ENTSOE_TOKEN not set in environment")
            print("  💡 Get a token at: https://transparency.entsoe.eu/")
            return False
        
        if not status["connection_ok"]:
            print(f"  ❌ Connection failed: {status.get('error', 'Unknown error')}")
            return False
        
        if not status["data_available"]:
            print("  ❌ API responded but no data for Ireland/SEM")
            return False
        
        print("  ✅ ENTSO-E connection OK, fetching data...")
        
        client = EntsoeClientFixed()
        df = client.fetch_recent_chunked(days=days)
        
        if df.empty:
            print("  ❌ No data returned")
            return False
        
        # Save to expected locations
        output_dir = PROJECT_ROOT / "data"
        raw_dir = output_dir / "raw"
        processed_dir = output_dir / "processed"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw prices
        df.to_parquet(raw_dir / "dam_prices_entsoe.parquet")
        
        # Build features and save training data
        from src.features.build_features import build_feature_table
        from src.features.targets import make_day_ahead_target
        
        # Convert to local time for feature building
        prices = df["dam_eur_mwh"].copy()
        prices.index = prices.index.tz_convert("Europe/Dublin").tz_localize(None)
        
        # Minimal features
        X = build_feature_table(
            prices,
            prices.copy(),  # Use prices as load proxy
            None,
            None
        )
        y = make_day_ahead_target(prices)
        
        train_df = X.copy()
        train_df["target"] = y
        train_df = train_df.dropna(subset=["target"])
        
        train_df.to_parquet(processed_dir / "train.parquet")
        
        print(f"  ✅ Saved {len(df)} price points")
        print(f"  ✅ Date range: {df.index.min()} to {df.index.max()}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  💡 Run: pip install entsoe-py python-dotenv")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def try_semopx(days: int) -> bool:
    """Try fetching from SEMOpx."""
    print("\n📡 Trying SEMOpx API...")
    
    try:
        from src.data.semopx_api_v2 import SEMOpxClient
        
        client = SEMOpxClient()
        df = client.fetch_recent(days=days)
        
        if df.empty:
            print("  ❌ No data returned from SEMOpx")
            print("  💡 The SEMOpx API structure may have changed")
            print("  💡 Try downloading lookback files manually from:")
            print("     https://www.semopx.com/market-data/document-library")
            return False
        
        # Save data
        output_dir = PROJECT_ROOT / "data"
        raw_dir = output_dir / "raw"
        processed_dir = output_dir / "processed"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(raw_dir / "dam_prices_semopx.parquet", index=False)
        
        print(f"  ✅ Saved {len(df)} price points")
        print(f"  ✅ Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def use_synthetic(days: int) -> bool:
    """Generate synthetic data."""
    print("\n🎲 Generating synthetic data...")
    
    try:
        generate_for_dashboard(days=days, output_dir=str(PROJECT_ROOT / "data"))
        print("  ✅ Synthetic data generated successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch or generate Irish DAM data")
    parser.add_argument("--entsoe", action="store_true", help="Use ENTSO-E only")
    parser.add_argument("--semopx", action="store_true", help="Use SEMOpx only")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--days", type=int, default=90, help="Days of data to fetch/generate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Irish DAM Data Fetcher")
    print("=" * 60)
    print(f"Target: {args.days} days of data")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    success = False
    
    if args.synthetic:
        success = use_synthetic(args.days)
    elif args.entsoe:
        success = try_entsoe(args.days)
    elif args.semopx:
        success = try_semopx(args.days)
    else:
        # Try all sources in order
        print("\nTrying data sources in order of preference...")
        
        # 1. Try SEMOpx first (official source)
        success = try_semopx(args.days)
        
        # 2. Fall back to ENTSO-E
        if not success:
            success = try_entsoe(args.days)
        
        # 3. Fall back to synthetic
        if not success:
            print("\n⚠️ All external sources failed, using synthetic data")
            success = use_synthetic(args.days)
    
    print("\n" + "=" * 60)
    
    if success:
        print("✅ SUCCESS - Data ready!")
        print(f"\nNext steps:")
        print(f"  1. Run the dashboard: streamlit run src/dashboard/app_optimized.py")
        print(f"  2. Or the original: streamlit run src/dashboard/app.py")
    else:
        print("❌ FAILED - Could not fetch or generate data")
        print("\nTroubleshooting:")
        print("  - For ENTSO-E: Ensure ENTSOE_TOKEN is set in .env")
        print("  - For SEMOpx: Download lookback files manually")
        print("  - For synthetic: Check Python dependencies")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
