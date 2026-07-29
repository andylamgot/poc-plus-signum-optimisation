"""
=================================================================
  Signum Blockchain - Stratified Random Block Sampling Scraper
=================================================================
  Pulls ~417 random blocks from each of the past 12 calendar months
  using stratified random sampling (total ~5,000 blocks).

  Outputs: CSV and JSON, both with a 'month' column added.

  Sampling method: Stratified Random Sampling
  ----------------------------------------------------------------
  - Population:    All blocks mined in each calendar month over
                    the past 12 months of the Signum blockchain.
  - Strata:        Calendar months (12 strata, one per month).
  - Sample size:   ~417 blocks per stratum (proportional
                    allocation, ~5,000 total).
  - Selection:     Simple random sampling without replacement
                    within each stratum.
  - Reproducibility: Fixed random seed (seed=42) ensures
                    identical results on re-runs.
  - Reference:
        Creswell and Creswell (2018). Research Design: Qualitative,
        Quantitative, and Mixed Methods Approaches, 5th ed. SAGE.
        Trochim (2006). Stratified Random Sampling.
        Research Methods Knowledge Base.

  Usage:
    pip install aiohttp
    python signum_block_scraper.py
    python signum_block_scraper.py --count 300          # 300 per month
    python signum_block_scraper.py --months 6           # last 6 months
    python signum_block_scraper.py --format json        # JSON only
    python signum_block_scraper.py --seed 123           # custom seed
    python signum_block_scraper.py --node https://us.signum.network/burst
    python signum_block_scraper.py --output my_data      # custom filename prefix

  Requirements:
    pip install aiohttp

  Author:  [Your Name]
  Thesis:  [LJMU Master Thesis Title]
  Date:    2026
=================================================================
"""

import argparse
import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime, timezone, timedelta
from calendar import monthrange

import aiohttp


# ==================== CONFIGURATION ====================

DEFAULT_NODE = "https://europe.signum.network/burst"
BATCH_SIZE = 100           # Max blocks per API request (API cap)
CONCURRENCY = 20           # Parallel HTTP connections
DEFAULT_PER_MONTH = 417    # ~417 x 12 = ~5,002 total blocks
DEFAULT_MONTHS_BACK = 12
SIGNUM_EPOCH_UNIX = 1407731470   # 2014-08-11 04:31:10 UTC
BLOCK_TIME_SECS = 240           # ~4 minutes average block time


# ==================== HELPERS ====================


def ts2dt(signum_ts: int) -> datetime:
    """Convert Signum timestamp to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(SIGNUM_EPOCH_UNIX + int(signum_ts), tz=timezone.utc)


def ts2iso(signum_ts: int) -> str:
    """Convert Signum timestamp to ISO 8601 string."""
    return ts2dt(signum_ts).strftime("%Y-%m-%d %H:%M:%S")


def dt2ts(dt: datetime) -> int:
    """Convert a UTC datetime to a Signum timestamp."""
    return int(dt.timestamp()) - SIGNUM_EPOCH_UNIX


def month_key(signum_ts: int) -> str:
    """Return 'YYYY-MM' for a Signum timestamp."""
    return ts2dt(signum_ts).strftime("%Y-%m")


def nqt_to_signa(nqt_str: str) -> float:
    """Convert NQT (nano-SIGNA, 1e-8) to SIGNA."""
    return int(nqt_str) / 1e8


def generate_months(cutoff: datetime, tip_dt: datetime) -> list:
    """Return list of (month_start_dt, month_end_dt, 'YYYY-MM') tuples."""
    months = []
    cur = cutoff
    while cur <= tip_dt:
        _, last_day = monthrange(cur.year, cur.month)
        m_end = cur.replace(day=last_day, hour=23, minute=59, second=59)
        months.append((cur, m_end, cur.strftime("%Y-%m")))
        # Advance to next month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


# ==================== API FETCHER ====================


async def fetch_batch(session: aiohttp.ClientSession,
                      first_index: int, last_index: int) -> list:
    """
    Fetch one batch of blocks (up to 100) from the Signum node API.
    The API uses 0-based offset from the chain tip.
    """
    url = (f"{args.node if 'args' in dir() else DEFAULT_NODE}"
           f"?requestType=getBlocks&firstIndex={first_index}&lastIndex={last_index}")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("blocks", [])
    except Exception:
        return []


async def fetch_month_blocks(session: aiohttp.ClientSession,
                             node: str, tip_height: int, tip_ts: int,
                             month_start: datetime, month_end: datetime) -> list:
    """
    Fetch all blocks for one calendar month using concurrent batched
    requests.  Returns raw block dicts.
    """
    # Estimate height range for this month (with buffer)
    h_start = tip_height - int((tip_ts - dt2ts(month_start)) / BLOCK_TIME_SECS) - 200
    h_end = tip_height - int((tip_ts - dt2ts(month_end)) / BLOCK_TIME_SECS) + 200
    h_start = max(0, h_start)
    h_end = min(tip_height, h_end)

    total_blocks = h_end - h_start + 1
    num_batches = (total_blocks + BATCH_SIZE - 1) // BATCH_SIZE
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = [[] for _ in range(num_batches)]

    async def _fetch_one(i: int):
        offset_start = i * BATCH_SIZE
        offset_end = min(offset_start + BATCH_SIZE - 1, total_blocks - 1)
        # Convert absolute height to API offset-from-tip
        api_first = tip_height - (h_start + offset_end)
        api_last = tip_height - (h_start + offset_start)
        async with semaphore:
            url = (f"{node}?requestType=getBlocks&firstIndex={api_first}"
                   f"&lastIndex={api_last}")
            try:
                async with session.get(url,
                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        results[i] = (await resp.json()).get("blocks", [])
            except Exception:
                pass

    await asyncio.gather(*[_fetch_one(i) for i in range(num_batches)])

    # Flatten results
    blocks = []
    for r in results:
        blocks.extend(r)
    return blocks


# ==================== EXPORTERS ====================

CSV_COLUMNS = [
    "month", "height", "datetime_utc", "generator_RS", "generator_public_key",
    "num_transactions", "total_amount_signa", "total_fee_signa",
    "fee_cashback_signa", "fee_burnt_signa", "block_reward_signa",
    "payload_length_bytes", "base_target", "avg_commitment_signa",
    "cumulative_difficulty", "scoop_num", "version", "num_tx_ids",
    "block_id", "previous_block_id", "previous_block_hash",
    "payload_hash", "generation_signature", "nonce", "block_signature",
]


def block_to_csv_row(b: dict) -> dict:
    """Flatten a block dict into a CSV-friendly row."""
    return {
        "month":                b["month"],
        "height":               b["height"],
        "datetime_utc":         ts2iso(b["timestamp"]),
        "generator_RS":         b["generatorRS"],
        "generator_public_key": b["generatorPublicKey"],
        "num_transactions":     b["numberOfTransactions"],
        "total_amount_signa":   nqt_to_signa(b["totalAmountNQT"]),
        "total_fee_signa":      nqt_to_signa(b["totalFeeNQT"]),
        "fee_cashback_signa":   nqt_to_signa(b.get("totalFeeCashBackNQT", "0")),
        "fee_burnt_signa":      nqt_to_signa(b.get("totalFeeBurntNQT", "0")),
        "block_reward_signa":   nqt_to_signa(b["blockRewardNQT"]),
        "payload_length_bytes": b["payloadLength"],
        "base_target":          b["baseTarget"],
        "avg_commitment_signa": nqt_to_signa(b.get("averageCommitmentNQT", "0")),
        "cumulative_difficulty":b["cumulativeDifficulty"],
        "scoop_num":            b["scoopNum"],
        "version":              b["version"],
        "num_tx_ids":           len(b.get("transactions", [])),
        "block_id":             b["block"],
        "previous_block_id":    b.get("previousBlock", ""),
        "previous_block_hash":  b.get("previousBlockHash", ""),
        "payload_hash":         b.get("payloadHash", ""),
        "generation_signature": b.get("generationSignature", ""),
        "nonce":                b["nonce"],
        "block_signature":      b.get("blockSignature", ""),
    }


def save_csv(blocks: list, path: str):
    """Write sampled blocks to a flat CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for b in blocks:
            writer.writerow(block_to_csv_row(b))


def save_json(blocks: list, path: str):
    """Write full block objects (with 'month' field) to JSON."""
    with open(path, "w") as f:
        json.dump(blocks, f, indent=2)


# ==================== MAIN ====================


async def run(node: str, per_month: int, months_back: int,
              fmt: str, output: str, seed: int):
    """Main sampling pipeline."""
    t0 = time.time()
    rng = random.Random(seed)

    print("=" * 58)
    print("  Signum Stratified Random Block Sampling")
    print("=" * 58)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Step 1: Get chain tip
        print("\n[1/3] Getting chain tip...")
        tip = await fetch_month_blocks.__wrapped__ if False else []
        # Get tip block directly
        try:
            async with session.get(
                f"{node}?requestType=getBlocks&firstIndex=0&lastIndex=0",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                tip_data = (await resp.json()).get("blocks", [])
        except Exception:
            tip_data = []

        if not tip_data:
            print("  ERROR: Could not reach node API")
            return

        tip_height = tip_data[0]["height"]
        tip_ts = tip_data[0]["timestamp"]
        tip_dt = ts2dt(tip_ts)
        print(f"  Tip: #{tip_height:,} ({tip_dt.strftime('%Y-%m-%d %H:%M UTC')})")

        # Step 2: Calculate month boundaries and sample
        cutoff = tip_dt - timedelta(days=months_back * 30.44)
        cutoff = cutoff.replace(day=1, hour=0, minute=0, second=0)
        months = generate_months(cutoff, tip_dt)

        print(f"\n[2/3] Sampling {per_month} blocks per month...")
        print(f"  Range: {months[0][2]} to {months[-1][2]} ({len(months)} months)")
        print(f"  {'Month':>10} {'Pulled':>8} {'Filtered':>9} {'Sampled':>8}")
        print("-" * 42)

        all_sampled = []
        total_pulled = 0

        for m_start, m_end, m_key in months:
            # Fetch all blocks in this month's estimated height range
            blocks = await fetch_month_blocks(
                session, node, tip_height, tip_ts, m_start, m_end)
            total_pulled += len(blocks)

            # Filter to actual calendar month
            ts_lo = dt2ts(m_start)
            ts_hi = dt2ts(m_end) + 86400  # +1 day buffer
            month_blocks = [b for b in blocks if ts_lo <= b["timestamp"] <= ts_hi]

            # Random sample without replacement
            n = min(per_month, len(month_blocks))
            if n == 0:
                print(f"  {m_key:>10} {len(blocks):>8,} {len(month_blocks):>9,} {n:>8}  (skipped)")
                continue
            chosen = rng.sample(month_blocks, n)
            for b in chosen:
                b["month"] = m_key
            all_sampled.extend(chosen)
            print(f"  {m_key:>10} {len(blocks):>8,} {len(month_blocks):>9,} {n:>8}")

    # Step 3: Save outputs
    print(f"\n[3/3] Saving outputs...")
    all_sampled.sort(key=lambda b: (b["month"], b["height"]))

    csv_path = f"{output}.csv"
    json_path = f"{output}.json"
    save_csv(all_sampled, csv_path)
    save_json(all_sampled, json_path)

    elapsed = time.time() - t0
    csv_mb = os.path.getsize(csv_path) / 1e6
    json_mb = os.path.getsize(json_path) / 1e6
    months_covered = sorted(set(b["month"] for b in all_sampled))

    print(f"\n{'=' * 58}")
    print(f"  DONE in {elapsed:.0f}s")
    print(f"  Total pulled (all months): {total_pulled:,}")
    print(f"  Total sampled: {len(all_sampled)}")
    print(f"  Months: {months_covered[0]} to {months_covered[-1]} ({len(months_covered)})")
    print(f"  Unique miners: {len(set(b['generatorRS'] for b in all_sampled))}")
    print(f"  CSV:  {csv_path} ({csv_mb:.1f} MB)")
    print(f"  JSON: {json_path} ({json_mb:.1f} MB)")
    print(f"{'=' * 58}")


def main():
    global args
    parser = argparse.ArgumentParser(
        description="Stratified random sampling of Signum blocks by month.")
    parser.add_argument("--count", type=int, default=DEFAULT_PER_MONTH,
        help=f"Blocks to sample per month (default: {DEFAULT_PER_MONTH})")
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS_BACK,
        help=f"Number of months to cover (default: {DEFAULT_MONTHS_BACK})")
    parser.add_argument("--format", choices=["both", "csv", "json"], default="both",
        help="Output format (default: both)")
    parser.add_argument("--node", default=DEFAULT_NODE,
        help=f"Node API URL (default: {DEFAULT_NODE})")
    parser.add_argument("--output", type=str, default="signum_stratified_blocks",
        help="Output filename prefix (default: signum_stratified_blocks)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    asyncio.run(run(args.node, args.count, args.months,
                    args.format, args.output, args.seed))


if __name__ == "__main__":
    main()
