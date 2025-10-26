#!/usr/bin/env python3
# monitor_pubchem_db.py
#
# Live-progress watcher for /mnt/sdc1/chemicals/pubchem/Extras/pubchem_scan.sqlite
# - Reads in WAL-friendly, read-only mode
# - Filters by ruleset/alpha_pi (so you can reuse the DB for other runs)
# - Prints one concise line every --interval seconds with totals, rate, % and ETA
#
# Usage example:
#   python3 monitor_pubchem_db.py \
#     --db /mnt/sdc1/chemicals/pubchem/Extras/pubchem_scan.sqlite \
#     --ruleset RS3 --alpha-pi 1.15 \
#     --expected 120000000 --interval 10
#
import argparse, sqlite3, time, math, sys

def connect_ro(db_path: str) -> sqlite3.Connection:
    # Read-only, WAL-friendly
    uri = f"file:{db_path}?mode=ro&cache=shared"
    con = sqlite3.connect(uri, uri=True, timeout=5.0)
    con.execute("PRAGMA query_only=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA read_uncommitted=ON;")
    return con

Q_TOTAL = """
SELECT COUNT(*) FROM scan_results
WHERE ruleset=? AND ABS(alpha_pi - ?) < 1e-12;
"""

Q_COUNTS = """
SELECT status, COUNT(*) FROM scan_results
WHERE ruleset=? AND ABS(alpha_pi - ?) < 1e-12
GROUP BY status;
"""

def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0: return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    ap = argparse.ArgumentParser(description="Live monitor for pubchem_scan.sqlite")
    ap.add_argument("--db", required=True, help="Path to SQLite DB (pubchem_scan.sqlite)")
    ap.add_argument("--ruleset", default="RS3", choices=["RS1","RS2","RS3"])
    ap.add_argument("--alpha-pi", type=float, default=1.15)
    ap.add_argument("--expected", type=int, default=0, help="Expected total rows for this run (for % & ETA)")
    ap.add_argument("--interval", type=float, default=10.0, help="Seconds between refresh")
    ap.add_argument("--once", action="store_true", help="Print once and exit")
    args = ap.parse_args()

    con = connect_ro(args.db)

    last_t = time.time()
    last_n = None
    start_t = last_t
    start_n = None

    # Single pass function
    def snapshot():
        cur = con.cursor()
        total = cur.execute(Q_TOTAL, (args.ruleset, args.alpha_pi)).fetchone()[0]
        by = dict(cur.execute(Q_COUNTS, (args.ruleset, args.alpha_pi)).fetchall())
        cur.close()
        return total, by

    def print_line(total, by, rate, eta_s):
        exp = args.expected
        pct = (100.0 * total / exp) if exp > 0 else 0.0
        statuses = ("valid","invalid","parse_fail","kekulize_fail","no_heavy","salt_accept","unsupported","malformed")
        pieces = [f"total={total}"]
        for k in statuses:
            if k in by: pieces.append(f"{k}={by[k]}")
        if exp > 0: pieces.append(f"{pct:5.2f}%")
        pieces.append(f"{rate:.1f}/s")
        pieces.append(f"ETA {fmt_eta(eta_s)}")
        print(" | ".join(pieces), flush=True)

    # initial snapshot
    total, by = snapshot()
    start_n = total
    print_line(total, by, 0.0, float("inf"))
    if args.once:
        return

    while True:
        time.sleep(args.interval)
        try:
            total2, by2 = snapshot()
        except sqlite3.OperationalError:
            # transient while writer is committing—skip a beat
            continue
        now = time.time()
        dt = now - last_t
        dn = (total2 - (last_n if last_n is not None else total))
        inst_rate = (dn / dt) if dt > 0 else 0.0

        # ETA based on overall average rate since start (smoother)
        elapsed = now - start_t
        overall_rate = (total2 - start_n) / elapsed if elapsed > 0 else 0.0
        if args.expected > 0 and overall_rate > 0:
            remaining = max(0, args.expected - total2)
            eta_s = remaining / overall_rate
        else:
            eta_s = float("inf")

        print_line(total2, by2, overall_rate if overall_rate>0 else inst_rate, eta_s)

        last_t, last_n = now, total2

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)