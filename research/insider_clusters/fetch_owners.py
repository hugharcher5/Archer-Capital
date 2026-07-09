"""
Phase 1 fetch: enrich the ~10.6k Form-4 filings (from mgmt_pit's existing
cache) that contain at least one open-market purchase (action=A, price>0)
with reporting-owner identity + 10b5-1 flag. Resumable (skips accns already
cached in cache/form4_owner/).
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R


def main():
    with open(R.OWN_CACHE / "universes.pkl", "rb") as f:
        d = pickle.load(f)
    all_ciks = sorted(d["all_ciks"])
    print(f"[Fetch] {len(all_ciks)} unique CIKs across 24 quarters")

    R.load_form4_basic_cache()

    with open(R.OWN_CACHE / "fetch_queue.pkl", "rb") as f:
        queue = pickle.load(f)
    print(f"[Fetch] Queue: {len(queue)} filings need owner-identity fetch")

    t0 = time.time()
    done = 0
    errors = 0
    for i, (cik, accn, doc) in enumerate(queue):
        result = R.parse_form4_owner_xml(cik, accn, doc)
        if result is None:
            errors += 1
        else:
            done += 1
        if (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(queue) - i - 1) / rate if rate > 0 else float("nan")
            print(f"  [{i+1}/{len(queue)}] done={done} errors={errors} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"[Fetch] COMPLETE: {done} fetched, {errors} errors, "
          f"{time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
