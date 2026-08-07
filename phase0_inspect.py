"""Phase 0 — is this dataset scoreable?

Answers four things and nothing else. No method, no model, no building.

  1. Every column in both frames.
  2. Is there a laundering label?
  3. Class balance, and the base rate.
  4. Basic shape — accounts, banks, currencies, time span.

If there is no label, STOP. The whole plan depends on being able to score
from day one, which is the one thing KOMPOSOS-SEC could not do for a year.
"""
import os
import sys

import pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TX = os.path.join(DATA, "aml_transactions_1M.parquet")
ACC = os.path.join(DATA, "aml_accounts_1M.parquet")

LABEL_HINTS = ("launder", "label", "is_fraud", "fraud", "suspicious",
               "illicit", "sar", "flag", "target", "class")


def describe(path, title):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print("=" * 70)
    print(f"{title}")
    print(f"  {md.num_rows:,} rows   {md.num_row_groups} row groups   "
          f"{os.path.getsize(path)/1024/1024:.1f} MB")
    print("  columns:")
    for f in pf.schema_arrow:
        print(f"     {f.name:<28} {f.type}")
    return pf


tx = describe(TX, "TRANSACTIONS (edge frame)")
acc = describe(ACC, "ACCOUNTS (vertex frame)")
print()

tx_cols = [f.name for f in tx.schema_arrow]
acc_cols = [f.name for f in acc.schema_arrow]

hits = [c for c in tx_cols + acc_cols
        if any(h in c.lower().replace(" ", "_") for h in LABEL_HINTS)]

print("=" * 70)
print("LABEL CHECK")
if not hits:
    print("  NO column matched any label hint.")
    print(f"  hints searched: {', '.join(LABEL_HINTS)}")
    print()
    print("  >>> STOP. Without ground truth this dataset cannot be scored,")
    print("  >>> and scoring from day one is the entire reason for this repo.")
    print("  >>> Get the original from github.com/IBM/AML-Data instead.")
    sys.exit(0)

print(f"  candidate label columns: {hits}")
print()

# Read the whole (small) table to count the label and the shape.
table = pq.read_table(TX)
for col in hits:
    if col not in tx_cols:
        continue
    vals = table.column(col).to_pylist()
    n = len(vals)
    distinct = {}
    for v in vals:
        distinct[v] = distinct.get(v, 0) + 1
    print(f"  {col}: {len(distinct)} distinct values")
    for v, c in sorted(distinct.items(), key=lambda kv: -kv[1])[:6]:
        print(f"     {str(v):<12} {c:>10,}   {c/n:.4%}")
    pos = sum(c for v, c in distinct.items()
              if str(v) not in ("0", "False", "false", "None", ""))
    print(f"  >>> BASE RATE: {pos:,} of {n:,} = {pos/n:.4%}")
    print(f"  >>> at this rate, a model predicting 'never' is "
          f"{1-pos/n:.2%} accurate and useless. Use precision@k.")
    print()

print("=" * 70)
print("SHAPE")
for col in tx_cols:
    low = col.lower()
    if any(k in low for k in ("bank", "currency", "format", "payment")):
        vals = table.column(col).to_pylist()
        u = sorted({str(v) for v in vals})
        print(f"  {col}: {len(u)} distinct" +
              (f"  {u[:8]}" if len(u) <= 20 else ""))
    if "time" in low or "date" in low or "step" in low:
        vals = table.column(col).to_pylist()
        print(f"  {col}: min={min(vals)}  max={max(vals)}")

# account-level shape
acct_cols_lower = {c.lower(): c for c in tx_cols}
src = next((tx_cols[i] for i, c in enumerate(tx_cols)
            if "from" in c.lower() or "source" in c.lower()
            or c.lower().startswith("src")), None)
dst = next((tx_cols[i] for i, c in enumerate(tx_cols)
            if "to" in c.lower() or "dest" in c.lower()
            or c.lower().startswith("dst")), None)
if src and dst:
    s = set(table.column(src).to_pylist())
    d = set(table.column(dst).to_pylist())
    print(f"  distinct {src}: {len(s):,}")
    print(f"  distinct {dst}: {len(d):,}")
    print(f"  union (accounts touched): {len(s | d):,}")
