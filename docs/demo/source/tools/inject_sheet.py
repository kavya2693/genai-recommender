"""Inline the exported outreach plan into compositions/value.html (renders cannot fetch files).

Usage: python3 tools/inject_sheet.py assets/outreach_plan.json compositions/value.html --rows 0,1,2,9,15,66
(--rows are 0-based positions in the plan's row list, shown in that order as HCP-<position + 1>)
"""

import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("html")
ap.add_argument("--rows", required=True, help="plan row positions shown on the sheet, in order")
a = ap.parse_args()

with open(a.plan) as fh:
    d = json.load(fh)
picks = [int(v) for v in a.rows.split(",")]
ROW_KEYS = ("specialty", "title", "channel", "p_chosen", "snippet_ids", "grounded")
slim = {
    "region": d["region"],
    "prescribers": d["prescribers"],
    "rep_share_historical": d["rep_share_historical"],
    "rep_budget": d["rep_budget"],
    "channels": d["channels"],
    "grounded": d["grounded"],
    "rows": [{"pos": i} | {k: d["rows"][i][k] for k in ROW_KEYS} for i in picks],
}
with open(a.html) as fh:
    html = fh.read()
block = "/* SHEET:BEGIN */ const SHEET = " + json.dumps(slim, separators=(",", ":")) + "; /* SHEET:END */"
html, n = re.subn(r"/\* SHEET:BEGIN \*/.*?/\* SHEET:END \*/", lambda _: block, html, flags=re.S)
assert n == 1, "marker not found"
with open(a.html, "w") as fh:
    fh.write(html)
print("injected", len(slim["rows"]), "rows")
