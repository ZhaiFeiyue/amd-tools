#!/usr/bin/env python3
"""Read rocprofv3 profile DB and print PMC counters."""
import sqlite3, sys
from collections import defaultdict

db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v11_profile_results.db"
db = sqlite3.connect(db_path)
cur = db.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
pmc = [t for t in tables if "pmc_event" in t][0]
disp = [t for t in tables if "kernel_dispatch" in t][0]
ksym = [t for t in tables if "kernel_symbol" in t][0]
ipmc = [t for t in tables if "info_pmc" in t][0]

pmc_names = {r[0]: r[1] for r in cur.execute("SELECT id, name FROM " + ipmc)}

rows = cur.execute(
    "SELECT d.id, k.display_name, d.workgroup_size_x, d.grid_size_x, "
    "k.arch_vgpr_count, k.accum_vgpr_count, k.sgpr_count, k.group_segment_size "
    "FROM " + disp + " d JOIN " + ksym + " k ON d.kernel_id=k.id ORDER BY d.id"
).fetchall()

print("Last 5 dispatches:")
for r in rows[-5:]:
    name = str(r[1])[:50]
    print("  dispatch=%d kernel=%s WG=%d Grid=%d vgpr=%d agpr=%d sgpr=%d lds=%d" %
          (r[0], name, r[2], r[3], r[4], r[5], r[6], r[7]))

gemm_id = None
for r in reversed(rows):
    nm = str(r[1]).lower()
    if "v11" in nm or "v10" in nm or "gemm" in nm:
        gemm_id = r[0]
        break

if gemm_id:
    counters = defaultdict(float)
    for r in cur.execute("SELECT pmc_id, value FROM " + pmc + " WHERE event_id=%d" % gemm_id):
        name = pmc_names.get(r[0], "unk_%d" % r[0])
        counters[name] += r[1] if r[1] else 0

    waves = counters.get("SQ_WAVES", 0)
    print("\nPMC Counters (dispatch %d):" % gemm_id)
    for n in sorted(counters):
        if counters[n] > 0:
            pw = " (%.1f/wave)" % (counters[n] / waves) if waves > 0 and n != "SQ_WAVES" else ""
            print("  %-30s = %15.0f%s" % (n, counters[n], pw))

    if waves > 0:
        busy = counters.get("SQ_BUSY_CYCLES", 1)
        wait = counters.get("SQ_WAIT_ANY", 0)
        vmem = counters.get("SQ_INSTS_VMEM_RD", 0) + counters.get("SQ_INSTS_VMEM_WR", 0)
        lds = counters.get("SQ_INSTS_LDS", 0)
        valu = counters.get("SQ_INSTS_VALU", 0)
        salu = counters.get("SQ_INSTS_SALU", 0)
        print("\n  Wait/Busy = %.1f%%" % (wait / busy * 100))
        if vmem + lds > 0:
            print("  VMEM fraction = %.0f%%" % (vmem / (vmem + lds) * 100))
        print("  VALU/wave = %.1f" % (valu / waves))
        print("  SALU/wave = %.1f" % (salu / waves))
        print("  VMEM_RD/wave = %.1f" % (counters.get("SQ_INSTS_VMEM_RD", 0) / waves))
else:
    print("No GEMM dispatch found")

db.close()
