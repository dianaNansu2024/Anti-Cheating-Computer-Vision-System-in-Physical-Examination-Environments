"""
ReportGenerator v3 — Video-aware report with incident timeline.
Groups incidents by video timestamp and seat, shows clickable time markers.
"""

import os
from datetime import datetime
from collections import defaultdict


class ReportGenerator:
    def __init__(self, report_dir="reports"):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def generate(self, session):
        exam_id    = session.get("exam_id", "UNKNOWN")
        alerts     = session.get("alerts", [])
        persons    = session.get("persons", {})
        video_name = session.get("video_name", "Unknown")
        duration   = session.get("duration_sec", 0)
        fps        = session.get("fps", 25)

        high   = sum(1 for a in alerts if a.get("severity") == "high")
        medium = sum(1 for a in alerts if a.get("severity") == "medium")
        low    = sum(1 for a in alerts if a.get("severity") == "low")

        # Group by seat
        by_seat = defaultdict(list)
        for a in alerts:
            by_seat[str(a.get("seat", "?"))].append(a)

        seats_html = ""
        for sid in sorted(by_seat.keys()):
            seat_alerts = by_seat[sid]
            info = persons.get(int(sid) if sid.isdigit() else sid, {})
            risk = info.get("risk", 0)
            color = "#ff4060" if risk > 60 else "#f5a623" if risk > 30 else "#20e070"
            count = len(seat_alerts)
            seats_html += f"""
            <div class="seat-card">
              <div class="seat-header">
                <span class="seat-label">SEAT {sid}</span>
                <span style="color:{color};font-family:'IBM Plex Mono',monospace;font-weight:600">{risk:.0f}% RISK</span>
                <span class="seat-count">{count} incident{'s' if count!=1 else ''}</span>
              </div>
              <div class="seat-timeline">
                {''.join(self._incident_chip(a) for a in reversed(seat_alerts))}
              </div>
            </div>"""

        # Full incident log sorted by video time
        sorted_alerts = sorted(alerts, key=lambda a: a.get("frame", 0))
        timeline_html = ""
        for a in sorted_alerts:
            sv = a.get("severity", "medium")
            bc = "#ff4060" if sv == "high" else "#f5a623" if sv == "medium" else "#5a7a96"
            vt = a.get("video_time", "--:--:--")
            timeline_html += f"""
            <div class="inc-row">
              <span class="inc-time">{vt}</span>
              <span class="inc-seat">SEAT {a.get('seat','?')}</span>
              <span class="inc-badge" style="border-color:{bc};color:{bc}">{sv.upper()}</span>
              <span class="inc-type">{a.get('type','')}</span>
              <span class="inc-detail">{a.get('detail','')}</span>
            </div>"""

        if not timeline_html:
            timeline_html = '<div class="clean">✅ No incidents detected during this recording.</div>'

        mins  = int(duration // 60)
        secs  = int(duration % 60)
        dur_fmt = f"{mins}m {secs}s"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>InvigilAI v3 Report — {exam_id}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{{--bg:#060a0f;--bg2:#0b1320;--bg3:#101c2e;--border:#172436;--teal:#00c8aa;--amber:#f5a623;--red:#ff4060;--green:#20e070;--blue:#3ab5ff;--ink:#e8f0f7;--dim:#5a7a96}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--ink);font-family:'Syne',sans-serif;padding:2.5rem;max-width:1100px;margin:0 auto;line-height:1.5}}
h1{{color:var(--teal);font-size:1.8rem;font-weight:800;letter-spacing:.06em;margin-bottom:.3rem}}
.subtitle{{color:var(--dim);font-family:'IBM Plex Mono',monospace;font-size:.75rem;letter-spacing:.1em;margin-bottom:2rem}}
h2{{color:var(--blue);font-size:1rem;font-weight:600;letter-spacing:.1em;border-bottom:1px solid var(--border);padding-bottom:.5rem;margin:2rem 0 1rem}}
.meta-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:.5rem;margin-bottom:2rem;background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:1.2rem}}
.meta-item{{display:flex;flex-direction:column;gap:2px}}
.meta-item .k{{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:var(--dim);letter-spacing:.1em}}
.meta-item .v{{font-size:.95rem;font-weight:600}}
.stat-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem;margin-bottom:2rem}}
.stat{{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:1rem;text-align:center}}
.stat .n{{font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:600;line-height:1}}
.stat .l{{font-size:.65rem;color:var(--dim);letter-spacing:.08em;margin-top:.3rem}}
.seat-card{{background:var(--bg2);border:1px solid var(--border);border-radius:6px;margin:.6rem 0;overflow:hidden}}
.seat-header{{display:flex;align-items:center;gap:1rem;padding:.7rem 1rem;background:var(--bg3)}}
.seat-label{{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:.9rem}}
.seat-count{{margin-left:auto;color:var(--dim);font-size:.8rem}}
.seat-timeline{{padding:.6rem 1rem;display:flex;flex-wrap:wrap;gap:.4rem}}
.chip{{font-family:'IBM Plex Mono',monospace;font-size:.65rem;padding:2px 8px;border-radius:12px;border:1px solid;white-space:nowrap}}
.chip.high{{border-color:var(--red);color:var(--red);background:rgba(255,64,96,.08)}}
.chip.medium{{border-color:var(--amber);color:var(--amber);background:rgba(245,166,35,.08)}}
.chip.low{{border-color:var(--dim);color:var(--dim)}}
.inc-row{{display:grid;grid-template-columns:90px 70px 80px 200px 1fr;gap:.8rem;align-items:center;padding:.5rem .8rem;border-bottom:1px solid var(--border);font-size:.8rem}}
.inc-row:last-child{{border-bottom:none}}
.inc-row:nth-child(even){{background:var(--bg2)}}
.inc-time{{font-family:'IBM Plex Mono',monospace;color:var(--teal);font-size:.72rem}}
.inc-seat{{font-family:'IBM Plex Mono',monospace;color:var(--dim);font-size:.72rem}}
.inc-badge{{font-family:'IBM Plex Mono',monospace;font-size:.62rem;padding:1px 6px;border-radius:10px;border:1px solid;text-align:center}}
.inc-type{{font-weight:600;font-size:.75rem}}
.inc-detail{{color:var(--dim);font-size:.72rem}}
.clean{{color:var(--green);text-align:center;padding:3rem;font-size:1.1rem}}
.tbl-header{{display:grid;grid-template-columns:90px 70px 80px 200px 1fr;gap:.8rem;padding:.4rem .8rem;background:var(--bg3);font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:var(--dim);letter-spacing:.08em;border-radius:4px 4px 0 0}}
footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--dim);font-size:.72rem;font-family:'IBM Plex Mono',monospace}}
</style></head><body>
<h1>🎓 INVIGILAI v3 — EXAM ANALYSIS REPORT</h1>
<div class="subtitle">BEHAVIORAL DETECTION · VIDEO FILE MODE · REGISTRATION-FREE</div>

<div class="meta-grid">
  <div class="meta-item"><span class="k">EXAM ID</span><span class="v">{exam_id}</span></div>
  <div class="meta-item"><span class="k">VIDEO FILE</span><span class="v">{video_name}</span></div>
  <div class="meta-item"><span class="k">DURATION</span><span class="v">{dur_fmt}</span></div>
  <div class="meta-item"><span class="k">GENERATED</span><span class="v">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></div>
</div>

<div class="stat-row">
  <div class="stat"><div class="n" style="color:var(--teal)">{len(alerts)}</div><div class="l">TOTAL</div></div>
  <div class="stat"><div class="n" style="color:var(--red)">{high}</div><div class="l">HIGH</div></div>
  <div class="stat"><div class="n" style="color:var(--amber)">{medium}</div><div class="l">MEDIUM</div></div>
  <div class="stat"><div class="n" style="color:var(--dim)">{low}</div><div class="l">LOW</div></div>
  <div class="stat"><div class="n" style="color:var(--blue)">{len(persons)}</div><div class="l">SEATS</div></div>
</div>

<h2>PER-SEAT SUMMARY</h2>
{seats_html if seats_html else '<div class="clean">No persons were tracked.</div>'}

<h2>FULL INCIDENT TIMELINE</h2>
<div class="tbl-header"><span>VIDEO TIME</span><span>SEAT</span><span>SEVERITY</span><span>TYPE</span><span>DETAIL</span></div>
<div style="background:var(--bg2);border:1px solid var(--border);border-radius:0 0 4px 4px">{timeline_html}</div>

<footer>
⚠ This report assists human invigilators. All incidents require human review before any action is taken.<br>
InvigilAI v3 — Video File Behavioral Monitor | {datetime.now().strftime('%Y-%m-%d')}
</footer>
</body></html>"""

        path = f"{self.report_dir}/{exam_id}_report.html"
        with open(path, "w") as f:
            f.write(html)
        print(f"[Report] Saved → {path}")
        return path

    def _incident_chip(self, a):
        sv = a.get("severity", "medium")
        vt = a.get("video_time", "?")
        t  = a.get("type", "?").replace("_", " ")
        return f'<span class="chip {sv}" title="{t}">{vt}</span>'
