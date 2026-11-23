"""Dark 'projection booth' look: palette, page CSS and an Altair theme. One accent (gold) is used for the Hit line; green and orange are reserved for Hit and Flop."""
BG, PANEL, PANEL2, LINE = "#0c1116", "#131a21", "#19222a", "#26323b"
TEXT, MUTED, GOLD = "#e8eef0", "#94a4aa", "#f2b632"
HIT, FLOP = "#4fd1a5", "#ff7a59"
BLUE, SLATE = "#6fa8dc", "#7c8b94"

CSS = f"""
<style>
@font-face {{font-family: "Archivo VF"; src: url("/app/static/archivo-wdth.woff2") format("woff2"); font-weight: 100 900; font-display: swap;}}
@font-face {{font-family: "Barlow Cond"; src: url("/app/static/barlow-condensed-800.woff2") format("woff2"); font-weight: 800; font-display: swap;}}
@font-face {{font-family: "Barlow Cond"; src: url("/app/static/barlow-condensed-700.woff2") format("woff2"); font-weight: 700; font-display: swap;}}
:root {{--bg:{BG}; --panel:{PANEL}; --panel2:{PANEL2}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED}; --gold:{GOLD}; --hit:{HIT}; --flop:{FLOP};
       --head:"Barlow Cond","Arial Narrow","Roboto Condensed",sans-serif;}}
.stApp {{background: var(--bg); color: var(--text);}}
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea, .stApp a {{font-family: "Archivo VF", "Segoe UI", system-ui, sans-serif;}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{font-family: var(--head); color: var(--text);}}
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded {{font-family: "Material Symbols Rounded" !important;}}
.block-container {{padding-top: 3.2rem !important; padding-bottom: 2rem !important; max-width: 1240px;}}
h2, h3 {{font-weight: 800 !important; font-size: 1.7rem !important; line-height: 1.15 !important; letter-spacing: .005em; padding: .1rem 0 .3rem 0 !important; font-synthesis: none;}}
h4 {{font-weight: 700 !important; font-size: 1.2rem !important;}}
.stApp a {{color: var(--gold);}}
[data-testid="stCaptionContainer"], .stApp small {{color: var(--muted) !important;}}

.top {{display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap;}}
.logo {{font-family:var(--head); font-weight:800; font-size:3rem; line-height:1.15; padding-top:.06em; color:var(--text); letter-spacing:.005em; font-synthesis:none;}}
.logo i {{font-style:normal; color:var(--gold);}}
.top a {{font-weight:600; text-decoration:underline; text-underline-offset:3px; font-size:.92rem; margin-top:.9rem;}}
.lede {{font-size:1.02rem; color:var(--muted); max-width: 72ch; margin: 2px 0 14px 0; line-height:1.5;}}
.kpis {{display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:10px; margin: 4px 0 8px 0;}}
.kpi {{background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px 14px;}}
.kpi b {{display:block; font-family:var(--head); font-weight:800; font-size:1.9rem; line-height:1.1; color:var(--text); font-synthesis:none; font-variant-numeric: tabular-nums;}}
.kpi span {{color: var(--muted); font-size:.84rem; line-height:1.3;}}

[data-testid="stVerticalBlockBorderWrapper"] {{background: var(--panel); border-color: var(--line) !important; border-radius: 14px;}}
[data-testid="stVerticalBlock"] {{gap: .7rem;}}
div[data-testid="stButton"] button {{font-weight:700; padding:.6rem 1rem; border-radius:10px;}}
div[data-testid="stButton"] button[kind="primary"] {{background: var(--gold); color:#14120a; border:0;}}
div[data-testid="stButton"] button[kind="primary"]:hover {{background:#ffcb55; color:#14120a;}}
button[data-baseweb="tab"] p {{font-weight:650; font-size:.98rem;}}
[data-testid="stMetric"] {{background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px 14px;}}
[data-testid="stDataFrame"] {{border:1px solid var(--line); border-radius:10px;}}

.ticket {{position:relative; background:linear-gradient(180deg,#17212a,#101820); color:var(--text); border:1px solid var(--line); border-radius:16px; margin: 2px 0 10px 0; box-shadow: 0 12px 34px rgba(0,0,0,.45);}}
.tk-top {{display:flex; justify-content:space-between; gap:12px; background:var(--gold); color:#14120a; font-weight:700; font-size:.82rem; padding:8px 22px; border-radius:15px 15px 0 0;}}
.tk-main {{position:relative; padding:16px 22px 14px 22px;}}
.tk-title {{font-family:var(--head); font-weight:800; font-size:1.9rem; line-height:1.1; padding-right:130px; overflow-wrap:anywhere; font-synthesis:none;}}
.tk-meta {{color:var(--muted); font-size:.9rem; margin-top:4px; padding-right:130px; line-height:1.35;}}
.tk-num {{margin-top:10px; line-height:.95; display:flex; align-items:baseline;}}
.tk-num .n {{font-family:var(--head); font-weight:800; font-size:5.4rem; line-height:.95; font-variant-numeric: tabular-nums; font-synthesis:none;}}
.tk-num .pc {{font-family:var(--head); font-weight:700; font-size:2.2rem; margin-left:4px; color:var(--muted);}}
.tk-lab {{color:#c3cfd3; font-size:.95rem; margin-top:2px;}}
.track {{position:relative; height:11px; background:rgba(255,255,255,.12); border-radius:8px; margin:30px 4px 4px 0;}}
.fill {{height:100%; border-radius:8px; background:var(--text);}}
.thr {{position:absolute; top:-8px; bottom:-8px; width:3px; background:var(--gold); border-radius:2px; transform:translateX(-1px);}}
.thr span {{position:absolute; bottom:calc(100% + 3px); left:50%; transform:translateX(-50%); white-space:nowrap; font-size:.75rem; font-weight:700; color:var(--gold);}}
.stamp {{position:absolute; top:14px; right:20px; transform:rotate(-7deg); border:3px solid; border-radius:8px; padding:0 12px; font-family:var(--head); font-weight:800; font-size:2.1rem; line-height:1.25; letter-spacing:.04em; font-synthesis:none;}}
.ticket.hit .stamp {{color:var(--hit); border-color:var(--hit);}}
.ticket.flop .stamp {{color:var(--flop); border-color:var(--flop);}}
.tk-perf {{position:relative; border-top:2px dashed rgba(255,255,255,.22); margin:0 18px;}}
.tk-perf::before, .tk-perf::after {{content:""; position:absolute; top:-12px; width:22px; height:22px; border-radius:50%; background:var(--bg); border:1px solid var(--line);}}
.tk-perf::before {{left:-30px;}} .tk-perf::after {{right:-30px;}}
.tk-foot {{padding:10px 22px 14px 22px; color:var(--muted); font-size:.8rem; line-height:1.4;}}

.ghost {{border:2px dashed #34424c; border-radius:16px; padding:22px 24px 18px 24px; color:var(--muted); background:rgba(255,255,255,.02);}}
.ghost .g-title {{font-family:var(--head); font-weight:800; font-size:1.8rem; line-height:1.15; color:var(--text); margin-bottom:6px; font-synthesis:none;}}
.ghost p {{margin:0 0 8px 0; line-height:1.5; font-size:.98rem;}}
.ghost ul {{margin:4px 0 10px 18px; padding:0; line-height:1.55;}}
.ghost .small {{font-size:.85rem;}}
.note {{background:#1d1a0e; border-left:4px solid var(--gold); border-radius:6px; padding:12px 16px; margin:8px 0 14px 0; line-height:1.5; color:var(--text);}}
.pill {{display:inline-block; border:1px solid var(--line); border-radius:999px; padding:1px 10px; font-size:.8rem; color:var(--muted); margin-right:6px;}}
.pill.good {{color:var(--hit); border-color:#1f5b4a;}} .pill.bad {{color:var(--flop); border-color:#6b3a2c;}}
.footer {{margin-top:30px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem;}}
@media (max-width: 800px) {{
  .logo {{font-size:2.5rem;}} .tk-num .n {{font-size:4.6rem;}} .tk-title {{padding-right:0; padding-top:46px;}} .tk-meta {{padding-right:0;}} .stamp {{font-size:1.7rem; top:10px;}}
}}
@media (prefers-reduced-motion: reduce) {{* {{transition: none !important; animation: none !important;}}}}
</style>
"""


def altair_style(chart, height=None):
    """Apply the dark chart style to any Altair chart."""
    c = chart.configure(background="transparent").configure_view(strokeWidth=0)
    c = c.configure_axis(labelColor=TEXT, titleColor=MUTED, gridColor=LINE, domainColor=LINE, tickColor=LINE, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
    c = c.configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=12, orient="top")
    c = c.configure_title(color=TEXT, fontSize=14)
    return c.properties(height=height) if height else c
