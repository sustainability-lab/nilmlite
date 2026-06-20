"""Render evaluation results as a text table and a standalone HTML leaderboard."""
from __future__ import annotations

from pathlib import Path

__all__ = ["leaderboard_text", "leaderboard_html"]

_MEDALS = ["🥇", "🥈", "🥉"]


def _params(p) -> str:
    if p is None:
        return "—"
    if p >= 1e6:
        return f"{p/1e6:.1f}M"
    if p >= 1e3:
        return f"{p/1e3:.0f}K"
    return str(int(p))


def _ranked(result: dict) -> list[tuple[str, dict]]:
    return sorted(result["models"].items(), key=lambda kv: kv[1]["mae"])


def leaderboard_text(result: dict) -> str:
    rows = _ranked(result)
    out = [f"\n▶ {result['task']}  ({result['kind']})",
           f"  appliances: {', '.join(result['appliances'])}",
           "  " + "-" * 52,
           f"  {'model':<16}{'MAE↓ W':>10}{'F1↑':>8}{'params':>9}{'infer':>9}",
           "  " + "-" * 52]
    for name, m in rows:
        out.append(f"  {name:<16}{m['mae']:>10.2f}{m['f1']:>8.3f}"
                   f"{_params(m['params']):>9}{m['infer_ms']:>7.0f}ms")
    out.append("  " + "-" * 52)
    out.append("  leaderboard (mean MAE, lower is better)")
    for i, (name, m) in enumerate(rows):
        medal = _MEDALS[i] if i < 3 else f" {i+1}"
        out.append(f"   {medal} {name:<14} {m['mae']:.2f} W")
    return "\n".join(out)


def _table_html(result: dict) -> str:
    rows = _ranked(result)
    body = []
    for i, (name, m) in enumerate(rows):
        medal = _MEDALS[i] if i < 3 else f"{i+1}"
        cls = ' class="top"' if i == 0 else ""
        body.append(
            f"<tr{cls}><td class='rank'>{medal}</td><td class='name'>{name}</td>"
            f"<td>{m['mae']:.2f}</td><td>{m['f1']:.3f}</td>"
            f"<td>{_params(m['params'])}</td><td>{m['infer_ms']:.0f} ms</td></tr>")
    return f"""
    <section class="task">
      <h2>{result['task']} <span class="kind">{result['kind']}</span></h2>
      <p class="apps">appliances: {', '.join(result['appliances'])}</p>
      <table>
        <thead><tr><th></th><th>model</th><th>MAE&#8595; (W)</th>
        <th>F1&#8593;</th><th>params</th><th>inference</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
      {f'<p class="note">{result["note"]}</p>' if result.get("note") else ''}
    </section>"""


def _gap_callout(results: list[dict]) -> str:
    by_kind = {r["kind"]: r for r in results}
    cb, cd = by_kind.get("cross_building"), by_kind.get("cross_dataset")
    if not (cb and cd):
        return ""
    win = min(cb["models"], key=lambda k: cb["models"][k]["mae"])
    if win not in cd["models"]:
        return ""
    a, b = cb["models"][win]["mae"], cd["models"][win]["mae"]
    pct = (b - a) / a * 100 if a else 0
    return f"""
    <div class="gap">
      <div class="gap-h">the generalization gap</div>
      <div class="gap-b"><b>{win}</b> degrades
      <b>{a:.1f} W → {b:.1f} W</b> ({pct:+.0f}%) moving from a new <i>home</i>
      to a new <i>dataset</i>. Cross-dataset generalization is the open problem.</div>
    </div>"""


def leaderboard_html(results: list[dict] | dict, out_path: str | Path,
                     title: str = "NILMBench Leaderboard") -> Path:
    if isinstance(results, dict):
        results = [results]
    tables = "\n".join(_table_html(r) for r in results)
    gap = _gap_callout(results)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root{{--acc:#e4634f;--bg:#0c1018;--card:#141a26;--mut:#8b93a4;--line:#222c3d}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:#eef2f8;
    font-family:-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;padding:48px 20px}}
  .wrap{{max-width:840px;margin:0 auto}}
  .kic{{letter-spacing:.28em;text-transform:uppercase;color:var(--acc);font-weight:700;font-size:13px}}
  h1{{font-family:Georgia,serif;font-size:40px;margin:.2em 0 .1em}}
  .sub{{color:var(--mut);margin-bottom:34px}}
  section.task{{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:22px 26px;margin-bottom:22px}}
  h2{{font-size:22px;margin:0 0 2px}}
  .kind{{font-size:12px;font-weight:700;color:var(--acc);border:1px solid var(--acc);
    border-radius:999px;padding:2px 10px;margin-left:8px;vertical-align:middle}}
  .apps{{color:var(--mut);font-size:13px;margin:0 0 14px}}
  table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
  th,td{{text-align:right;padding:9px 8px;border-bottom:1px solid var(--line)}}
  th{{color:var(--mut);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}}
  th:nth-child(2),td.name{{text-align:left}}
  td.rank{{text-align:center;font-size:17px;width:34px}}
  td.name{{font-weight:700}}
  tr.top td{{background:rgba(228,99,79,.10)}}
  tr.top td.name{{color:var(--acc)}}
  .note{{color:var(--mut);font-size:13px;margin:12px 0 0}}
  .gap{{background:linear-gradient(180deg,rgba(228,99,79,.14),rgba(228,99,79,.04));
    border:1px solid var(--acc);border-radius:14px;padding:20px 24px;margin-bottom:22px}}
  .gap-h{{color:var(--acc);font-weight:800;letter-spacing:.04em;text-transform:uppercase;
    font-size:13px;margin-bottom:6px}}
  .gap-b{{font-size:16px;line-height:1.5}}
  footer{{color:var(--mut);font-size:12px;text-align:center;margin-top:26px}}
  footer a{{color:var(--acc)}}
</style></head>
<body><div class="wrap">
  <div class="kic">ACM BuildSys 2026 · NILMBench</div>
  <h1>{title}</h1>
  <p class="sub">Add a model, read the leaderboard. Same harness, sealed test sets,
  every metric. Generated by <code>nilmlite</code>.</p>
  {gap}
  {tables}
  <footer>github.com/sustainability-lab/nilmlite — toward an ImageNet moment for NILM</footer>
</div></body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
