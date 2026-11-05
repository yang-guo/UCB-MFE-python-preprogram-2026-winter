"""Render checked tables separately from the agent's interpretation."""

from html import escape
import json
from pathlib import Path

from .data import portfolio_exposure
from .memory import connect, evidence


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_report(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    with connect(run) as db:
        tasks = [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY ticker")]
        records = evidence(db)
        notes = {r["ticker"]: dict(r) for r in db.execute("SELECT * FROM notes")}
        call_count = db.execute("SELECT count(*) FROM calls").fetchone()[0]
        usage = [json.loads(row[0]) for row in db.execute("SELECT payload FROM events WHERE kind='agent_usage' ORDER BY id")]
    usage_text = (f"Recorded model requests: {sum(u['requests'] for u in usage)}. "
                  f"Successful agent tool calls: {sum(u['tool_calls'] for u in usage)}. "
                  f"Usage receipts: {len(usage)} of {sum(t['attempts'] for t in tasks)} attempts.")
    lookup = {e["id"]: e for e in records}
    for task in tasks:
        if task["status"] == "complete":
            note = notes.get(task["ticker"])
            if not note:
                raise ValueError("Completed task has no note")
            cited = json.loads(note["citations"])
            if any(c not in lookup or lookup[c]["ticker"] != task["ticker"] for c in cited):
                raise ValueError("Broken evidence reference")
            if not {"source", "metrics"}.issubset({lookup[c]["kind"] for c in cited}):
                raise ValueError("Completed task lacks required evidence")
    metrics = {e["ticker"]: e["payload"] for e in records if e["kind"] == "metrics"}
    completed = {t["ticker"] for t in tasks if t["status"] == "complete"}
    state = "COMPLETE" if len(completed) == len(tasks) else "INCOMPLETE"
    rows = []
    for ticker in sorted(completed):
        m = metrics[ticker]
        rows.append([ticker, m["first_date"], m["last_date"], str(m["observations"]),
                     f'{m["total_return"]:.2%}', f'{m["annualized_log_volatility"]:.2%}',
                     f'{m["max_drawdown"]:.2%}'])
    header = ["Ticker", "First date", "Last date", "Prices", "Total return", "Ann. log vol.", "Max drawdown"]
    md = ["# Stock research report", "", f"**{state}** · {config['dataset_label']}", "",
          f"Question: {config['question']}", "",
          f"Requested window: {config['start']} through {config['as_of']}.",
          f"Engine: {config['engine']}. Completed {len(completed)} of {len(tasks)} tickers. Research actions: {call_count}.", "", usage_text, "",
          "## Computed evidence", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Stock research report</title>",
            "<style>body{font:17px system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#183044}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:9px;text-align:left}pre{white-space:pre-wrap}small{color:#526272}</style>",
            f"<h1>Stock research report</h1><p><strong>{state}</strong> · {escape(config['dataset_label'])}</p>",
            f"<p>{escape(config['question'])}</p><p>Requested window: {config['start']} through {config['as_of']}. Engine: {config['engine']}.</p>",
            f"<p>Completed {len(completed)} of {len(tasks)} tickers. Research actions: {call_count}.</p><p>{usage_text}</p><h2>Computed evidence</h2><table><tr>",
            "".join(f"<th>{escape(x)}</th>" for x in header) + "</tr>"]
    html += ["<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>" for row in rows]
    html.append("</table>")
    conventions = "Returns and drawdown use adjusted prices. Annualized volatility uses the sample standard deviation of daily log returns and 252 sessions per year. The dates and row counts show the history available for each stock."
    md += ["", conventions]
    html.append(f"<p>{conventions}</p>")
    # Whole-period comparisons need identical windows. Avoid ranking unequal coverage.
    windows = {(metrics[t]["first_date"], metrics[t]["last_date"], metrics[t]["observations"]) for t in completed}
    comparison = "No common-window ranking is available."
    if len(completed) >= 2 and len(windows) == 1:
        highest = max(sorted(completed), key=lambda t: metrics[t]["annualized_log_volatility"])
        comparison = f"{highest} has the highest annualized volatility over the shared date range."
    md += ["", comparison, "", "## Portfolio exposure", ""]
    html += [f"<p>{escape(comparison)}</p><h2>Portfolio exposure</h2>"]
    exposure = None
    try:
        exposure = portfolio_exposure(config["holdings_path"], {t: metrics[t] for t in completed}, config["cash"])
        exposure_text = (f"Valuation date: {exposure['valuation_date']}. Total value including cash: "
                         f"{exposure['total_value']:,.2f}. Cash: {exposure['cash']:,.2f} "
                         f"({exposure['cash_weight']:.2%}). Values use the supplied price units.")
    except ValueError as error:
        exposure_text = f"Portfolio exposure is unavailable: {error}."
    md += [exposure_text, ""]
    html.append(f"<p>{escape(exposure_text)}</p>")
    if exposure:
        columns = ["Ticker", "Shares", "Weighted entry", "Raw close", "Market value", "Weight", "Unrealized P&L"]
        positions = [[p["ticker"], f'{p["shares"]:g}', f'{p["weighted_entry_price"]:.2f}',
                      f'{p["raw_close"]:.2f}', f'{p["market_value"]:,.2f}',
                      f'{p["weight"]:.2%}', f'{p["unrealized_pnl"]:,.2f}'] for p in exposure["positions"]]
        md += ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        md += ["| " + " | ".join(row) + " |" for row in positions]
        html.append("<table><tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>")
        html += ["<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in positions]
        html.append("</table>")
    exposure_scope = "Each weight is the holding's market value divided by total portfolio value, including cash."
    md += ["", exposure_scope, "", "## Research notes", ""]
    html += [f"<p>{exposure_scope}</p><h2>Research notes</h2>"]
    for ticker in sorted(completed):
        note = notes[ticker]
        ids = json.loads(note["citations"])
        md += ["", f"### {ticker}", "", note["body"], "", "Evidence: " + ", ".join(ids)]
        html += [f"<h3>{ticker}</h3><p>{escape(note['body'])}</p><small>Evidence: {escape(', '.join(ids))}</small>"]
    md += ["", "## Task status", ""]
    html.append("<h2>Task status</h2>")
    for task in tasks:
        line = f"{task['ticker']}: {task['status']}, attempts={task['attempts']}, error={task['error'] or 'none'}"
        md += ["", line]
        html.append(f"<p>{escape(line)}</p>")
    provenance = {key: config[key] for key in ["contract_version", "question", "dataset", "start", "as_of", "engine", "model", "code_sha256", "data_sha256", "sources_sha256", "holdings_sha256"]}
    md += ["", "## Project provenance", "", "```json", json.dumps(provenance, indent=2), "```"]
    html.append(f"<h2>Project provenance</h2><pre>{escape(json.dumps(provenance, indent=2))}</pre>")
    md += ["", "## Evidence appendix", "", "See evidence.json for the complete payloads and source hashes."]
    html.append("<h2>Evidence appendix</h2><p>See evidence.json for full payloads and source hashes.</p>")
    for e in records:
        line = f"{e['id']} · {e['kind']} · {e['ticker']}"
        md += ["", f"### {line}", "", "```json", json.dumps(e["payload"], indent=2), "```"]
        html.append(f"<h3>{escape(line)}</h3><pre>{escape(json.dumps(e['payload'], indent=2))}</pre>")
    html.append("</html>")
    atomic_text(run / "evidence.json", json.dumps(records, indent=2, allow_nan=False))
    atomic_text(run / "summary.json", json.dumps({"status": state, "tasks": tasks, "metrics": {t: metrics[t] for t in sorted(completed)}, "portfolio": exposure, "provenance": provenance, "usage": usage}, indent=2, allow_nan=False))
    atomic_text(run / "report.md", "\n".join(md) + "\n")
    atomic_text(run / "report.html", "\n".join(html))
    return state
