from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from jinja2 import Environment, FileSystemLoader, select_autoescape

_WORKLOAD_CATALOG: dict[str, dict[str, str]] = {
    "short_qa": {
        "name": "short_qa",
        "scenario": "客服 / 搜索问答 / FAQ",
        "description": "短 prompt、短回答、高交互频率，优先关注 TTFT 与 P95，避免排队导致首 token 变慢。",
    },
    "long_gen": {
        "name": "long_gen",
        "scenario": "内容生成 / 摘要 / 长文改写",
        "description": "较长输出、decode 占比高，优先 tok/s 与成本，同时约束质量与显存。",
    },
    "agent_like": {
        "name": "agent_like",
        "scenario": "Agent / RAG 多轮",
        "description": "多轮上下文增长、长度波动大，容易拉高 queue/prefill 与尾延迟；需兼顾 TTFT、P95 与 tail_amp。",
    },
}


@dataclass
class RenderReportConfig:
    out_dir: str = "reports"
    template_dir: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_upward(start: Path, relative_candidates: list[str]) -> Path | None:
    cur = start.resolve()
    chain = [cur, *cur.parents]
    for parent in chain:
        for rel in relative_candidates:
            p = parent / rel
            if p.exists():
                return p
    return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _fmt_num(v: Any, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "-"


def _candidate_files(art_dir: Path) -> list[Path]:
    return sorted(art_dir.glob("candidate_*.json"))


def _load_candidates(art_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in _candidate_files(art_dir):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _dedupe_recommendations(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in recs:
        sig = json.dumps(
            r.get("params") or r.get("serving_config") or {}, sort_keys=True, ensure_ascii=False
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
    return out


def _extract_recommendations(art_dir: Path, tuning_summary: dict[str, Any]) -> dict[str, Any]:
    raw = tuning_summary.get("recommendations") or []
    recs: list[dict[str, Any]] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        params = r.get("params") or {}
        ov = r.get("objective_values") or {}
        scfg = r.get("serving_config") or {}
        recs.append(
            {
                "candidate_idx": r.get("candidate_idx"),
                "score": r.get("score"),
                "note": r.get("note", ""),
                "params": params,
                "objective_values": ov,
                "serving_config": scfg,
            }
        )
    recs = _dedupe_recommendations(recs)
    if recs:
        return {"topk": recs[:5], "has_recommendations": True}

    # Fallback: build from candidate artifacts
    cands = _load_candidates(art_dir)
    if not cands:
        return {"topk": [], "has_recommendations": False}

    feasible = [c for c in cands if bool(c.get("feasible", False))]
    pool = feasible or cands

    def score(c: dict[str, Any]) -> tuple[float, float, float]:
        ov = c.get("objective_values") or {}
        return (
            _safe_float(ov.get("latency_p95_s", ov.get("p95_s", 1e9)), 1e9),
            _safe_float(ov.get("ttft_p95_s", 1e9), 1e9),
            -_safe_float(ov.get("tok_s", 0.0), 0.0),
        )

    pool = sorted(pool, key=score)
    topk: list[dict[str, Any]] = []
    for c in pool[:5]:
        topk.append(
            {
                "candidate_idx": c.get("idx"),
                "score": None,
                "note": "fallback_rank",
                "params": c.get("params") or {},
                "objective_values": c.get("objective_values") or {},
                "serving_config": c.get("serving_config") or {},
            }
        )
    topk = _dedupe_recommendations(topk)
    return {"topk": topk, "has_recommendations": bool(topk)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _ensure_tail_plot(art_dir: Path) -> str | None:
    if plt is None:
        return None
    out_path = art_dir / "tail_attribution.png"
    if out_path.exists():
        return out_path.name

    payload = _load_json(art_dir / "tail_compare.json")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not rows:
        # Build from traces fallback
        rows = []
        label_to_file = {
            "Baseline": "traces_baseline.jsonl",
            "Random-best": "traces_random_best.jsonl",
            "NSGA2-best": "traces_nsga2_best.jsonl",
        }
        for label, fn in label_to_file.items():
            traces = _load_jsonl(art_dir / fn)
            if not traces:
                continue

            # permissive field names
            def pick(d: dict[str, Any], *keys: str) -> float:
                for k in keys:
                    if k in d:
                        return _safe_float(d[k])
                return 0.0

            vals_q = [pick(t, "queue_s", "queue_time_s", "queue") for t in traces]
            vals_p = [pick(t, "prefill_s", "prefill_time_s", "prefill") for t in traces]
            vals_d = [pick(t, "decode_s", "decode_time_s", "decode") for t in traces]
            vals_r = [pick(t, "retry_count", "retries", "retry") for t in traces]

            def p95(vals: list[float]) -> float:
                if not vals:
                    return 0.0
                vals = sorted(vals)
                i = max(0, min(len(vals) - 1, int(0.95 * (len(vals) - 1))))
                return float(vals[i])

            rows.append(
                {
                    "method": label,
                    "queue_p95_s": p95(vals_q),
                    "prefill_p95_s": p95(vals_p),
                    "decode_p95_s": p95(vals_d),
                    "retry_rate": (sum(1 for v in vals_r if v > 0) / len(vals_r))
                    if vals_r
                    else 0.0,
                    "p95_s": p95(
                        [a + b + c for a, b, c in zip(vals_q, vals_p, vals_d, strict=False)]
                    ),
                }
            )

    if not rows:
        return None

    labels = [str(r.get("method", "")) for r in rows]
    queue = [_safe_float(r.get("queue_p95_s")) for r in rows]
    prefill = [_safe_float(r.get("prefill_p95_s")) for r in rows]
    decode = [_safe_float(r.get("decode_p95_s")) for r in rows]
    retry_rate = [_safe_float(r.get("retry_rate")) for r in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    x = list(range(len(labels)))
    ax.bar(x, queue, label="queue")
    ax.bar(x, prefill, bottom=queue, label="prefill")
    bottom2 = [a + b for a, b in zip(queue, prefill, strict=False)]
    ax.bar(x, decode, bottom=bottom2, label="decode")

    for i, rr in enumerate(retry_rate):
        ax.text(
            i,
            bottom2[i] + decode[i] + 0.03,
            f"retry {rr:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Tail Latency Attribution (P95 decomposition)")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def _ensure_pareto_plot(art_dir: Path, recommendations: dict[str, Any]) -> str | None:
    if plt is None:
        return None
    out_path = art_dir / "pareto_front.png"
    if out_path.exists():
        return out_path.name

    cands = _load_candidates(art_dir)
    if not cands:
        return None

    xs_all: list[float] = []
    ys_all: list[float] = []
    xs_feas: list[float] = []
    ys_feas: list[float] = []
    top_marks: list[tuple[float, float, str]] = []

    top_idxs = {
        int(r.get("candidate_idx"))
        for r in (recommendations.get("topk") or [])
        if r.get("candidate_idx") is not None
    }
    for c in cands:
        ov = c.get("objective_values") or {}
        p95 = _safe_float(ov.get("latency_p95_s", ov.get("p95_s")))
        tok = _safe_float(ov.get("tok_s", ov.get("throughput_tok_s")))
        if p95 <= 0 or tok <= 0:
            continue
        xs_all.append(p95)
        ys_all.append(tok)
        if bool(c.get("feasible", False)):
            xs_feas.append(p95)
            ys_feas.append(tok)
        if int(c.get("idx", -1)) in top_idxs:
            top_marks.append((p95, tok, f"#{c.get('idx')}"))

    if not xs_all:
        return None

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(xs_all, ys_all, alpha=0.35, s=18, label="all")
    if xs_feas:
        ax.scatter(xs_feas, ys_feas, s=22, label="feasible")
    for x, y, lab in top_marks:
        ax.scatter([x], [y], s=52, marker="x")
        ax.text(x, y, f" {lab}", fontsize=8, va="bottom")
    ax.set_xlabel("P95 latency (s) ↓")
    ax.set_ylabel("tok/s ↑")
    ax.set_title("Pareto Search Cloud (P95 vs tok/s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def _derive_workload(
    art_dir: Path,
    comparison: dict[str, Any],
    run_config: dict[str, Any],
    tuning_summary: dict[str, Any],
) -> dict[str, Any]:
    wl = ""
    if isinstance(comparison, dict):
        wl = str(comparison.get("workload") or "")
    if not wl:
        rid = str((tuning_summary or {}).get("run_id") or art_dir.name)
        for k in _WORKLOAD_CATALOG:
            if k in rid:
                wl = k
                break
    info = dict(
        _WORKLOAD_CATALOG.get(wl, {"name": wl or "unknown", "scenario": "", "description": ""})
    )
    # user-provided config can override
    for d in [run_config.get("workload") if isinstance(run_config, dict) else None]:
        if isinstance(d, dict):
            info.update({k: v for k, v in d.items() if v not in (None, "")})
    return info


def _build_risk_hints(
    rows: list[dict[str, Any]],
    constraints: dict[str, Any],
    tuning_summary: dict[str, Any],
    recommendations: dict[str, Any],
    workload: dict[str, Any],
) -> list[str]:
    hints: list[str] = []

    by_method = {str(r.get("method")): r for r in rows if isinstance(r, dict)}
    b = by_method.get("Baseline")
    n = by_method.get("NSGA2-best")
    if b and n:
        if _safe_float(n.get("p95_s")) > _safe_float(b.get("p95_s")) * 1.10:
            hints.append(
                "NSGA2-best 的 P95 高于 Baseline 超过 10%，说明当前目标仍偏向吞吐/成本，建议进一步提高 P95 与 TTFT 权重。"
            )
        if _safe_float(n.get("ttft_p95_s")) > _safe_float(b.get("ttft_p95_s")) * 1.05:
            hints.append(
                "NSGA2-best 的 TTFT P95 高于 Baseline，agent_like 场景建议再收紧 TTFT 约束（如 baseline × 1.05）。"
            )
        if _safe_float(n.get("quality")) < _safe_float(b.get("quality")):
            hints.append(
                "NSGA2-best 质量分略低于 Baseline，建议扩大 eval pack 并按任务类型设置分项下限（结构化/代码/摘要）。"
            )
        if _safe_float(n.get("vram_peak_mb")) > 0 and _safe_float(
            n.get("vram_peak_mb")
        ) < _safe_float(b.get("vram_peak_mb")):
            hints.append(
                "NSGA2-best 显存占用显著低于 Baseline，这对线上稳定性是加分项，可进一步用于容量规划。"
            )

    n_feas = int(tuning_summary.get("n_feasible") or 0)
    n_all = int(tuning_summary.get("n_candidates") or 0)
    if n_all > 0 and (n_feas / n_all) < 0.35:
        hints.append(
            "可行解比例偏低，建议继续使用‘分场景搜索空间 + 两阶段搜索（粗搜→局部细搜）’来提升收敛效率。"
        )

    vlim = _safe_float(constraints.get("vram_limit_mb"))
    margin = _safe_float(constraints.get("vram_safety_margin_ratio"))
    if vlim > 0 and recommendations.get("topk"):
        top1 = recommendations["topk"][0]
        ov = top1.get("objective_values") or {}
        top_vram = _safe_float(ov.get("vram_peak_mb"))
        if top_vram > 0:
            soft_cap = vlim * max(0.0, 1.0 - margin)
            if top_vram >= soft_cap * 0.97:
                hints.append(
                    "Top1 显存接近安全阈值，建议继续下调 vram_limit 或提高 safety margin（例如 8%）。"
                )

    wl_name = str(workload.get("name", ""))
    if wl_name == "agent_like" and recommendations.get("topk"):
        top1 = recommendations["topk"][0]
        params = top1.get("params") or {}
        if params.get("enable_prefix_caching") is not True:
            hints.append(
                "agent_like 场景 Top1 未启用 prefix caching；真实后端中通常会显著影响 TTFT/吞吐，建议强制开启或优先搜索。"
            )

    # Dedupe hint
    topk = recommendations.get("topk") or []
    if len(topk) >= 2:
        sigs = [
            json.dumps((r.get("params") or {}), sort_keys=True, ensure_ascii=False)
            for r in topk[:3]
        ]
        if len(sigs) != len(set(sigs)):
            hints.append(
                "Top-K 推荐存在重复参数组合，已在报告渲染阶段去重；建议上游推荐逻辑也做去重。"
            )

    return hints


def _load_comparison_summary(art_dir: Path) -> dict[str, Any]:
    p = _find_upward(
        art_dir,
        [
            "results/formal_comparison/comparison_summary.json",
            "../results/formal_comparison/comparison_summary.json",
        ],
    )
    return _load_json(p) if p else {}


def _build_ctx(art_dir: Path) -> dict[str, Any]:
    tuning_summary = _load_json(art_dir / "tuning_summary.json")
    summary = _load_json(art_dir / "summary.json")
    run_config = _load_json(art_dir / "run_config.json")
    quality_summary = _load_json(art_dir / "quality_summary.json")
    comparison_summary = _load_comparison_summary(art_dir)

    recommendations = _extract_recommendations(art_dir, tuning_summary)
    pareto_png = _ensure_pareto_plot(art_dir, recommendations)
    tail_png = _ensure_tail_plot(art_dir)

    workload = _derive_workload(art_dir, comparison_summary, run_config, tuning_summary)
    constraints = (
        comparison_summary.get("constraints") if isinstance(comparison_summary, dict) else {}
    )
    objective = comparison_summary.get("objective") if isinstance(comparison_summary, dict) else {}
    rows = comparison_summary.get("rows") if isinstance(comparison_summary, dict) else []
    if not isinstance(rows, list):
        rows = []

    risk_hints = _build_risk_hints(
        rows, constraints or {}, tuning_summary or {}, recommendations, workload
    )

    plots = {
        "pareto_png": pareto_png,
        "tail_png": tail_png,
    }

    # For legacy templates
    ctx: dict[str, Any] = {
        "title": f"ServingBench-Tuner 推荐报告 · {workload.get('name', art_dir.name)}",
        "exp_id": str(summary.get("exp_id") or "formal_comparison"),
        "run_id": str(tuning_summary.get("run_id") or art_dir.name),
        "tuning_summary": tuning_summary,
        "summary": summary,
        "run_config": run_config,
        "quality_summary": quality_summary,
        "workload": workload,
        "recommendations": recommendations,
        "risk_hints": risk_hints,
        "plots": plots,
        "rows": rows,
        "constraints": constraints or {},
        "objective": objective or {},
        "comparison_summary": comparison_summary,
    }
    return ctx


def _resolve_out_root(
    art_dir: Path, cfg: RenderReportConfig, explicit_out_dir: str | Path | None
) -> Path:
    out_base = Path(explicit_out_dir) if explicit_out_dir is not None else Path(cfg.out_dir)
    exp_id = "formal_comparison"
    run_id = art_dir.name
    return out_base / exp_id / run_id


def render_report(
    run_artifact: str | Path | dict[str, Any],
    out_dir: str | Path | None = None,
    cfg: RenderReportConfig | None = None,
) -> dict[str, str]:
    """
    Compatible renderer:
      - render_report(Path(...), out_dir=...)
      - render_report(Path(...), cfg=RenderReportConfig(...))
      - render_report(dict_payload, ...)
    Returns keys: report_md, report_html (+ md/html aliases)
    """
    cfg = cfg or RenderReportConfig()

    if isinstance(run_artifact, (str, Path)):
        art_dir = Path(run_artifact)
        ctx = _build_ctx(art_dir)
    elif isinstance(run_artifact, dict):
        # dict mode: best-effort render without artifact scanning
        art_dir = Path(".")
        ctx = dict(run_artifact)
        ctx.setdefault("recommendations", {"topk": [], "has_recommendations": False})
        ctx.setdefault("plots", {})
        ctx.setdefault("risk_hints", [])
        ctx.setdefault("rows", [])
        ctx.setdefault("workload", {"name": "unknown", "scenario": "", "description": ""})
        ctx.setdefault("tuning_summary", {})
        ctx.setdefault("summary", {})
        ctx.setdefault("run_config", {})
    else:
        raise TypeError("run_artifact must be path or dict")

    template_dir = (
        Path(cfg.template_dir)
        if cfg.template_dir
        else (Path(__file__).resolve().parent / "templates")
    )
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
    )

    out_root = _resolve_out_root(art_dir, cfg, out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy images if generated under artifact dir
    if isinstance(run_artifact, (str, Path)):
        for k in ["pareto_png", "tail_png"]:
            name = (ctx.get("plots") or {}).get(k)
            if name:
                src = Path(run_artifact) / name
                dst = out_root / name
                try:
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                except Exception:
                    pass

    md_tpl = env.get_template("report.md.j2")
    html_tpl = env.get_template("report.html.j2")

    md = md_tpl.render(**ctx)
    html = html_tpl.render(**ctx)

    md_path = out_root / "report.md"
    html_path = out_root / "report.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    return {
        "report_md": str(md_path),
        "report_html": str(html_path),
        "md": str(md_path),
        "html": str(html_path),
    }
