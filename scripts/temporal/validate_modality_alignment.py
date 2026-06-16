"""Validate temporal modality alignment between JMeter JTL and OTEL traces.

Phase 2 goals:
- Timestamp alignment: For each business transaction sample in JTL (label starts with 'T_'),
  check that at least one trace span interval is contained within [timeStamp, timeStamp+elapsed].
- Feature extraction: For a target transaction label (e.g., 'T_SearchFlights'), extract backend
  class names from spans and report coverage.

Assumptions / inputs:
- JTL is CSV with headers (as produced by JMeter -l in this project).
- all_traces.json may be:
  - OTEL JSON exporter envelope objects (with 'resourceSpans')
  - JSON Lines / NDJSON (one JSON object per line)
  - Concatenated JSON objects separated by newlines (collector file sink)

This script is intentionally conservative: it reports alignment stats and leaves strict thresholds
(configurable) to the caller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class JtlSample:
    label: str
    ts_ms: int
    elapsed_ms: int
    success: bool

    @property
    def start_us(self) -> int:
        return self.ts_ms * 1000

    @property
    def end_us(self) -> int:
        return (self.ts_ms + self.elapsed_ms) * 1000


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    start_us: int
    end_us: int
    attrs: Dict[str, Any]
    name: str


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        if not raw:
            return []
        # JSON array
        if raw.startswith("["):
            return json.loads(raw)
        # JSON Lines
        records: List[Dict[str, Any]] = []
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        if math.isnan(v):
            return None
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_jtl_samples(jtl_path: str, *, only_transactions: bool = True, only_success: bool = True) -> List[JtlSample]:
    samples: List[JtlSample] = []
    with open(jtl_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Expected columns: timeStamp, elapsed, label, success
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            if only_transactions and not label.startswith("T_"):
                continue
            ts = _to_int(row.get("timeStamp"))
            elapsed = _to_int(row.get("elapsed") or row.get("time"))
            if ts is None or elapsed is None:
                continue
            success_raw = (row.get("success") or "").strip().lower()
            success = success_raw in ("true", "1", "t", "yes")
            if only_success and not success:
                continue
            samples.append(JtlSample(label=label, ts_ms=ts, elapsed_ms=elapsed, success=success))
    return samples


def _normalize_span_record(rec: Dict[str, Any]) -> Optional[Span]:
    # Try common OTEL JSON exporter shapes.
    # We accept start/end in either nanos or micros; we normalize to micros.
    trace_id = str(rec.get("traceId") or rec.get("trace_id") or rec.get("trace_id_hex") or "")
    span_id = str(rec.get("spanId") or rec.get("span_id") or rec.get("span_id_hex") or "")
    name = str(rec.get("name") or rec.get("spanName") or rec.get("operationName") or "")

    attrs = rec.get("attributes")
    if not isinstance(attrs, dict):
        # sometimes flattened
        attrs = {k: v for k, v in rec.items() if k.startswith("attr.")}

    start_ns = _to_int(rec.get("startTimeUnixNano") or rec.get("start_time_unix_nano"))
    end_ns = _to_int(rec.get("endTimeUnixNano") or rec.get("end_time_unix_nano"))
    if start_ns is not None and end_ns is not None:
        return Span(trace_id=trace_id, span_id=span_id, start_us=start_ns // 1000, end_us=end_ns // 1000, attrs=attrs, name=name)

    start_us = _to_int(rec.get("startTimeUnixMicro") or rec.get("start_time_unix_micro") or rec.get("startTime"))
    end_us = _to_int(rec.get("endTimeUnixMicro") or rec.get("end_time_unix_micro") or rec.get("endTime"))
    if start_us is not None and end_us is not None:
        return Span(trace_id=trace_id, span_id=span_id, start_us=start_us, end_us=end_us, attrs=attrs, name=name)

    # Some exporters store ISO timestamps; unsupported for now.
    return None


def iter_json_objects(path: str) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a file.

    Supports:
    - JSON array (top-level list)
    - JSONL/NDJSON (one object per line)
    - Concatenated JSON objects

    This is the trace reader; it yields only dict objects.
    """

    def _iter_values(p: str) -> Iterator[Any]:
        with open(p, "rb") as fb:
            prefix = fb.read(4096).lstrip()

        if prefix.startswith(b"["):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data
            return

        decoder = json.JSONDecoder()
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            buf = ""
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.lstrip()
                    if not s:
                        buf = ""
                        break
                    try:
                        obj, idx = decoder.raw_decode(s)
                    except json.JSONDecodeError:
                        buf = s
                        break
                    yield obj
                    buf = s[idx:]

            tail = buf.strip()
            if tail:
                try:
                    obj, _ = decoder.raw_decode(tail)
                    yield obj
                except json.JSONDecodeError:
                    return

    for obj in _iter_values(path):
        if isinstance(obj, dict):
            yield obj


def iter_otel_spans(obj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield span-like dicts from an OTEL JSON envelope or a flat span record."""

    if not isinstance(obj, dict):
        return

    # Flat span record
    if "startTimeUnixNano" in obj or "endTimeUnixNano" in obj or "traceId" in obj:
        attrs = obj.get("attributes")
        if isinstance(attrs, list):
            attrs_out: Dict[str, Any] = {}
            for a in attrs:
                if not isinstance(a, dict):
                    continue
                k = a.get("key")
                v = a.get("value")
                if k is None:
                    continue
                attrs_out[str(k)] = _unwrap_otel_anyvalue(v)
            flat = dict(obj)
            flat["attributes"] = attrs_out
            yield flat
        else:
            yield obj
        return

    # OTEL traces JSON envelope
    rs = obj.get("resourceSpans")
    if not isinstance(rs, list):
        return

    for r in rs:
        if not isinstance(r, dict):
            continue

        resource_attrs: Dict[str, Any] = {}
        resource = r.get("resource")
        if isinstance(resource, dict):
            attrs = resource.get("attributes")
            if isinstance(attrs, list):
                for a in attrs:
                    if not isinstance(a, dict):
                        continue
                    k = a.get("key")
                    v = a.get("value")
                    if k is None:
                        continue
                    resource_attrs[str(k)] = _unwrap_otel_anyvalue(v)

        scope_spans = r.get("scopeSpans") or r.get("instrumentationLibrarySpans")
        if not isinstance(scope_spans, list):
            continue

        for ss in scope_spans:
            if not isinstance(ss, dict):
                continue

            scope = ss.get("scope") or ss.get("instrumentationLibrary")
            scope_name = None
            if isinstance(scope, dict):
                scope_name = scope.get("name")

            spans = ss.get("spans")
            if not isinstance(spans, list):
                continue

            for sp in spans:
                if not isinstance(sp, dict):
                    continue

                attrs_out: Dict[str, Any] = {k: _unwrap_otel_anyvalue(v) for k, v in resource_attrs.items()}
                if scope_name:
                    attrs_out.setdefault("otel.scope.name", scope_name)

                sattrs = sp.get("attributes")
                if isinstance(sattrs, list):
                    for a in sattrs:
                        if not isinstance(a, dict):
                            continue
                        k = a.get("key")
                        v = a.get("value")
                        if k is None:
                            continue
                        attrs_out[str(k)] = _unwrap_otel_anyvalue(v)

                flat = dict(sp)
                flat["attributes"] = attrs_out
                yield flat


def _find_trace_id_in_log_record(rec: Any) -> Optional[str]:
    """Best-effort extraction of traceId from a log record.

    Supports common shapes:
    - Flat fields: traceId / trace_id / trace_id_hex
    - Nested fields: attributes.trace_id, resource.attributes... (rare)
    - W3C traceparent: 'traceparent' header value
    """

    if not isinstance(rec, dict):
        return None

    # direct keys
    for k in ("traceId", "trace_id", "trace_id_hex", "traceid", "traceID"):
        v = rec.get(k)
        if v:
            return str(v)

    # nested common keys
    attrs = rec.get("attributes")
    if isinstance(attrs, dict):
        for k in ("traceId", "trace_id", "trace_id_hex", "traceid"):
            v = attrs.get(k)
            if v:
                return str(v)

        tp = attrs.get("traceparent")
        if tp and isinstance(tp, str):
            tid = _trace_id_from_traceparent(tp)
            if tid:
                return tid

    # sometimes 'traceparent' is on root
    tp2 = rec.get("traceparent")
    if tp2 and isinstance(tp2, str):
        tid = _trace_id_from_traceparent(tp2)
        if tid:
            return tid

    return None


def _trace_id_from_traceparent(tp: str) -> Optional[str]:
    # W3C traceparent: version-traceid-spanid-flags
    # Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    parts = tp.split("-")
    if len(parts) >= 4:
        trace_id = parts[1].strip()
        if len(trace_id) == 32:
            return trace_id
    return None


def _extract_logger_hint(rec: Any) -> Optional[str]:
    """Try to extract a class/logger hint from a log record."""

    if not isinstance(rec, dict):
        return None

    # common fields
    for k in ("loggerName", "logger", "logger.name", "source", "class", "className"):
        v = rec.get(k)
        if v:
            return str(v)

    # nested
    attrs = rec.get("attributes")
    if isinstance(attrs, dict):
        for k in ("loggerName", "logger", "logger.name", "source", "class", "className"):
            v = attrs.get(k)
            if v:
                return str(v)

    body = rec.get("body")
    if isinstance(body, dict):
        for k in ("stringValue", "text", "message"):
            v = body.get(k)
            if v and isinstance(v, str):
                # no strong parse here; just signal that logs have payload
                return "<message-present>"

    msg = rec.get("message")
    if isinstance(msg, str) and msg.strip():
        return "<message-present>"

    return None


def load_spans(traces_path: str) -> List[Span]:
    spans: List[Span] = []

    for obj in iter_json_objects(traces_path):
        for rec in iter_otel_spans(obj):
            if not isinstance(rec, dict):
                continue
            s = _normalize_span_record(rec)
            if s is None:
                continue
            if s.start_us <= 0 or s.end_us <= 0 or s.end_us < s.start_us:
                continue
            spans.append(s)

    return spans


def _span_within_window(span: Span, start_us: int, end_us: int, *, slack_us: int = 0) -> bool:
    return (span.start_us >= start_us - slack_us) and (span.end_us <= end_us + slack_us)


def _span_overlaps_window(span: Span, start_us: int, end_us: int, *, slack_us: int = 0) -> bool:
    # Overlap if they intersect (with optional slack)
    a0, a1 = span.start_us, span.end_us
    b0, b1 = start_us - slack_us, end_us + slack_us
    return (a0 <= b1) and (a1 >= b0)


def _match_span(span: Span, start_us: int, end_us: int, *, slack_us: int, mode: str) -> bool:
    if mode == "contain":
        return _span_within_window(span, start_us, end_us, slack_us=slack_us)
    if mode == "overlap":
        return _span_overlaps_window(span, start_us, end_us, slack_us=slack_us)
    raise ValueError(f"Unknown match mode: {mode}")


def alignment_stats(samples: List[JtlSample], spans: List[Span], *, slack_ms: int = 50, match_mode: str = "contain") -> Dict[str, Any]:
    slack_us = slack_ms * 1000
    total = len(samples)
    if total == 0:
        return {"total": 0, "aligned": 0, "aligned_ratio": 0.0}

    aligned = 0
    # naive O(n*m) is ok for quick validation; can optimize later.
    for s in samples:
        w0, w1 = s.start_us, s.end_us
        ok = any(_match_span(sp, w0, w1, slack_us=slack_us, mode=match_mode) for sp in spans)
        if ok:
            aligned += 1

    return {
        "total": total,
        "aligned": aligned,
        "aligned_ratio": aligned / total,
        "slack_ms": slack_ms,
        "match_mode": match_mode,
    }


def extract_backend_class_names(spans: List[Span]) -> List[str]:
    # Prefer code.namespace / code.function (Java class+method) if present.
    names: List[str] = []
    for sp in spans:
        attrs = sp.attrs or {}

        ns = _attr_get(attrs, "code.namespace")
        fn = _attr_get(attrs, "code.function")
        if ns and fn:
            names.append(f"{ns}.{fn}")
            continue
        if ns:
            names.append(str(ns))
            continue

        # Common alternates
        cls = _attr_get(attrs, "code.class")
        m = _attr_get(attrs, "code.method")
        if cls and m:
            names.append(f"{cls}.{m}")
            continue
        if cls:
            names.append(str(cls))
            continue

        scope = _attr_get(attrs, "otel.scope.name")
        if scope:
            names.append(f"otel.scope.name={scope}")

        route = _attr_get(attrs, "http.route")
        if route:
            names.append(f"http.route={route}")

        svc = _attr_get(attrs, "service.name")
        if svc:
            names.append(f"service.name={svc}")

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _unwrap_otel_anyvalue(v: Any) -> Any:
    """Unwrap OTEL AnyValue-ish dicts (stringValue/intValue/arrayValue/...)

    The file exporter commonly uses {'stringValue': 'x'} etc.
    """

    if isinstance(v, dict) and v:
        # common shape: {"stringValue": "..."} 
        if len(v) == 1:
            k0, v0 = next(iter(v.items()))
            if k0.endswith("Value"):
                if k0 == "arrayValue" and isinstance(v0, dict) and "values" in v0:
                    return [_unwrap_otel_anyvalue(x) for x in (v0.get("values") or [])]
                if k0 == "kvlistValue" and isinstance(v0, dict) and "values" in v0:
                    out = {}
                    for item in v0.get("values") or []:
                        if isinstance(item, dict) and "key" in item and "value" in item:
                            out[str(item["key"])] = _unwrap_otel_anyvalue(item["value"])
                    return out
                return v0
    return v


def _attr_get(attrs: Dict[str, Any], key: str) -> Any:
    if not attrs:
        return None
    if key not in attrs:
        return None
    return _unwrap_otel_anyvalue(attrs.get(key))


def summarize_call_chain(spans: List[Span]) -> Dict[str, Any]:
    """Summarize a transaction window into a compact call-chain signature."""

    # sort by time to approximate nested chain
    spans_sorted = sorted(spans, key=lambda s: (s.start_us, s.end_us))

    routes: List[str] = []
    scopes: List[str] = []
    services: List[str] = []
    code: List[str] = []

    for sp in spans_sorted:
        attrs = sp.attrs or {}
        r = _attr_get(attrs, "http.route")
        if r:
            routes.append(str(r))
        sc = _attr_get(attrs, "otel.scope.name")
        if sc:
            scopes.append(str(sc))
        sv = _attr_get(attrs, "service.name")
        if sv:
            services.append(str(sv))
        ns = _attr_get(attrs, "code.namespace")
        fn = _attr_get(attrs, "code.function")
        if ns and fn:
            code.append(f"{ns}.{fn}")

    def _uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return {
        "span_count": len(spans_sorted),
        "services": _uniq(services),
        "routes": _uniq(routes),
        "scopes": _uniq(scopes),
        "code": _uniq(code),
        # A stable-ish signature for coverage stats
        "signature": "|".join(_uniq(routes) + _uniq(scopes)),
    }


def spans_in_window(spans: List[Span], start_us: int, end_us: int, *, slack_ms: int = 50, match_mode: str = "contain") -> List[Span]:
    slack_us = slack_ms * 1000
    return [sp for sp in spans if _match_span(sp, start_us, end_us, slack_us=slack_us, mode=match_mode)]


def group_spans_by_trace_id(spans: List[Span]) -> Dict[str, List[Span]]:
    out: Dict[str, List[Span]] = {}
    for sp in spans:
        tid = sp.trace_id or "<no-trace-id>"
        out.setdefault(tid, []).append(sp)
    return out


def spans_matching_route(spans: List[Span], route: str) -> List[Span]:
    out: List[Span] = []
    for sp in spans:
        r = _attr_get(sp.attrs or {}, "http.route")
        if r and str(r) == route:
            out.append(sp)
    return out


def select_best_trace_for_route(win_spans: List[Span], target_route: str) -> Optional[Tuple[str, List[Span]]]:
    """Pick the traceId that most likely represents the target business route.

    Strategy:
    - Find traceIds that contain at least one span with http.route == target_route
    - For each candidate traceId, score by (#spans in that trace within window, then #route hits)
    """

    by_tid = group_spans_by_trace_id(win_spans)
    candidates: List[Tuple[int, int, str]] = []  # (span_count, route_hits, tid)

    for tid, sps in by_tid.items():
        hits = 0
        for sp in sps:
            r = _attr_get(sp.attrs or {}, "http.route")
            if r and str(r) == target_route:
                hits += 1
        if hits > 0:
            candidates.append((len(sps), hits, tid))

    if not candidates:
        return None

    # choose highest span_count, then highest route_hits
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    tid = candidates[0][2]
    return tid, by_tid.get(tid, [])


# -----------------------------
# Global attribute scan helpers
# -----------------------------

def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(v)
    except Exception:
        return "<unprintable>"


def scan_code_attributes(
    traces_path: str,
    *,
    top_n: int = 30,
    include_routes: bool = True,
    include_scopes: bool = True,
    include_services: bool = True,
) -> Dict[str, Any]:
    """Scan the whole trace file and report whether code.* attributes exist.

    This is meant to be a cheap, objective "stop/continue" signal for Phase 2 depth:
    - If code.namespace/code.function never appear, method-level chaining from traces is unlikely.
    - If they appear only for a subset of services/scopes, you can focus instrumentation there.

    Returns a dict suitable for printing/logging.
    """

    total_spans = 0
    spans_with_any_code = 0
    spans_with_code_ns = 0
    spans_with_code_fn = 0
    spans_with_code_class = 0
    spans_with_code_method = 0

    # small groupings to indicate where code.* exists
    by_service: Dict[str, int] = {}
    by_scope: Dict[str, int] = {}
    by_route: Dict[str, int] = {}
    by_ns: Dict[str, int] = {}

    def _inc(d: Dict[str, int], k: str) -> None:
        if not k:
            k = "<missing>"
        d[k] = d.get(k, 0) + 1

    for obj in iter_json_objects(traces_path):
        for rec in iter_otel_spans(obj):
            if not isinstance(rec, dict):
                continue
            sp = _normalize_span_record(rec)
            if sp is None:
                continue
            if sp.start_us <= 0 or sp.end_us <= 0 or sp.end_us < sp.start_us:
                continue

            total_spans += 1
            attrs = sp.attrs or {}

            ns = _attr_get(attrs, "code.namespace")
            fn = _attr_get(attrs, "code.function")
            cls = _attr_get(attrs, "code.class")
            m = _attr_get(attrs, "code.method")

            has_any = any([ns, fn, cls, m])
            if has_any:
                spans_with_any_code += 1

                if ns:
                    spans_with_code_ns += 1
                    _inc(by_ns, _safe_str(ns))
                if fn:
                    spans_with_code_fn += 1
                if cls:
                    spans_with_code_class += 1
                if m:
                    spans_with_code_method += 1

                if include_services:
                    _inc(by_service, _safe_str(_attr_get(attrs, "service.name")))
                if include_scopes:
                    _inc(by_scope, _safe_str(_attr_get(attrs, "otel.scope.name")))
                if include_routes:
                    _inc(by_route, _safe_str(_attr_get(attrs, "http.route")))

    def _top(d: Dict[str, int]) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    return {
        "traces_path": traces_path,
        "total_spans": total_spans,
        "spans_with_any_code_attr": spans_with_any_code,
        "ratio_with_any_code_attr": (spans_with_any_code / total_spans) if total_spans else 0.0,
        "spans_with_code.namespace": spans_with_code_ns,
        "spans_with_code.function": spans_with_code_fn,
        "spans_with_code.class": spans_with_code_class,
        "spans_with_code.method": spans_with_code_method,
        "top_code.namespace": _top(by_ns),
        "top_service.name_with_code": _top(by_service),
        "top_otel.scope.name_with_code": _top(by_scope),
        "top_http.route_with_code": _top(by_route),
    }


def scan_logs_trace_id_coverage(
    logs_path: str,
    trace_ids: List[str],
    *,
    top_n: int = 30,
    max_logs: Optional[int] = None,
) -> Dict[str, Any]:
    """Given a list of traceIds (typically from matched JTL samples), check if logs contain them.

    Also reports whether logs provide usable class/logger hints for those traceIds.
    """

    trace_id_set = {t for t in trace_ids if t and t != "<no-trace-id>"}
    if not trace_id_set:
        return {
            "logs_path": logs_path,
            "input_trace_ids": len(trace_ids),
            "unique_trace_ids": 0,
            "total_log_records_scanned": 0,
            "matched_log_records": 0,
            "trace_ids_with_any_log": 0,
            "trace_ids_with_logger_hint": 0,
            "top_logger_hints": [],
        }

    total = 0
    matched_records = 0
    tid_has_log: Dict[str, bool] = {t: False for t in trace_id_set}
    tid_has_logger: Dict[str, bool] = {t: False for t in trace_id_set}
    logger_counts: Dict[str, int] = {}

    for item in iter_json_objects(logs_path):
        # records are dicts
        total += 1
        if max_logs is not None and total > max_logs:
            break

        tid = _find_trace_id_in_log_record(item)
        if not tid or tid not in trace_id_set:
            continue

        matched_records += 1
        tid_has_log[tid] = True

        hint = _extract_logger_hint(item)
        if hint:
            tid_has_logger[tid] = True
            logger_counts[hint] = logger_counts.get(hint, 0) + 1

    def _top(d: Dict[str, int]) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    return {
        "logs_path": logs_path,
        "input_trace_ids": len(trace_ids),
        "unique_trace_ids": len(trace_id_set),
        "total_log_records_scanned": total,
        "matched_log_records": matched_records,
        "trace_ids_with_any_log": sum(1 for v in tid_has_log.values() if v),
        "trace_ids_with_logger_hint": sum(1 for v in tid_has_logger.values() if v),
        "top_logger_hints": _top(logger_counts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jtl", required=True, help="Path to *_results.jtl (CSV with headers)")
    ap.add_argument(
        "--traces",
        default="data/processed/traces/all_traces.json",
        help="Path to all_traces.json (JSON array or JSONL)",
    )
    ap.add_argument(
        "--logs",
        default="data/processed/logs/all_logs.json",
        help="Path to all_logs.json (used only for --scan-logs).",
    )
    ap.add_argument("--slack-ms", type=int, default=50, help="Allowed slack when checking containment")
    ap.add_argument(
        "--match-mode",
        choices=["contain", "overlap"],
        default="contain",
        help="How to match spans to a JTL sample window. 'contain' requires full containment, 'overlap' requires any intersection.",
    )
    ap.add_argument("--target-label", default="T_SearchFlights", help="Transaction label to deep-dive for backend class extraction")
    ap.add_argument("--max-target-samples", type=int, default=20, help="Max number of target samples to inspect")
    ap.add_argument("--top-traces", type=int, default=3, help="For each target sample, print details for the top-N traceIds with most spans")
    ap.add_argument(
        "--target-route",
        default="/rest/api/flights/queryflights",
        help="For target-label deep dive, prefer traceIds that contain this http.route to reduce contamination.",
    )

    ap.add_argument(
        "--scan-code-attrs",
        action="store_true",
        help="Scan all traces and report whether code.* attributes exist (to decide whether deeper call-chain extraction is feasible).",
    )
    ap.add_argument("--scan-top", type=int, default=30, help="Top-N items to print in scan report")

    ap.add_argument(
        "--scan-logs",
        action="store_true",
        help="Evaluate whether logs can be linked to traces via traceId, and whether logs provide class/logger hints.",
    )
    ap.add_argument(
        "--scan-logs-max",
        type=int,
        default=0,
        help="Optional cap for number of log records to scan (0 = no cap).",
    )

    args = ap.parse_args()

    if not os.path.exists(args.traces):
        raise SystemExit(
            f"Trace file not found: {args.traces}\n"
            "Tip: OTEL collector writes to data/processed/traces/all_traces.json in this repo; "
            "or pass --traces <path>."
        )

    if args.scan_code_attrs:
        report = scan_code_attributes(args.traces, top_n=args.scan_top)
        print("[Scan] code.* attribute presence")
        for k in [
            "total_spans",
            "spans_with_any_code_attr",
            "ratio_with_any_code_attr",
            "spans_with_code.namespace",
            "spans_with_code.function",
            "spans_with_code.class",
            "spans_with_code.method",
        ]:
            print(f"  {k}: {report.get(k)}")

        def _print_top(title: str, items: List[Tuple[str, int]]) -> None:
            if not items:
                print(f"  {title}: <none>")
                return
            print(f"  {title}:")
            for name, cnt in items:
                print(f"    - {name}: {cnt}")

        _print_top("top_code.namespace", report.get("top_code.namespace") or [])
        _print_top("top_service.name_with_code", report.get("top_service.name_with_code") or [])
        _print_top("top_otel.scope.name_with_code", report.get("top_otel.scope.name_with_code") or [])
        _print_top("top_http.route_with_code", report.get("top_http.route_with_code") or [])
        return 0

    samples = load_jtl_samples(args.jtl, only_transactions=True, only_success=True)
    spans = load_spans(args.traces)

    # If requested, evaluate log linkage using traceIds from the currently configured target samples.
    if args.scan_logs:
        if not os.path.exists(args.logs):
            raise SystemExit(
                f"Log file not found: {args.logs}\n"
                "Tip: logs are expected at data/processed/logs/all_logs.json in this repo; "
                "or pass --logs <path>."
            )

        target_samples = [s for s in samples if s.label == args.target_label][: args.max_target_samples]
        trace_ids: List[str] = []
        for s in target_samples:
            win_spans = spans_in_window(spans, s.start_us, s.end_us, slack_ms=args.slack_ms, match_mode=args.match_mode)
            best = select_best_trace_for_route(win_spans, args.target_route)
            if best is not None:
                trace_ids.append(best[0])

        max_logs = None if args.scan_logs_max in (0, None) else int(args.scan_logs_max)
        rep = scan_logs_trace_id_coverage(args.logs, trace_ids, top_n=args.scan_top, max_logs=max_logs)

        print("[Scan] logs↔traces linkage via traceId")
        print(f"  target_label: {args.target_label}")
        print(f"  inspected_target_samples: {len(target_samples)}")
        print(f"  collected_trace_ids: {rep.get('unique_trace_ids')} (from best_trace_for_route)")
        print(f"  total_log_records_scanned: {rep.get('total_log_records_scanned')}")
        print(f"  matched_log_records: {rep.get('matched_log_records')}")
        print(f"  trace_ids_with_any_log: {rep.get('trace_ids_with_any_log')}/{rep.get('unique_trace_ids')}")
        print(f"  trace_ids_with_logger_hint: {rep.get('trace_ids_with_logger_hint')}/{rep.get('unique_trace_ids')}")

        top_hints = rep.get("top_logger_hints") or []
        if top_hints:
            print("  top_logger_hints:")
            for name, cnt in top_hints:
                print(f"    - {name}: {cnt}")
        else:
            print("  top_logger_hints: <none>")

        return 0

    print(f"JTL transactions (success only): {len(samples)}")
    print(f"Traces spans loaded: {len(spans)}")

    stats = alignment_stats(samples, spans, slack_ms=args.slack_ms, match_mode=args.match_mode)
    print("\n[Alignment summary]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # deep dive for a target transaction
    target_samples = [s for s in samples if s.label == args.target_label]
    print(f"\n[Target transaction] {args.target_label} samples={len(target_samples)}")

    target_aligned = 0
    target_zero = 0
    target_route_hits = 0
    target_best_trace_found = 0

    for i, s in enumerate(target_samples[: args.max_target_samples]):
        win_spans = spans_in_window(spans, s.start_us, s.end_us, slack_ms=args.slack_ms, match_mode=args.match_mode)
        if win_spans:
            target_aligned += 1
        else:
            target_zero += 1

        route_spans = spans_matching_route(win_spans, args.target_route)
        if route_spans:
            target_route_hits += 1

        best = select_best_trace_for_route(win_spans, args.target_route)
        if best is not None:
            target_best_trace_found += 1

        chain = summarize_call_chain(win_spans)
        classes = extract_backend_class_names(win_spans)

        print(
            f"\n  sample#{i+1}: window_ms=[{s.ts_ms}, {s.ts_ms + s.elapsed_ms}] "
            f"elapsed={s.elapsed_ms}ms matched_spans={len(win_spans)}"
        )
        print(f"    routes: {chain['routes']}")
        print(f"    scopes: {chain['scopes']}")
        print(f"    services: {chain['services']}")
        print(f"    route_hits({args.target_route}): {len(route_spans)}")

        if best is None:
            print("    best_trace_for_route: <none-found>")
        else:
            best_tid, best_spans = best
            best_chain = summarize_call_chain(best_spans)
            best_classes = extract_backend_class_names(best_spans)
            print(f"    best_trace_for_route: {best_tid} spans={len(best_spans)}")
            print(f"      best.routes: {best_chain['routes']}")
            print(f"      best.scopes: {best_chain['scopes']}")
            if best_classes:
                print("      best.backend_class_names:")
                for c in best_classes[:30]:
                    print(f"        - {c}")

        by_tid = group_spans_by_trace_id(win_spans)
        top = sorted(by_tid.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.top_traces]
        if top:
            print("    traceIds(top):")
            for tid, sps in top:
                sub = summarize_call_chain(sps)
                print(f"      - {tid}: spans={len(sps)} routes={sub['routes']} scopes={sub['scopes']}")

        if not classes:
            print("    backend_class_names: <none-found>")
        else:
            print("    backend_class_names:")
            for c in classes[:50]:
                print(f"      - {c}")

    if target_samples:
        n = min(len(target_samples), args.max_target_samples)
        print(
            f"\n[Target coverage] inspected={n} aligned={target_aligned} zero_spans={target_zero} "
            f"mode={args.match_mode} slack_ms={args.slack_ms}"
        )
        print(
            f"[Target route coverage] route={args.target_route} "
            f"route_hit_samples={target_route_hits}/{n} best_trace_found={target_best_trace_found}/{n}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
