from pathlib import Path
from typing import Dict, Set, Optional, Tuple
import json
import numpy as np


# Heuristic reverse mapping: URL / keywords / entities / collections -> class names
# Note: this is an initial version for AcmeAir; it can be maintained in a separate config file if needed.
URL_TO_CLASS: Dict[str, str] = {
    "/rest/api/login": "com.acmeair.web.LoginREST",
    "/rest/api/login/logout": "com.acmeair.web.LoginREST",
    "/rest/api/flights/queryflights": "com.acmeair.web.FlightsREST",
    "/rest/api/bookings/bookflights": "com.acmeair.web.BookingsREST",
    "/rest/api/customer/byid": "com.acmeair.web.CustomerREST",
    "/rest/info/loader/load": "com.acmeair.config.LoaderREST",
    "/rest/api/config/runtime": "com.acmeair.web.AppConfig",
}

# To improve precision when paths have containment relations, sort URL keys by length
# in descending order to implement a "longest-prefix-first" matching strategy.
_sorted_url_keys = sorted(URL_TO_CLASS.keys(), key=len, reverse=True)

KEYWORD_TO_CLASS: Dict[str, str] = {
    "bookFlights": "com.acmeair.service.BookingService",
    "getCustomerByUsername": "com.acmeair.service.CustomerService",
    "getTripFlights": "com.acmeair.service.FlightService",
    "loadCustomers": "com.acmeair.loader.CustomerLoader",
    "loadFlights": "com.acmeair.loader.FlightLoader",
    "MongoConnectionManager": "com.acmeair.morphia.services.util.MongoConnectionManager",
}

ENTITY_TO_CLASS: Dict[str, str] = {
    "booking": "com.acmeair.morphia.entities.BookingImpl",
    "customer": "com.acmeair.morphia.entities.CustomerImpl",
    "flight": "com.acmeair.morphia.entities.FlightImpl",
    "session": "com.acmeair.morphia.entities.CustomerSessionImpl",
}

# Enhanced mapping based on Mongo collection name: one collection may map to multiple related classes (Service + Impl)
COLLECTION_TO_CLASSES: Dict[str, list[str]] = {
    "customer": [
        "com.acmeair.service.CustomerService",
        "com.acmeair.morphia.services.CustomerServiceImpl",
    ],
    "flight": [
        "com.acmeair.service.FlightService",
        "com.acmeair.morphia.services.FlightServiceImpl",
    ],
    "booking": [
        "com.acmeair.service.BookingService",
        "com.acmeair.morphia.services.BookingServiceImpl",
    ],
    "airportCodeMapping": [
        "com.acmeair.service.FlightService",
    ],
}


def _extract_string_attributes(span: Dict) -> Dict[str, str]:
    """Helper: extract stringValue attributes from span.attributes into a flat dict."""
    attrs: Dict[str, str] = {}
    for a in span.get("attributes", []):
        key = a.get("key")
        v = a.get("value", {})
        if not key or not isinstance(v, dict):
            continue
        if "stringValue" in v:
            attrs[key] = v["stringValue"]
    return attrs


def _is_target_service(resource_attributes: Dict[str, str], service_name: str) -> bool:
    """Check if current resource belongs to the target service (by service.name)."""
    return resource_attributes.get("service.name", "") == service_name


def _extract_resource_string_attributes(resource: Dict) -> Dict[str, str]:
    """Extract stringValue fields from resource.attributes into a flat dict."""
    attrs: Dict[str, str] = {}
    for a in resource.get("attributes", []):
        key = a.get("key")
        v = a.get("value", {})
        if not key or not isinstance(v, dict):
            continue
        if "stringValue" in v:
            attrs[key] = v["stringValue"]
    return attrs


def resolve_class_indices(
    span_attrs: Dict[str, str],
    span_name: str,
    log_body: str,
    class_to_idx: Dict[str, int],
) -> Set[int]:
    """Resolve multiple class indices from a span using heuristic evidence.

    This is safer than returning a single idx because many apps only emit coarse
    servlet/JSP spans; we intentionally *expand* one request into a small set of
    entrypoint+domain+dao/entity classes to densify temporal co-occurrence.
    """
    out: Set[int] = set()

    def _add_fqcn_list(fqcns: list[str]) -> None:
        for fqcn in fqcns:
            idx = class_to_idx.get(fqcn)
            if idx is not None:
                out.add(idx)

    def _extract_query_action_key(attrs: Dict[str, str]) -> str:
        """Best-effort extraction of a 'controller/action' key from OpenTelemetry url.query.

        OpenTelemetry may provide:
        - a raw query string in 'url.query' (e.g., 'viewItem=&itemId=EST-1')
        - a parsed JSON string in 'url.query' (e.g., '{"viewItem":"","itemId":"EST-1"}')

        We normalize to the *first* query key (action) and strip '_', '-' then lowercase.
        """
        raw = (attrs.get("url.query", "") or "").strip()
        if not raw:
            return ""

        # Case A: parsed JSON string
        if raw.startswith("{") and raw.endswith("}"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj:
                    k0 = next(iter(obj.keys()))
                    return str(k0).strip()
            except Exception:
                pass

        # Case B: regular query string
        # Prefer first key before '&' and before '='
        tok = raw.split("&", 1)[0].strip()
        if not tok:
            return ""
        return tok.split("=", 1)[0].strip()

    # 0) Existing AcmeAir URL longest-prefix mapping (single class)
    target = (
        span_attrs.get("http.route")
        or span_attrs.get("url.path")
        or span_attrs.get("url.full")
        or ""
    )

    # Also keep a concrete path for frameworks that set http.route to wildcards (e.g. /foo/*.jsf)
    url_path = (span_attrs.get("url.path") or "").strip()

    if target:
        for path in _sorted_url_keys:
            if path in target:
                cls = URL_TO_CLASS[path]
                idx = class_to_idx.get(cls)
                if idx is not None:
                    out.add(idx)
                break

    # 1) JPetStore actions from url.query
    action_key = _extract_query_action_key(span_attrs)
    if action_key:
        action_norm = action_key.replace("_", "").replace("-", "").lower()
        mapper = globals().get("JPETSTORE_ACTION_TO_CONTROLLER")
        if isinstance(mapper, dict):
            mapped = mapper.get(action_norm)
            if mapped:
                _add_fqcn_list(mapped)

    # 2) JPetStore JSP internal span name anchors
    sname = (span_name or "").lower()
    jsp_mapper = globals().get("JPETSTORE_JSP_TO_CONTROLLER")
    if isinstance(jsp_mapper, dict):
        for jsp_key, mapped in jsp_mapper.items():
            if jsp_key in sname:
                _add_fqcn_list(mapped)

    # 3) Plants: match against concrete url.path first (http.route is often a wildcard)
    plants_match_src = url_path or target
    plants_mapper = globals().get("PLANTS_PATH_TO_CLASSES")
    if plants_match_src and isinstance(plants_mapper, dict):
        for pfx, mapped in plants_mapper.items():
            if pfx.lower() in plants_match_src.lower():
                _add_fqcn_list(mapped)

    # 4) DayTrader: prefer query-driven action mapping (url.path is too coarse)
    q = (span_attrs.get("url.query", "") or "").strip()
    action = ""

    # Robust token-based extraction (handles both 'action=register' and 'actionregister')
    if q:
        for tok in q.split("&"):
            tok = tok.strip()
            if not tok:
                continue
            if not tok.lower().startswith("action"):
                continue
            if "=" in tok:
                action = tok.split("=", 1)[1].strip()
            else:
                action = tok[len("action"):].strip()
            break

    if q and not action:
        qp = _parse_query_params(q)
        # true canonical form: action=marketSummary
        action = (qp.get("action") or "").strip()
        if not action:
            # fallback: concatenated token like 'actionlogin' or 'actionmarketSummary'
            for k in qp.keys():
                if k.startswith("action") and len(k) > len("action"):
                    action = k[len("action"):]

    if not action:
        # fallback: infer from span.name/http.route/url.path when query is absent
        hint_text = " ".join(
            [
                span_name or "",
                span_attrs.get("http.route", "") or "",
                span_attrs.get("url.path", "") or "",
            ]
        )
        infer_fn = globals().get("_infer_daytrader_action_from_text")
        if callable(infer_fn):
            action = infer_fn(hint_text)

    action_norm = (action or "").replace("_", "").replace("-", "").lower()
    dt_action_map = globals().get("DAYTRADER_ACTION_TO_CLASSES")
    if isinstance(dt_action_map, dict):
        mapped = dt_action_map.get(action_norm)
        if mapped:
            _add_fqcn_list(mapped)

    # 4b) DayTrader: keep legacy conservative hint scan as a backstop
    hint_src = (span_attrs.get("url.query", "") or "") + " " + (url_path or target or "")
    hint_src = hint_src.lower()
    dt_hint_map = globals().get("DAYTRADER_HINT_TO_CLASSES")
    if isinstance(dt_hint_map, dict):
        for hint, mapped in dt_hint_map.items():
            if hint and hint in hint_src:
                _add_fqcn_list(mapped)

    # 5) Mongo collection expansion (AcmeAir)
    collection = span_attrs.get("db.mongodb.collection", "")
    if collection and collection in COLLECTION_TO_CLASSES:
        _add_fqcn_list(COLLECTION_TO_CLASSES[collection])

    # 6) Keyword in span.name (AcmeAir)
    name = span_name or ""
    for kw, cls in KEYWORD_TO_CLASS.items():
        if kw and kw in name:
            idx = class_to_idx.get(cls)
            if idx is not None:
                out.add(idx)

    # 7) Entity word in db.statement / log body (AcmeAir)
    combined = (span_attrs.get("db.statement", "") + (log_body or "")).lower()
    if combined:
        for ent, cls in ENTITY_TO_CLASS.items():
            if ent and ent in combined:
                idx = class_to_idx.get(cls)
                if idx is not None:
                    out.add(idx)

    return out


def resolve_class_index(
    span_attrs: Dict[str, str],
    span_name: str,
    log_body: str,
    class_to_idx: Dict[str, int],
) -> Optional[int]:
    """Backward-compatible single-index resolver.

    Prefer `resolve_class_indices` for trace usage.
    """
    idxs = resolve_class_indices(span_attrs, span_name, log_body, class_to_idx)
    if not idxs:
        return None
    return next(iter(idxs))


def extract_classes_from_logs(
    log_json_path: Path,
    class_to_idx: Dict[str, int],
    service_name: str = "acmeair",
) -> Dict[str, Set[int]]:
    """Extract traceId -> {class_index} from OTLP logs.

    Key points:
    - Use scope.name as a high-signal hint to map logger FQCN to a class.
    - If the log body contains entity keywords, augment the mapping via ENTITY_TO_CLASS.
    - Keep only classes that exist in class_to_idx.
    """
    trace_class_map: Dict[str, Set[int]] = {}

    if not log_json_path.exists():
        return trace_class_map

    with log_json_path.open("r", encoding="utf-8") as lf:
        for line in lf:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for resource_log in data.get("resourceLogs", []):
                resource = resource_log.get("resource", {}) or {}
                r_attrs = _extract_resource_string_attributes(resource)
                if not _is_target_service(r_attrs, service_name):
                    # Keep only logs from the target service
                    continue

                for scope_log in resource_log.get("scopeLogs", []):
                    scope = scope_log.get("scope", {}) or {}
                    scope_name = scope.get("name", "")

                    for record in scope_log.get("logRecords", []):
                        trace_id = record.get("traceId")
                        if not trace_id:
                            continue
                        body = ""
                        body_val = record.get("body", {})
                        if isinstance(body_val, dict):
                            body = body_val.get("stringValue", "") or ""

                        idx: Optional[int] = None
                        if scope_name:
                            idx = class_to_idx.get(scope_name)
                        if idx is None and body:
                            # reuse ENTITY_TO_CLASS heuristic if possible
                            lower_body = body.lower()
                            for ent, cls in ENTITY_TO_CLASS.items():
                                if ent and ent in lower_body:
                                    cand = class_to_idx.get(cls)
                                    if cand is not None:
                                        idx = cand
                                        break

                        if idx is None:
                            continue

                        trace_class_map.setdefault(trace_id, set()).add(idx)

    return trace_class_map


def extract_classes_from_traces(
    trace_json_path: Path,
    class_to_idx: Dict[str, int],
    service_name: str = "acmeair",
    *,
    debug: bool = False,
    debug_topk: int = 10,
    debug_sample_spans: int = 0,
) -> Dict[str, Set[int]]:
    """Extract traceId -> {class_index} set from OTEL trace JSON (basic version).

    Improvements (2026-02):
    - If the caller does not provide service_name, auto-infer it from the filename
      (e.g., plantsbywebsphere.json -> plantsbywebsphere).
    - Apply lightweight noise filtering to reduce cross-cutting framework spans.

    Debug (optional):
    - Prints service.name frequency TopK observed in the trace file
    - Prints resource service_name match / mismatch counts
    - Prints span mapping hit-rate (resolve_class_indices empty vs non-empty)

    Notes:
    - OTLP exporter outputs are typically one JSON object per line (NDJSON); we parse line-by-line.
    - Prefer span.attributes["code.namespace"] as a direct FQCN signal when present.
    - Fall back to other common fields (e.g., rpc.service).
    - Keep only classes that exist in class_to_idx.
    """
    trace_map: Dict[str, Set[int]] = {}

    # Auto-infer service.name from filename when not explicitly set
    if not service_name and trace_json_path and trace_json_path.name.endswith(".json"):
        service_name = trace_json_path.stem

    if not trace_json_path.exists():
        if debug:
            print(f"[trace-debug] missing file: {trace_json_path.as_posix()}")
        return trace_map

    # debug counters
    service_freq: Dict[str, int] = {}
    span_attr_key_freq: Dict[str, int] = {}
    resource_total = 0
    resource_match = 0
    resource_mismatch = 0
    span_total = 0
    span_traceid_missing = 0
    span_hit = 0
    span_miss = 0

    # Collect a few samples for printing when debug_sample_spans>0
    samples: list[dict] = []

    def _tally_attr_keys(span: Dict) -> None:
        for a in span.get("attributes", []) or []:
            k = a.get("key")
            if not k:
                continue
            span_attr_key_freq[k] = span_attr_key_freq.get(k, 0) + 1

    def _pick_prefixed(attrs: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k, v in attrs.items():
            if k.startswith(("http.", "url.", "db.", "rpc.", "code.", "otel.", "net.", "server.", "client.")):
                out[k] = v
        return out

    def _is_noise_fqcn_local(fqcn: str) -> bool:
        # Keep this conservative: only drop obvious framework glue that tends to connect everything.
        s = (fqcn or "").lower()
        return any(k in s for k in (
            "java.",
            "sun.",
        ))

    with trace_json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for resource_span in data.get("resourceSpans", []):
                resource_total += 1
                resource = resource_span.get("resource", {}) or {}
                r_attrs = _extract_resource_string_attributes(resource)
                sn = (r_attrs.get("service.name") or "").strip()
                if sn:
                    service_freq[sn] = service_freq.get(sn, 0) + 1

                if service_name and not _is_target_service(r_attrs, service_name):
                    resource_mismatch += 1
                    continue

                resource_match += 1

                for scope_span in resource_span.get("scopeSpans", []):
                    scope = scope_span.get("scope", {}) or {}
                    scope_name = (scope.get("name") or "").strip()

                    for span in scope_span.get("spans", []):
                        span_total += 1
                        _tally_attr_keys(span)

                        span_attrs = _extract_string_attributes(span)
                        tid = span.get("traceId")
                        if not tid:
                            span_traceid_missing += 1
                            continue

                        if debug and int(debug_sample_spans) > 0 and len(samples) < int(debug_sample_spans):
                            samples.append(
                                {
                                    "scope.name": scope_name,
                                    "name": span.get("name", ""),
                                    "attr_keys": [a.get("key") for a in (span.get("attributes", []) or []) if a.get("key")],
                                    "attr_prefixed": _pick_prefixed(span_attrs),
                                }
                            )

                        # 1) direct FQCN when present
                        fqcn = span_attrs.get("code.namespace") or span_attrs.get("rpc.service") or ""
                        if fqcn and _is_noise_fqcn_local(fqcn):
                            fqcn = ""

                        idxs: Set[int] = set()
                        if fqcn:
                            direct = class_to_idx.get(fqcn)
                            if direct is not None:
                                idxs.add(direct)

                        # 2) heuristic expansion (url.query/url.path/jsp span names/etc)
                        idxs.update(resolve_class_indices(span_attrs, span.get("name", ""), "", class_to_idx))

                        if not idxs:
                            span_miss += 1
                            continue

                        span_hit += 1
                        trace_map.setdefault(tid, set()).update(idxs)

    if debug:
        # service.name TopK
        top = sorted(service_freq.items(), key=lambda kv: kv[1], reverse=True)[: int(debug_topk)]
        top_str = ", ".join([f"{k}={v}" for k, v in top]) if top else "(none)"

        # attribute keys TopK
        key_top = sorted(span_attr_key_freq.items(), key=lambda kv: kv[1], reverse=True)[: int(debug_topk)]
        key_top_str = ", ".join([f"{k}={v}" for k, v in key_top]) if key_top else "(none)"

        # compute hit rate
        denom = max(1, span_hit + span_miss)
        hit_rate = span_hit / float(denom)

        print(f"[trace-debug] file={trace_json_path.as_posix()} target_service={service_name!r}")
        print(f"[trace-debug] resources: total={resource_total} match={resource_match} mismatch={resource_mismatch}")
        print(f"[trace-debug] service.name Top{int(debug_topk)}: {top_str}")
        print(f"[trace-debug] attr.keys Top{int(debug_topk)}: {key_top_str}")
        print(
            f"[trace-debug] spans: total={span_total} traceId_missing={span_traceid_missing} "
            f"mapped(hit)={span_hit} unmapped(miss)={span_miss} hit_rate={hit_rate:.3f}"
        )
        print(f"[trace-debug] traces(out)={len(trace_map)}")

        if int(debug_sample_spans) > 0:
            print(f"[trace-debug] sample_spans={int(debug_sample_spans)} (first spans observed after service.name filter)")
            for i, s in enumerate(samples):
                keys = s.get("attr_keys", [])
                pref = s.get("attr_prefixed", {})
                print(f"  [span {i}] scope.name={s.get('scope.name','')} name={s.get('name','')}")
                print(f"    keys({len(keys)}): {keys[:40]}" + (" ..." if len(keys) > 40 else ""))
                if pref:
                    # print only a few to keep logs readable
                    pref_items = list(pref.items())[:20]
                    print(f"    prefixed({len(pref)}): " + ", ".join([f"{k}={v}" for k, v in pref_items]))

    return trace_map


def extract_classes_from_traces_hybrid(
    trace_json_path: Path,
    class_to_idx: Dict[str, int],
    log_json_path: Optional[Path] = None,
    service_name: str = "acmeair",
) -> Dict[str, Set[int]]:
    """Hybrid extraction based on OTEL traces (+ optional logs): traceId -> {class_index}.

    - Augment trace-side mapping via heuristic reverse mapping (URL/collection/keywords/entities).
    - Use the high-signal logger FQCN from log scope.name, and merge via traceId.
    - Keep only span/log records whose service.name matches the target service to avoid cross-service noise.
    """
    # 1) Build traceId -> {class_index} from logs first (scope.name / body heuristics)
    log_trace_map: Dict[str, Set[int]] = {}
    if log_json_path is not None:
        log_trace_map = extract_classes_from_logs(log_json_path, class_to_idx, service_name=service_name)

    # 2) Parse traces, collect class indices, and merge with log-side results
    trace_map: Dict[str, Set[int]] = {}

    if trace_json_path.exists():
        with trace_json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for resource_span in data.get("resourceSpans", []):
                    resource = resource_span.get("resource", {}) or {}
                    r_attrs = _extract_resource_string_attributes(resource)
                    if not _is_target_service(r_attrs, service_name):
                        continue

                    for scope_span in resource_span.get("scopeSpans", []):
                        for span in scope_span.get("spans", []):
                            span_attrs = _extract_string_attributes(span)
                            tid = span.get("traceId")
                            if not tid:
                                continue

                            fqcn = span_attrs.get("code.namespace") or span_attrs.get("rpc.service") or ""
                            idx: Optional[int] = None
                            if fqcn:
                                idx = class_to_idx.get(fqcn)
                            if idx is None:
                                idx = resolve_class_index(span_attrs, span.get("name", ""), "", class_to_idx)
                            if idx is None:
                                continue

                            trace_map.setdefault(tid, set()).add(idx)

    # 3) Merge with the log trace_map
    for tid, cls_set in log_trace_map.items():
        if not cls_set:
            continue
        trace_map.setdefault(tid, set()).update(cls_set)

    return trace_map


def extract_trace_start_times(trace_json_path: Path) -> Dict[str, int]:
    """Extract mapping: traceId -> earliest startTimeUnixNano from a trace NDJSON file."""
    trace_time_map: Dict[str, int] = {}

    if not trace_json_path.exists():
        return trace_time_map

    with trace_json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for resource_span in data.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    for span in scope_span.get("spans", []):
                        tid = span.get("traceId")
                        if not tid:
                            continue
                        start = span.get("startTimeUnixNano")
                        if not isinstance(start, int):
                            continue
                        if tid not in trace_time_map or start < trace_time_map[tid]:
                            trace_time_map[tid] = start

    return trace_time_map


def calculate_s_temp(trace_map: Dict[str, Set[int]], num_classes: int) -> np.ndarray:
    """Build S_temp similarity matrix (Jaccard) from class co-occurrence in traces/sessions.

    Accepted input shapes:
    - traceId -> {class_index}
    - sessionId/threadName -> {class_index}

    Output is an N x N matrix aligned with class_order.
    """
    co_occurrence = np.zeros((num_classes, num_classes), dtype=np.float32)
    individual_occurrence = np.zeros(num_classes, dtype=np.float32)

    for indices in trace_map.values():
        if not indices:
            continue
        idx_list = list(indices)
        for i in idx_list:
            if 0 <= i < num_classes:
                individual_occurrence[i] += 1.0
        for i in range(len(idx_list)):
            for j in range(len(idx_list)):
                a = idx_list[i]
                b = idx_list[j]
                if 0 <= a < num_classes and 0 <= b < num_classes:
                    co_occurrence[a, b] += 1.0

    s_temp = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        for j in range(num_classes):
            inter = co_occurrence[i, j]
            if inter <= 0:
                continue
            union = individual_occurrence[i] + individual_occurrence[j] - inter
            if union <= 0:
                continue
            s_temp[i, j] = inter / union

    np.fill_diagonal(s_temp, 1.0)
    return s_temp


def _parse_query_params(q: str) -> Dict[str, str]:
    """Parse a URL query string into a dict.

    Supports both standard form (a=b&c=d) and DayTrader-ish concatenations (e.g.
    'actionlogin&uid=...'), by treating bare tokens as keys with empty values.
    """
    out: Dict[str, str] = {}
    if not q:
        return out
    q = q.strip().lstrip("?")
    for part in q.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
        else:
            out[part.strip().lower()] = ""
    return out


# DayTrader (minimal) action/hint -> class mapping for trace extraction.
# These FQCNs MUST match strings in data/processed/fusion/daytrader_class_order.json.
DAYTRADER_ACTION_TO_CLASSES: Dict[str, list[str]] = {
    # Most actions route through TradeAppServlet; expand with core service layer
    "home": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.ejb3.TradeSLSBBean",
        "com.ibm.websphere.samples.daytrader.ejb3.TradeSLSBLocal",
        "com.ibm.websphere.samples.daytrader.ejb3.TradeSLSBRemote",
    ],
    "login": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.ejb3.TradeSLSBBean",
        "com.ibm.websphere.samples.daytrader.ejb3.TradeSLSBLocal",
        "com.ibm.websphere.samples.daytrader.ejb3.TradeSLSBRemote",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.JSFLoginFilter",
        "com.ibm.websphere.samples.daytrader.web.jsf.LoginValidator",
    ],
    "logout": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
    ],
    "portfolio": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.PortfolioJSF",
        "com.ibm.websphere.samples.daytrader.web.jsf.HoldingData",
        "com.ibm.websphere.samples.daytrader.web.jsf.OrderData",
    ],
    "quotes": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],
    "buystock": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.OrderDataJSF",
    ],
    "sellstock": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.OrderDataJSF",
    ],
    "account": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.AccountDataJSF",
    ],
    "userinfo": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
    ],
    "order": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.web.OrdersAlertFilter",
    ],
    "marketsummary": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.TradeAction",
        "com.ibm.websphere.samples.daytrader.TradeServices",
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],

    # Scenario servlet for registration flows
    "register": [
        "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
    ],
    "registeruser": [
        "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
    ],
}


# Conservative hint scan backstop (lowercase substring -> classes)
DAYTRADER_HINT_TO_CLASSES: Dict[str, list[str]] = {
    "action=register": ["com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet"],
    "actionregister": ["com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet"],
    "action=login": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "actionlogin": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "action=marketsummary": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],
    "actionmarketsummary": [
        "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],
}


# PlantsByWebSphere (PBW) minimal path prefix -> class expansion for trace extraction.
# FQCNs must match data/processed/fusion/plants_class_order.json.
PLANTS_PATH_TO_CLASSES: Dict[str, list[str]] = {
    "/plantsbywebsphere/admin": [
        "com.ibm.websphere.samples.pbw.war.AdminServlet",
        "com.ibm.websphere.samples.pbw.bean.PopulateDBBean",
    ],
    "/plantsbywebsphere/login": [
        "com.ibm.websphere.samples.pbw.war.LoginInfo",
        "com.ibm.websphere.samples.pbw.war.AccountServlet",
        "com.ibm.websphere.samples.pbw.war.AccountBean",
        "com.ibm.websphere.samples.pbw.bean.CustomerMgr",
        "com.ibm.websphere.samples.pbw.jpa.Customer",
    ],
    "/plantsbywebsphere/shopping": [
        "com.ibm.websphere.samples.pbw.war.ShoppingBean",
        "com.ibm.websphere.samples.pbw.war.ShoppingItem",
        "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
        "com.ibm.websphere.samples.pbw.jpa.Inventory",
        "com.ibm.websphere.samples.pbw.war.ImageServlet",
    ],
    "/plantsbywebsphere/product": [
        "com.ibm.websphere.samples.pbw.war.ProductBean",
        "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
        "com.ibm.websphere.samples.pbw.jpa.Inventory",
        "com.ibm.websphere.samples.pbw.war.ImageServlet",
    ],
    "/plantsbywebsphere/cart": [
        "com.ibm.websphere.samples.pbw.bean.ShoppingCartBean",
        "com.ibm.websphere.samples.pbw.bean.ShoppingCartContent",
        "com.ibm.websphere.samples.pbw.war.ShoppingBean",
        "com.ibm.websphere.samples.pbw.war.ShoppingItem",
    ],
    "/plantsbywebsphere/orderinfo": [
        "com.ibm.websphere.samples.pbw.war.OrderInfo",
        "com.ibm.websphere.samples.pbw.jpa.Order",
        "com.ibm.websphere.samples.pbw.jpa.OrderItem",
        "com.ibm.websphere.samples.pbw.jpa.OrderKey",
    ],
    "/plantsbywebsphere/promo": [
        "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
        "com.ibm.websphere.samples.pbw.jpa.Inventory",
    ],
}


# JPetStore: query-key/JSP-name anchors -> controllers
# Notes:
# - Observed OTEL spans have url.path like '/jpetstore/actions/Catalog.action'
#   and url.query like 'search=' or 'viewCategory=&categoryId=FISH'.
# - We treat the *first query key* as the action signal.
JPETSTORE_ACTION_TO_CONTROLLER: Dict[str, list[str]] = {
    # Catalog
    "search": ["org.springframework.samples.jpetstore.web.spring.SearchProductsController"],
    "viewcategory": ["org.springframework.samples.jpetstore.web.spring.ViewCategoryController"],
    "viewproduct": ["org.springframework.samples.jpetstore.web.spring.ViewProductController"],
    "viewitem": ["org.springframework.samples.jpetstore.web.spring.ViewItemController"],
    # Cart
    "viewcart": ["org.springframework.samples.jpetstore.web.spring.ViewCartController"],
    "additemtocart": ["org.springframework.samples.jpetstore.web.spring.AddItemToCartController"],
    "removeitemfromcart": ["org.springframework.samples.jpetstore.web.spring.RemoveItemFromCartController"],
    "updatecartquantities": ["org.springframework.samples.jpetstore.web.spring.UpdateCartQuantitiesController"],
    # Account / auth
    "signon": ["org.springframework.samples.jpetstore.web.spring.SignonController"],
    "signoff": ["org.springframework.samples.jpetstore.web.spring.SignoffController"],
    "newaccount": ["org.springframework.samples.jpetstore.web.spring.AccountFormController"],
    "editaccount": ["org.springframework.samples.jpetstore.web.spring.AccountFormController"],
    # Orders
    "listorders": ["org.springframework.samples.jpetstore.web.spring.ListOrdersController"],
    "neworder": ["org.springframework.samples.jpetstore.web.spring.OrderFormController"],
    "vieworder": ["org.springframework.samples.jpetstore.web.spring.ViewOrderController"],
}

# JPetStore: JSP compilation spans (scope io.opentelemetry.jsp-2.3)
# Example span.name: 'Compile /WEB-INF/jsp/catalog/Category.jsp'
JPETSTORE_JSP_TO_CONTROLLER: Dict[str, list[str]] = {
    "main.jsp": ["org.springframework.samples.jpetstore.web.spring.SearchProductsController"],
    "category.jsp": ["org.springframework.samples.jpetstore.web.spring.ViewCategoryController"],
    "product.jsp": ["org.springframework.samples.jpetstore.web.spring.ViewProductController"],
    "item.jsp": ["org.springframework.samples.jpetstore.web.spring.ViewItemController"],
    "cart.jsp": ["org.springframework.samples.jpetstore.web.spring.ViewCartController"],
}


__all__ = [
    "extract_classes_from_traces",
    "extract_classes_from_traces_hybrid",
    "extract_classes_from_logs",
    "extract_trace_start_times",
    "calculate_s_temp",
]
