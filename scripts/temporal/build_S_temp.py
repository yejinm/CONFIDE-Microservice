from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import argparse
import json
import numpy as np

"""Build temporal modality matrix S_temp.

This script intentionally stays *minimal* and evidence-driven.

Temporal evidence sources (hybrid, always-on):
- JTL sessionization: build sessions from JMeter JTL, map labels->classes using
  STRICT (deterministic) rules, and compute class-class similarity.
- Trace co-occurrence: extract class co-occurrences from OTel trace exports.

Key design constraints (for maintainability + paper-defensibility):
- Output matrix is always NxN aligned with `data/processed/fusion/{system}_class_order.json`.
- The default and recommended behavior is JTL+TRACE hybrid with strict mapping.

NOTE:
- Older versions of this script supported `--mode` and `--strict`. Those options
  are now accepted for backward compatibility but ignored (hybrid+strict is always used).
"""

# NOTE: runnable as:
#   python scripts/temporal/build_S_temp.py ...
# or:
#   python -m temporal.build_S_temp ...  (with scripts/ on PYTHONPATH)
try:
    from .temporal_core import (
        extract_classes_from_traces,
        calculate_s_temp,
    )
except ImportError:  # pragma: no cover
    from temporal_core import (
        extract_classes_from_traces,
        calculate_s_temp,
    )

# --- System naming map: CLI name -> physical file name ---
SYSTEM_NAME_MAP = {
    "daytrader7": "daytrader",
    "plantsbywebsphere": "plants",
}

ROOT = Path(__file__).resolve().parents[2]

# --- (Optional) heuristic keyword->core mapping (non-strict only) ---
ENDPOINT_MAPS: Dict[str, Dict[str, List[str]]] = {
    "acmeair": {
        "login": ["com.acmeair.web.LoginREST"],
        "flight": ["com.acmeair.web.FlightsREST"],
        "book": ["com.acmeair.web.BookingsREST"],
        "customer": ["com.acmeair.web.CustomerREST"],
        "loader": ["com.acmeair.loader.Loader"],
    },
    "jpetstore": {
        "signon": ["org.springframework.samples.jpetstore.domain.Account"],
        "catalog": [
            "org.springframework.samples.jpetstore.domain.Product",
            "org.springframework.samples.jpetstore.domain.Category",
        ],
        "cart": ["org.springframework.samples.jpetstore.domain.Cart"],
        "order": ["org.springframework.samples.jpetstore.domain.Order"],
    },
    "daytrader7": {
        "login": [
            "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
            "com.ibm.websphere.samples.daytrader.interfaces.TradeService",
        ],
        "portfolio": ["com.ibm.websphere.samples.daytrader.entities.HoldingDataBean"],
        "quotes": ["com.ibm.websphere.samples.daytrader.entities.QuoteDataBean"],
        "buy": ["com.ibm.websphere.samples.daytrader.entities.OrderDataBean"],
    },
    "plantsbywebsphere": {
        "admin": [
            "com.ibm.websphere.samples.pbw.war.AdminServlet",
            "com.ibm.websphere.samples.pbw.bean.PopulateDBBean",
        ],
        "backorder": [
            "com.ibm.websphere.samples.pbw.bean.BackOrderMgr",
            "com.ibm.websphere.samples.pbw.jpa.BackOrder",
        ],
        "supplier": [
            "com.ibm.websphere.samples.pbw.bean.SuppliersBean",
            "com.ibm.websphere.samples.pbw.jpa.Supplier",
        ],
        "catalog": [
            "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
            "com.ibm.websphere.samples.pbw.jpa.Inventory",
        ],
    },
}

# --- Deterministic (strict) label->class mapping for tx/flow labels ---
MINIMAL_ENTRYPOINTS: Dict[str, Dict[str, object]] = {
    "acmeair": {
        # NOTE: Keep keys aligned with JTL label normalization.
        # The gate script uses `restloginf2`, `openhomepage`, etc. We mirror that here
        # so build-time sessionization does not drop all JTL evidence.
        "restloginf2": [
            "com.acmeair.web.LoginREST",
            "com.acmeair.service.CustomerService",
            "com.acmeair.morphia.services.CustomerServiceImpl",
            "com.acmeair.morphia.entities.CustomerImpl",
            "com.acmeair.service.KeyGenerator",
        ],
        "restloginf3": [
            "com.acmeair.web.LoginREST",
            "com.acmeair.service.CustomerService",
            "com.acmeair.morphia.services.CustomerServiceImpl",
        ],
        "restloginf4": [
            "com.acmeair.web.LoginREST",
            "com.acmeair.service.CustomerService",
            "com.acmeair.morphia.services.CustomerServiceImpl",
        ],
        "restsearchflightsf2": [
            "com.acmeair.web.FlightsREST",
            "com.acmeair.service.FlightService",
            "com.acmeair.morphia.services.FlightServiceImpl",
            "com.acmeair.morphia.entities.FlightImpl",
            "com.acmeair.morphia.entities.AirportCodeMappingImpl",
            "com.acmeair.service.DataServiceFactory",
        ],
        "restbookflightf2": [
            "com.acmeair.web.BookingsREST",
            "com.acmeair.service.BookingService",
            "com.acmeair.morphia.services.BookingServiceImpl",
            "com.acmeair.morphia.entities.BookingImpl",
        ],
        "restgetbookingsf3": [
            "com.acmeair.web.BookingsREST",
            "com.acmeair.service.BookingService",
            "com.acmeair.morphia.services.BookingServiceImpl",
            "com.acmeair.morphia.entities.BookingImpl",
        ],
        "restgetcustomerf4": [
            "com.acmeair.web.CustomerREST",
            "com.acmeair.service.CustomerService",
            "com.acmeair.morphia.services.CustomerServiceImpl",
            "com.acmeair.morphia.entities.CustomerImpl",
        ],
        "restupdatecustomerf4": [
            "com.acmeair.web.CustomerREST",
            "com.acmeair.service.CustomerService",
            "com.acmeair.morphia.services.CustomerServiceImpl",
            "com.acmeair.morphia.entities.CustomerImpl",
        ],
        # UI steps (coarse, but stabilizes session evidence)
        "openhomepage": [
            "com.acmeair.web.LoginREST",
        ],
        "opencustomerprofilepagef2": [
            "com.acmeair.web.CustomerREST",
            "com.acmeair.service.CustomerService",
        ],
    },
    "jpetstore": {
        # NOTE: JTL for our JPetStore workload uses these stable action labels.
        # Mirror the gate script so JTL evidence is retained at build time.
        "home": [
            "org.springframework.samples.jpetstore.web.spring.CatalogController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
        ],
        "search": [
            "org.springframework.samples.jpetstore.web.spring.SearchProductsController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.dao.ProductDao",
            "org.springframework.samples.jpetstore.domain.Product",
        ],
        "viewcategory": [
            "org.springframework.samples.jpetstore.web.spring.ViewCategoryController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.dao.CategoryDao",
            "org.springframework.samples.jpetstore.domain.Category",
            "org.springframework.samples.jpetstore.domain.Product",
        ],
        "viewproduct": [
            "org.springframework.samples.jpetstore.web.spring.ViewProductController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.dao.ProductDao",
            "org.springframework.samples.jpetstore.domain.Product",
            "org.springframework.samples.jpetstore.domain.Item",
        ],
        "viewitem": [
            "org.springframework.samples.jpetstore.web.spring.ViewItemController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.dao.ItemDao",
            "org.springframework.samples.jpetstore.domain.Item",
        ],
        "additemtocart": [
            "org.springframework.samples.jpetstore.web.spring.AddItemToCartController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Cart",
            "org.springframework.samples.jpetstore.dao.ItemDao",
        ],
        "viewcart": [
            "org.springframework.samples.jpetstore.web.spring.ViewCartController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Cart",
        ],
        "checkout": [
            "org.springframework.samples.jpetstore.web.spring.OrderFormController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Order",
            "org.springframework.samples.jpetstore.dao.OrderDao",
        ],
        "neworder": [
            "org.springframework.samples.jpetstore.web.spring.OrderFormController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Order",
            "org.springframework.samples.jpetstore.dao.OrderDao",
        ],
        "neworderform": [
            "org.springframework.samples.jpetstore.web.spring.OrderFormController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Order",
            "org.springframework.samples.jpetstore.dao.OrderDao",
        ],
        "signon": [
            "org.springframework.samples.jpetstore.web.spring.SignonController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Account",
            "org.springframework.samples.jpetstore.dao.AccountDao",
        ],
        # JTL flow label observed in the dataset
        "flowa_catalog": [
            "org.springframework.samples.jpetstore.web.spring.CatalogController",
            "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
            "org.springframework.samples.jpetstore.domain.Category",
            "org.springframework.samples.jpetstore.domain.Product",
            "org.springframework.samples.jpetstore.domain.Item",
        ],
    },
    "daytrader7": {
        "portfolio": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "quotes": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "tradestock": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "buystock": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "sellstock": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "login": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "userlogin": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "account": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "userinfo": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "register": "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
        "registeruser": "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
        "home": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "buy": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "sell": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "order": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "logout": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",

        # Also allow TransactionController parent labels
        "t_home": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "t_login": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "t_register": "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
        "t_portfolio": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "t_quotes": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "t_buy": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "t_sell": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
        "t_accountlogout": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",

        # Strict DayTrader: bounded in-window sub-label coverage
        "market summary": [
            "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
            "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
            "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
        ],
        "quotes": [
            "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
            "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
            "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
        ],
        "portfolio": [
            "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
            "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
            "com.ibm.websphere.samples.daytrader.web.jsf.PortfolioJSF",
            "com.ibm.websphere.samples.daytrader.web.jsf.HoldingData",
            "com.ibm.websphere.samples.daytrader.web.jsf.OrderData",
        ],
        "account info": [
            "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
            "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
            "com.ibm.websphere.samples.daytrader.web.jsf.AccountDataJSF",
        ],
        "user login": [
            "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
            "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
            "com.ibm.websphere.samples.daytrader.web.jsf.LoginValidator",
            "com.ibm.websphere.samples.daytrader.web.jsf.JSFLoginFilter",
        ],
        "user register": [
            "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
            "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
            "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
        ],
        "buy stock": [
            "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
            "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
        ],
        "sell stock (by holdingid)": [
            "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
            "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
        ],
    },
    # -------------------------------------------------------------------------
    # Paper Version 1.0 (Stabilized)
    # PlantsByWebSphere strict label->class mapping (JTL sessionization)
    # - ProductBean removed (high-frequency glue) based on session coverage audit.
    # - Mapping kept business-anchored and minimal for paper-defensible evidence.
    # -------------------------------------------------------------------------
    "plantsbywebsphere": {
        # --- Admin / backorder / supplier (exclusive) ---
        "adminhomepage": [
            "com.ibm.websphere.samples.pbw.war.AdminServlet",
            "com.ibm.websphere.samples.pbw.bean.PopulateDBBean",
            "com.ibm.websphere.samples.pbw.bean.ResetDBBean",
        ],
        "adminactionspage": [
            "com.ibm.websphere.samples.pbw.war.AdminServlet",
            "com.ibm.websphere.samples.pbw.bean.PopulateDBBean",
            "com.ibm.websphere.samples.pbw.bean.ResetDBBean",
        ],
        "adminbannerpage": [
            "com.ibm.websphere.samples.pbw.war.AdminServlet",
            "com.ibm.websphere.samples.pbw.bean.PopulateDBBean",
        ],
        "backorderadminpage": [
            "com.ibm.websphere.samples.pbw.war.BackOrderItem",
            "com.ibm.websphere.samples.pbw.bean.BackOrderMgr",
            "com.ibm.websphere.samples.pbw.jpa.BackOrder",
        ],
        "processbackorderaction": [
            "com.ibm.websphere.samples.pbw.war.BackOrderItem",
            "com.ibm.websphere.samples.pbw.bean.BackOrderMgr",
            "com.ibm.websphere.samples.pbw.jpa.BackOrder",
        ],
        "supplierconfigpage": [
            "com.ibm.websphere.samples.pbw.bean.SuppliersBean",
            "com.ibm.websphere.samples.pbw.jpa.Supplier",
        ],

        # --- Browse / catalog (exclusive-ish; keep Inventory here, NOT in order) ---
        "homepromopage": [
            "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
            "com.ibm.websphere.samples.pbw.jpa.Inventory",
            "com.ibm.websphere.samples.pbw.war.ImageServlet",
            "com.ibm.websphere.samples.pbw.war.ProductBean",
        ],
        "shoppingpage": [
            "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
            "com.ibm.websphere.samples.pbw.jpa.Inventory",
            "com.ibm.websphere.samples.pbw.war.ShoppingBean",
            "com.ibm.websphere.samples.pbw.war.ShoppingItem",
            "com.ibm.websphere.samples.pbw.war.ProductBean",
        ],
        "productdetailspage": [
            "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
            "com.ibm.websphere.samples.pbw.jpa.Inventory",
            "com.ibm.websphere.samples.pbw.war.ProductBean",
            "com.ibm.websphere.samples.pbw.war.ImageServlet",
        ],
        "j1userbrowse": [
            "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
            "com.ibm.websphere.samples.pbw.jpa.Inventory",
            "com.ibm.websphere.samples.pbw.war.ShoppingBean",
            "com.ibm.websphere.samples.pbw.war.ShoppingItem",
        ],

        # --- Cart (exclusive) ---
        "addtocartaction": [
            "com.ibm.websphere.samples.pbw.bean.ShoppingCartBean",
            "com.ibm.websphere.samples.pbw.bean.ShoppingCartContent",
            "com.ibm.websphere.samples.pbw.war.ShoppingBean",
            "com.ibm.websphere.samples.pbw.war.ShoppingItem",
        ],
        "cartpage": [
            "com.ibm.websphere.samples.pbw.bean.ShoppingCartBean",
            "com.ibm.websphere.samples.pbw.bean.ShoppingCartContent",
            "com.ibm.websphere.samples.pbw.war.ShoppingBean",
            "com.ibm.websphere.samples.pbw.war.ShoppingItem",
        ],

        # --- Auth (keep separate; don't anchor everything) ---
        "openloginpage": [
            "com.ibm.websphere.samples.pbw.war.LoginInfo",
            "com.ibm.websphere.samples.pbw.war.AccountServlet",
            "com.ibm.websphere.samples.pbw.war.AccountBean",
        ],
        "userlogin": [
            "com.ibm.websphere.samples.pbw.war.LoginInfo",
            "com.ibm.websphere.samples.pbw.war.AccountServlet",
            "com.ibm.websphere.samples.pbw.war.AccountBean",
        ],

        # --- Order / purchase (exclusive; DO NOT include Inventory here) ---
        "orderinfo": [
            "com.ibm.websphere.samples.pbw.jpa.Order",
            "com.ibm.websphere.samples.pbw.jpa.OrderItem",
            "com.ibm.websphere.samples.pbw.jpa.OrderKey",
            "com.ibm.websphere.samples.pbw.war.OrderInfo",
            "com.ibm.websphere.samples.pbw.war.OrderInfo",
        ],
        "submitorder": [
            "com.ibm.websphere.samples.pbw.jpa.Order",
            "com.ibm.websphere.samples.pbw.jpa.OrderItem",
            "com.ibm.websphere.samples.pbw.jpa.OrderKey",
            "com.ibm.websphere.samples.pbw.jpa.OrderItem.PK",
            "com.ibm.websphere.samples.pbw.war.OrderInfo",
        ],
        "j2userpurchase": [
            "com.ibm.websphere.samples.pbw.jpa.Order",
            "com.ibm.websphere.samples.pbw.jpa.OrderItem",
            "com.ibm.websphere.samples.pbw.jpa.OrderKey",
            "com.ibm.websphere.samples.pbw.war.OrderInfo",
            "com.ibm.websphere.samples.pbw.bean.CustomerMgr",
            "com.ibm.websphere.samples.pbw.jpa.Customer",
        ],
    },

    # NOTE: The remainder of this file (sessionization + fusion logic) was truncated.
    # Re-add the missing implementation to keep the pipeline functional.
}

def _normalize_label(label: str) -> str:
    # Normalize JMeter labels into stable keys for strict mapping.
    # NOTE: Some datasets include punctuation (e.g., 'Home / Promo Page', 'J1_User_Browse').
    # We strip common separators so strict keys remain deterministic.
    s = (label or "").strip().lower()
    for ch in [" ", "_", "-", "/", "\\", ":", ".", ","]:
        s = s.replace(ch, "")
    return s


def _load_class_order(system: str) -> List[str]:
    physical = SYSTEM_NAME_MAP.get(system, system)
    p = ROOT / "data" / "processed" / "fusion" / f"{physical}_class_order.json"
    if not p.is_file():
        alt = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"class_order not found for {system}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_jtl_rows(jtl_path: Path, *, max_rows: int) -> List[Dict[str, str]]:
    import csv

    rows: List[Dict[str, str]] = []
    if not jtl_path.is_file():
        return rows

    # Be robust to headerless JTL exports.
    # If the first row doesn't contain a header (common in some JMeter setups),
    # DictReader will treat the first data row as fieldnames and everything breaks.
    headerless_fieldnames = [
        "timeStamp",
        "elapsed",
        "label",
        "responseCode",
        "responseMessage",
        "threadName",
        "dataType",
        "success",
        "failureMessage",
        "bytes",
        "sentBytes",
        "grpThreads",
        "allThreads",
        "URL",
        "Latency",
        "IdleTime",
        "Connect",
        "threadIteration",
    ]

    def _looks_like_header(sample: str) -> bool:
        s = (sample or "").strip().lower()
        return ("label" in s) and ("timestamp" in s or "timeStamp" in s)

    with jtl_path.open("r", encoding="utf-8", newline="") as f:
        # Peek first line
        first_line = f.readline()
        f.seek(0)

        if first_line and not _looks_like_header(first_line):
            reader = csv.DictReader(f, fieldnames=headerless_fieldnames)
        else:
            reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            if row:
                rows.append(row)

    return rows


def _extract_thread_iteration(row: Dict[str, str]) -> Tuple[str, int]:
    thread_name = (row.get("threadName") or row.get("thread") or "").strip()
    it_raw = row.get("threadIteration") or row.get("iteration") or ""
    try:
        it = int(float(it_raw))
    except Exception:
        it = -1
    return thread_name, it


def _map_label_to_indices_strict(system: str, label: str, class_to_idx: Dict[str, int]) -> List[int]:
    sys_key = system
    if sys_key not in MINIMAL_ENTRYPOINTS:
        return []

    key = _normalize_label(label)
    # Also drop parentheses/brackets characters (common in AcmeAir labels: '(F2)')
    key = (
        key.replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
    )

    mapped = MINIMAL_ENTRYPOINTS[sys_key].get(key)
    if not mapped:
        return []

    mapped_list = [mapped] if isinstance(mapped, str) else list(mapped)
    out: List[int] = []
    for fqcn in mapped_list:
        idx = class_to_idx.get(str(fqcn))
        if idx is not None:
            out.append(int(idx))
    return out


def _build_sessions_from_jtl(
    system: str,
    *,
    strict: bool,
    group_by: str,
    window_size: int = 12,
    stride: int = 6,
    max_events: int = 80,
    min_events_per_session: int = 2,
    max_session_seconds: float = 5.0,
    debug: bool = False,
    debug_sessions: int = 5,
) -> List[List[int]]:
    """Rebuild sessions from JMeter JTL using strict mapping.

    Supported grouping:
      - thread
      - thread_iteration (recommended for paper baseline)
      - sliding_window (uses window_size/stride over the *event sequence*)

    DayTrader add-on:
      - max_session_seconds: caps unusually long thread sessions by time gaps.
        When the elapsed time from the first event in the current session exceeds
        this cap, a new session is started for the same group key.
    """

    physical = SYSTEM_NAME_MAP.get(system, system)
    jtl = ROOT / "results" / "jmeter" / f"{physical}_results.jtl"

    order = _load_class_order(system)
    class_to_idx = {c: i for i, c in enumerate(order)}

    # Use the mapping namespace key expected by MINIMAL_ENTRYPOINTS
    map_key = "daytrader7" if system in {"daytrader", "daytrader7"} else system
    if system in {"plants", "plantsbywebsphere"}:
        map_key = "plantsbywebsphere"

    rows = _load_jtl_rows(jtl, max_rows=max_events if max_events > 0 else 0)

    # Build raw mapped event stream per group
    grouped_events: Dict[str, List[Tuple[int, int]]] = {}  # key -> [(t_ms, class_idx)]
    seen_labels = 0
    mapped_events = 0
    dropped_no_label = 0
    dropped_unmapped = 0

    for row in rows:
        # Robust label field extraction (headerless and headered JTL variants)
        label = (
            (row.get("label") or "")
            or (row.get("Label") or "")
            or (row.get("SamplerLabel") or "")
            or (row.get("sampler_label") or "")
        ).strip()
        if not label:
            dropped_no_label += 1
            continue
        seen_labels += 1

        idxs = _map_label_to_indices_strict(map_key, label, class_to_idx) if strict else []
        if not idxs:
            dropped_unmapped += 1
            continue

        # timeStamp is standard JTL column
        t_raw = row.get("timeStamp") or row.get("timestamp") or ""
        try:
            t_ms = int(float(t_raw))
        except Exception:
            t_ms = -1

        if group_by == "thread":
            gk = (row.get("threadName") or row.get("thread") or "").strip()
        elif group_by == "thread_iteration":
            tname, it = _extract_thread_iteration(row)
            # If iteration is missing (common in some JTL exports), do not force-split.
            # Falling back to threadName prevents creating many degenerate groups where
            # each group contains only repeated occurrences of a single label.
            if it < 0:
                gk = (tname or "").strip()
            else:
                gk = f"{tname}::it={it}"
        elif group_by == "sliding_window":
            # put everything in one stream; windowing handled later
            gk = "__all__"
        else:
            raise ValueError(f"Unsupported group_by: {group_by}")

        if not gk:
            continue

        for idx in idxs:
            grouped_events.setdefault(gk, []).append((t_ms, int(idx)))
            mapped_events += 1

    sessions: List[List[int]] = []

    def _split_by_timecap(evts: List[Tuple[int, int]]) -> List[List[int]]:
        if not evts:
            return []
        # keep insertion order; if timestamps are missing (-1), skip time split
        if all(t < 0 for (t, _) in evts) or max_session_seconds <= 0:
            return [[idx for (_, idx) in evts]]

        out: List[List[int]] = []
        cur: List[int] = []
        start_t: Optional[int] = None
        for (t, idx) in evts:
            if start_t is None and t >= 0:
                start_t = t
            if start_t is not None and t >= 0:
                if (t - start_t) / 1000.0 > float(max_session_seconds):
                    if cur:
                        out.append(cur)
                    cur = []
                    start_t = t
            cur.append(idx)
        if cur:
            out.append(cur)
        return out

    if group_by in {"thread", "thread_iteration"}:
        for evts in grouped_events.values():
            for sess in _split_by_timecap(evts):
                if len(sess) >= int(min_events_per_session):
                    sessions.append(sess)

    else:  # sliding_window over the single stream
        evts = grouped_events.get("__all__", [])
        seq = [idx for (_, idx) in evts]
        if window_size <= 0:
            window_size = 12
        if stride <= 0:
            stride = max(1, window_size // 2)
        for i in range(0, max(0, len(seq) - window_size + 1), stride):
            w = seq[i : i + window_size]
            if len(w) >= int(min_events_per_session):
                sessions.append(w)

    if debug:
        # Summarize mapping effectiveness and show representative sessions
        print(
            "  [debug-jtl] rows=", len(rows),
            "seen_labels=", seen_labels,
            "mapped_events=", mapped_events,
            "dropped_no_label=", dropped_no_label,
            "dropped_unmapped=", dropped_unmapped,
            "groups=", len(grouped_events),
        )

        # Print first few sessions with both raw and unique counts
        shown = 0
        for k, evts in list(grouped_events.items())[: max(1, 50)]:
            seq = [idx for (_, idx) in evts]
            uniq = sorted(set(seq))
            if not seq:
                continue
            print(f"  [debug-jtl] group={k} events={len(seq)} unique_classes={len(uniq)}")
            # Print class names (unique) for readability
            cls_names = [order[i] for i in uniq if 0 <= i < len(order)]
            print("    ", "; ".join(cls_names[:40]))
            shown += 1
            if shown >= int(debug_sessions):
                break

    return sessions


def _cooccurrence_from_sessions(sessions: List[List[int]], n: int) -> np.ndarray:
    """Dense co-occurrence counts from sessions (set-based per session)."""
    M = np.zeros((n, n), dtype=np.float32)
    for s in sessions:
        uniq = sorted({int(x) for x in s if 0 <= int(x) < n})
        for a_i in range(len(uniq)):
            i = uniq[a_i]
            for a_j in range(a_i + 1, len(uniq)):
                j = uniq[a_j]
                M[i, j] += 1.0
                M[j, i] += 1.0
    return M


def _normalize_to_similarity(M: np.ndarray) -> np.ndarray:
    """Convert co-occurrence counts to a [0,1] similarity using standard Jaccard.

    Definitions
    -----------
    Let sessions be binary sets. Then:
      - occ(i)  = #sessions containing class i
      - co(i,j) = #sessions where i and j co-occur (i!=j)

    We compute:
      S(i,j) = co(i,j) / (occ(i) + occ(j) - co(i,j))

    Notes
    -----
    - This matches `temporal_core.calculate_s_temp`'s scaling, ensuring JTL and TRACE
      components are measured on the same scale.
    - Diagonal is forced to 0.0 (self-loops are not used in intra/inter stats).
    """
    if M.size == 0:
        return M

    # NEW: always compute Jaccard from a matrix with zero diagonal (pure co-occurrence)
    C = M.copy()
    np.fill_diagonal(C, 0.0)

    # Prefer true occurrence counts encoded on diagonal; fallback to row sums
    occ = np.diag(M).copy().reshape(-1, 1).astype(np.float32)
    if np.all(occ == 0):
        occ = C.sum(axis=1, keepdims=True).astype(np.float32)

    denom = occ + occ.T - C
    denom = np.where(denom <= 0, 1.0, denom)
    S = C / denom
    np.fill_diagonal(S, 0.0)
    return S.astype(np.float32)


def _build_s_from_jtl(system: str, *, group_by: str, max_events: int, min_events: int, max_session_seconds: float, jtl_drop_rate: float = 0.0, jtl_drop_seed: int = 1337, debug: bool = False, debug_sessions: int = 5) -> Tuple[np.ndarray, Dict[str, int]]:
    order = _load_class_order(system)
    n = len(order)
    sessions = _build_sessions_from_jtl(
        system,
        strict=True,
        group_by=group_by,
        window_size=12,
        stride=6,
        max_events=max_events,
        min_events_per_session=min_events,
        max_session_seconds=max_session_seconds,
        debug=debug,
        debug_sessions=debug_sessions,
    )

    # Optional: cold-start simulation by dropping JTL sessions
    jtl_drop_meta = {"jtl_sessions_before": int(len(sessions)), "jtl_sessions_after": int(len(sessions)), "jtl_sessions_dropped": 0}
    if float(jtl_drop_rate) > 0:
        sessions, jtl_drop_meta = _apply_jtl_session_drop(sessions, float(jtl_drop_rate), int(jtl_drop_seed))

    # Co-occurrence (off-diagonal) + per-class occurrence count (occ) on diagonal
    M = _cooccurrence_from_sessions(sessions, n=n)
    occ = np.zeros(n, dtype=np.float32)
    for s in sessions:
        uniq = {int(x) for x in s if 0 <= int(x) < n}
        for i in uniq:
            occ[i] += 1.0
    np.fill_diagonal(M, occ)

    S = _normalize_to_similarity(M)

    # off-diagonal nonzeros (count each directed entry; gate uses same convention)
    off = S.copy()
    np.fill_diagonal(off, 0.0)
    off_nz = int((off > 0).sum())
    meta = {
        "sessions": len(sessions),
        "offdiag_nonzero": int(off_nz),
        **{k: int(v) for k, v in (jtl_drop_meta or {}).items()},
    }
    return S, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="acmeair|daytrader|daytrader7|jpetstore|plants|plantsbywebsphere")

    # Keep backwards compat, but prefer explicit JTL/trace weights
    ap.add_argument("--alpha", type=float, default=None, help="(legacy) JTL weight; prefer --alpha-jtl")
    ap.add_argument("--beta", type=float, default=None, help="(legacy) Trace weight; prefer --beta-trace")
    ap.add_argument("--alpha-jtl", type=float, default=0.5)
    ap.add_argument("--beta-trace", type=float, default=0.5)

    ap.add_argument("--group-by", default="thread_iteration", choices=["thread", "thread_iteration", "sliding_window"])
    ap.add_argument("--max-events", type=int, default=80)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--max-session-seconds", type=float, default=5.0)

    # Cold-start simulation (JTL): randomly drop a portion of JTL sessions
    ap.add_argument(
        "--jtl_drop_rate",
        type=float,
        default=0.0,
        help=(
            "Randomly discard this fraction of JTL sessions before building the JTL temporal evidence (0..1). "
            "Use to simulate sparse/limited load-test runtime logs. Applied before co-occurrence computation."
        ),
    )
    ap.add_argument(
        "--jtl_drop_seed",
        type=int,
        default=1337,
        help="Seed for --jtl_drop_rate to make drops deterministic and reproducible.",
    )

    # Cold-start simulation: randomly drop a portion of traces (traceId groups)
    ap.add_argument(
        "--trace_drop_rate",
        type=float,
        default=0.0,
        help="Randomly discard this fraction of traces before building S_temp (0..1). Use to simulate sparse runtime data.",
    )
    ap.add_argument(
        "--trace_drop_seed",
        type=int,
        default=1337,
        help="Seed for --trace_drop_rate to make drops deterministic and reproducible.",
    )

    # Cold-start simulation (ALT): randomly drop spans/events within traces
    ap.add_argument(
        "--span_drop_rate",
        type=float,
        default=0.0,
        help=(
            "Randomly discard this fraction of span-to-class hits inside traces (0..1). "
            "Unlike --trace_drop_rate which drops whole requests, this simulates missing/partial instrumentation "
            "and can change S_temp sparsity. Applied after trace extraction but before building S_trace."
        ),
    )
    ap.add_argument(
        "--span_drop_seed",
        type=int,
        default=1337,
        help="Seed for --span_drop_rate to make drops deterministic and reproducible.",
    )

    # Optional smoothing for nicer paper heatmaps (purely cosmetic, tiny magnitude)
    ap.add_argument(
        "--package-smoothing",
        type=float,
        default=0.0,
        help="If >0, add a tiny constant to pairs in the same leaf package (after fusion).",
    )

    # Debug / audit switches
    ap.add_argument("--debug-jtl", action="store_true", help="Print mapped JTL sessions (first few) and mapping drop stats")
    ap.add_argument("--debug-jtl-sessions", type=int, default=5, help="How many mapped sessions to print when --debug-jtl is set")

    ap.add_argument(
        "--debug-trace",
        action="store_true",
        help="Print trace extraction diagnostics (service.name TopK, match counts, span hit-rate)",
    )
    ap.add_argument("--debug-trace-topk", type=int, default=10, help="TopK service.name values to show with --debug-trace")
    ap.add_argument(
        "--debug-trace-sample-spans",
        type=int,
        default=0,
        help="If >0, print a small sample of spans (name, keys, selected http/db/rpc/code attrs) for debugging",
    )
    return ap.parse_args()


def _apply_jtl_session_drop(sessions: List[List[int]], drop_rate: float, seed: int) -> tuple[List[List[int]], Dict[str, int]]:
    """Randomly discard a fraction of JTL sessions (cold-start simulation).

    This drops whole sessions to mimic fewer observed transactions.
    Deterministic given seed.
    """
    try:
        r = float(drop_rate)
    except Exception:
        r = 0.0

    n_before = int(len(sessions or []))
    if r <= 0.0 or n_before == 0:
        return sessions, {"jtl_sessions_before": n_before, "jtl_sessions_after": n_before, "jtl_sessions_dropped": 0}

    r = max(0.0, min(1.0, r))
    k_keep = int(round((1.0 - r) * n_before))
    k_keep = max(0, min(n_before, k_keep))

    rng = np.random.RandomState(int(seed))
    if k_keep == 0:
        kept_idx = set()
    elif k_keep == n_before:
        kept_idx = set(range(n_before))
    else:
        kept_idx = set(rng.choice(np.arange(n_before), size=k_keep, replace=False).tolist())

    out = [s for i, s in enumerate(sessions) if i in kept_idx]
    meta = {
        "jtl_sessions_before": n_before,
        "jtl_sessions_after": int(len(out)),
        "jtl_sessions_dropped": int(n_before - len(out)),
    }
    return out, meta


def _apply_leaf_package_smoothing(order: List[str], S: np.ndarray, eps: float) -> np.ndarray:
    """Add a tiny background similarity to classes sharing the same *leaf* package.

    This is intended only to avoid fully-white "islands" in heatmaps when a class
    never appears in JTL nor traces. Keep eps very small (e.g., 0.005-0.02).

    Notes:
    - Applies to off-diagonal entries only.
    - Does not renormalize; keep eps small so it doesn't affect downstream metrics.
    """
    if eps <= 0 or S.size == 0:
        return S

    groups: Dict[str, List[int]] = {}
    for i, cls in enumerate(order):
        pkg = cls.rsplit(".", 1)[0] if "." in cls else ""
        leaf = pkg.rsplit(".", 1)[-1] if pkg else ""
        groups.setdefault(leaf, []).append(i)

    S2 = S.copy()
    for leaf, idxs in groups.items():
        if not leaf or len(idxs) < 2:
            continue
        for a_i in range(len(idxs)):
            i = idxs[a_i]
            for a_j in range(a_i + 1, len(idxs)):
                j = idxs[a_j]
                if i == j:
                    continue
                S2[i, j] = float(S2[i, j]) + float(eps)
                S2[j, i] = float(S2[j, i]) + float(eps)

    np.fill_diagonal(S2, 0.0)
    return S2.astype(np.float32)


def _apply_trace_drop(trace_map: Dict[str, set], drop_rate: float, seed: int) -> tuple[Dict[str, set], Dict[str, int]]:
    """Randomly discard a fraction of traceIds (cold-start simulation).

    We drop whole traceId groups (not individual spans) to mimic fewer requests being observed.
    The sampling is deterministic given the same seed.
    """
    try:
        r = float(drop_rate)
    except Exception:
        r = 0.0
    if r <= 0.0 or not trace_map:
        return trace_map, {
            "traces_before": int(len(trace_map)),
            "traces_after": int(len(trace_map)),
            "traces_dropped": 0,
        }

    r = max(0.0, min(1.0, r))
    keys = list(trace_map.keys())
    n = len(keys)
    k_keep = int(round((1.0 - r) * n))
    k_keep = max(0, min(n, k_keep))

    rng = np.random.RandomState(int(seed))
    if k_keep == 0:
        kept = set()
    elif k_keep == n:
        kept = set(keys)
    else:
        kept = set(rng.choice(keys, size=k_keep, replace=False).tolist())

    out = {tid: v for tid, v in trace_map.items() if tid in kept}
    meta = {"traces_before": int(n), "traces_after": int(len(out)), "traces_dropped": int(n - len(out))}
    return out, meta


def _apply_span_drop(trace_map: Dict[str, set], drop_rate: float, seed: int) -> tuple[Dict[str, set], Dict[str, int]]:
    """Randomly discard a fraction of span hits *within* traces.

    `extract_classes_from_traces()` returns traceId -> set[int] (class indices seen in that trace).
    We approximate span/event loss by randomly dropping entries from these per-trace sets.

    Returns the filtered trace_map and meta counts.
    """
    try:
        r = float(drop_rate)
    except Exception:
        r = 0.0

    before_items = int(sum(len(v) for v in trace_map.values())) if trace_map else 0
    if r <= 0.0 or not trace_map:
        return trace_map, {
            "span_items_before": before_items,
            "span_items_after": before_items,
            "span_items_dropped": 0,
        }

    r = max(0.0, min(1.0, r))
    rng = np.random.RandomState(int(seed))

    out: Dict[str, set] = {}
    dropped = 0
    kept = 0

    for tid, cls_set in trace_map.items():
        if not cls_set:
            out[tid] = set()
            continue

        new_set = set()
        for x in sorted(cls_set):
            if rng.rand() < r:
                dropped += 1
            else:
                new_set.add(int(x))
                kept += 1
        out[tid] = new_set

    meta = {
        "span_items_before": int(before_items),
        "span_items_after": int(kept),
        "span_items_dropped": int(dropped),
    }
    return out, meta


def main() -> None:
    args = parse_args()

    # explicit flags win; legacy flags override only if user actually provides them
    alpha_jtl = float(args.alpha_jtl)
    beta_trace = float(args.beta_trace)
    if args.alpha is not None:
        alpha_jtl = float(args.alpha)
    if args.beta is not None:
        beta_trace = float(args.beta)

    # Build JTL component using strict mapping
    S_jtl, jtl_meta = _build_s_from_jtl(
        args.system,
        group_by=args.group_by,
        max_events=int(args.max_events),
        min_events=int(args.min_events),
        max_session_seconds=float(args.max_session_seconds),
        jtl_drop_rate=float(args.jtl_drop_rate),
        jtl_drop_seed=int(args.jtl_drop_seed),
        debug=bool(args.debug_jtl),
        debug_sessions=int(args.debug_jtl_sessions),
    )

    order = _load_class_order(args.system)
    class_to_idx = {c: i for i, c in enumerate(order)}

    physical = SYSTEM_NAME_MAP.get(args.system, args.system)

    # DayTrader traces are recorded under service.name=daytrader7
    trace_stem = physical
    service_name = args.system
    if args.system in {"daytrader", "daytrader7"}:
        trace_stem = "daytrader7"
        service_name = "daytrader7"

    # Plants traces are recorded under service.name=plantsbywebsphere, but we store as plants.json
    if args.system in {"plants", "plantsbywebsphere"}:
        trace_stem = "plants"
        service_name = "plantsbywebsphere"

    trace_path = ROOT / "data" / "processed" / "traces" / f"{trace_stem}.json"
    trace_map = extract_classes_from_traces(
        trace_path,
        class_to_idx,
        service_name=service_name,
        debug=bool(args.debug_trace),
        debug_topk=int(args.debug_trace_topk),
        debug_sample_spans=int(args.debug_trace_sample_spans),
    )

    # Optional: cold-start simulation by dropping traces
    trace_drop_meta = {
        "traces_before": int(len(trace_map)),
        "traces_after": int(len(trace_map)),
        "traces_dropped": 0,
    }
    if float(args.trace_drop_rate) > 0:
        trace_map, trace_drop_meta = _apply_trace_drop(trace_map, float(args.trace_drop_rate), int(args.trace_drop_seed))

    # Optional: cold-start simulation by dropping span hits within traces
    span_drop_meta = {"span_items_before": 0, "span_items_after": 0, "span_items_dropped": 0}
    if float(args.span_drop_rate) > 0:
        trace_map, span_drop_meta = _apply_span_drop(trace_map, float(args.span_drop_rate), int(args.span_drop_seed))

    S_trace = calculate_s_temp(trace_map, num_classes=len(order))

    # Fuse
    S = (alpha_jtl * S_jtl) + (beta_trace * S_trace)

    # Optional cosmetic smoothing
    if float(args.package_smoothing) > 0:
        S = _apply_leaf_package_smoothing(order, S, float(args.package_smoothing))

    # NEW: force diagonal to 0.0 (no self-loops)
    # This avoids unstable self-sim signals (e.g., 0.5) impacting Laplacian/fusion.
    if S.size:
        np.fill_diagonal(S, 0.0)

    out_path = ROOT / "data" / "processed" / "temporal" / f"{physical}_S_temp.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), S.astype(np.float32))

    # Sidecar meta for reproducible audit (cold-start kept/dropped)
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "system": str(args.system),
        "physical": str(physical),
        "trace_path": trace_path.as_posix(),
        "service_name": str(service_name),
        "alpha_jtl": float(alpha_jtl),
        "beta_trace": float(beta_trace),
        "trace_drop_rate": float(args.trace_drop_rate),
        "trace_drop_seed": int(args.trace_drop_seed),
        **{k: int(v) for k, v in (trace_drop_meta or {}).items()},
        "span_drop_rate": float(args.span_drop_rate),
        "span_drop_seed": int(args.span_drop_seed),
        **{k: int(v) for k, v in (span_drop_meta or {}).items()},
        "jtl_drop_rate": float(args.jtl_drop_rate),
        "jtl_drop_seed": int(args.jtl_drop_seed),
        "jtl_sessions_before": int(jtl_meta.get("jtl_sessions_before", jtl_meta.get("sessions", 0))),
        "jtl_sessions_after": int(jtl_meta.get("jtl_sessions_after", jtl_meta.get("sessions", 0))),
        "jtl_sessions_dropped": int(jtl_meta.get("jtl_sessions_dropped", 0)),
        "sessions": int(jtl_meta.get("sessions", 0)),
        "jtl_offdiag_nonzero": int(jtl_meta.get("offdiag_nonzero", 0)),
    }

    # --- Debug / audit output ---------------------------------------------------
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass
    # -----------------------------------------------------------------------------

    off = S.copy()
    np.fill_diagonal(off, 0.0)
    off_nz = int((off > 0).sum())

    print(f"[build_S_temp] system={args.system} physical={physical}")
    print(
        f"  JTL: sessions={jtl_meta.get('sessions', 0)} offdiag_nonzero={jtl_meta.get('offdiag_nonzero', 0)} alpha={alpha_jtl}"
        + (
            f" jtl_drop_rate={float(args.jtl_drop_rate):.2f} seed={int(args.jtl_drop_seed)}"
            f" sessions_dropped={int(jtl_meta.get('jtl_sessions_dropped', 0))}"
            f" kept={int(jtl_meta.get('jtl_sessions_after', jtl_meta.get('sessions', 0)))}/{int(jtl_meta.get('jtl_sessions_before', jtl_meta.get('sessions', 0)))}"
            if float(args.jtl_drop_rate) > 0
            else ""
        )
    )

    print(
        f"  TRACE: traces={len(trace_map)} beta={beta_trace} path={trace_path.as_posix()} service_name={service_name}"
        + (
            f" trace_drop_rate={float(args.trace_drop_rate):.2f} seed={int(args.trace_drop_seed)}"
            f" traces_dropped={trace_drop_meta.get('traces_dropped', 0)}"
            f" traces_kept={trace_drop_meta.get('traces_after', 0)}/{trace_drop_meta.get('traces_before', 0)}"
            if float(args.trace_drop_rate) > 0
            else ""
        )
        + (
            f" span_drop_rate={float(args.span_drop_rate):.2f} seed={int(args.span_drop_seed)}"
            f" span_items_dropped={span_drop_meta.get('span_items_dropped', 0)}"
            f" span_items_kept={span_drop_meta.get('span_items_after', 0)}/{span_drop_meta.get('span_items_before', 0)}"
            if float(args.span_drop_rate) > 0
            else ""
        )
    )

    if float(args.package_smoothing) > 0:
        print(f"  SMOOTH: package_smoothing={float(args.package_smoothing)} (leaf package)")
    print(f"  FUSED: n={S.shape[0]} offdiag_nonzero={off_nz} out={out_path.as_posix()}")


if __name__ == "__main__":
    main()