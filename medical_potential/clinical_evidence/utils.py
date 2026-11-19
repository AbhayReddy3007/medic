"""Shared helpers for the clinical evidence dimension: buffered logging, HTTP and text utilities."""

import contextvars
import sys
import time
from statistics import mode as _stat_mode

import requests

_log_buffer_var = contextvars.ContextVar("log_buffer", default=None)


def submit_with_ctx(executor, fn, *args, **kwargs):
    """
    Submit fn to executor, explicitly copying the current contextvars
    context (including _log_buffer_var) into the worker thread.

    Plain ThreadPoolExecutor.submit() does NOT propagate contextvars to
    the new OS thread -- each new thread starts with a fresh/default
    context. Without this helper, nested thread pools (e.g. the trial
    source fetches or RWE source fetches spawned inside run_molecule)
    would lose track of which molecule's log buffer they belong to and
    fall back to printing directly, reintroducing interleaved output.
    """
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


class Log:
    @staticmethod
    def _emit(line):
        buf = _log_buffer_var.get()
        if buf is not None:
            buf.write(line + "\n")
        else:
            print(line)

    @staticmethod
    def header(msg):  Log._emit(f"\n{'='*72}\n  {msg}\n{'='*72}")
    @staticmethod
    def step(n, msg): Log._emit(f"\n{'-'*72}\n  STEP {n}: {msg}\n{'-'*72}")
    @staticmethod
    def info(msg):    Log._emit(f"    {msg}")
    @staticmethod
    def ok(msg):       Log._emit(f"    [OK]   {msg}")
    @staticmethod
    def warn(msg):     Log._emit(f"    [WARN] {msg}")
    @staticmethod
    def err(msg):
        buf = _log_buffer_var.get()
        line = f"    [ERR]  {msg}"
        if buf is not None:
            buf.write(line + "\n")
        else:
            print(line, file=sys.stderr)
    @staticmethod
    def sub(msg):      Log._emit(f"\n  --- {msg} ---")
    @staticmethod
    def raw(text):
        """Emit a large pre-formatted block (e.g. the final report) as-is."""
        buf = _log_buffer_var.get()
        if buf is not None:
            buf.write(text + "\n")
        else:
            print(text)

log = Log()


def http_get(url, params=None, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1)); continue
            log.warn(f"HTTP {r.status_code} from {url}")
            return r
        except requests.RequestException as e:
            log.warn(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def safe_str(val):
    """Convert any value to an ASCII-safe string for report/output serialization."""
    if val is None:
        return ""
    s = str(val)
    # Replace non-ASCII with closest ASCII or strip
    return s.encode("ascii", errors="replace").decode("ascii")


def _wrap_text(text, width=88, indent="    "):
    """Wrap long text to specified width with indent prefix."""
    import textwrap
    lines = textwrap.wrap(text, width=width - len(indent))
    return "\n".join(f"{indent}{line}" for line in lines) if lines else f"{indent}{text}"


def get_innovator(trials, molecule=None):
    """
    Determine the innovator (originating company) for a molecule.

    Uses the company_name field recorded against each individual trial
    (the most direct, per-trial source of truth) and takes the mode
    across all trials, rather than relying on a single upstream
    sponsor lookup done once in Step 0. Falls back to the molecule's
    'sponsor' field (from FDA/ClinicalTrials.gov normalization) only
    if no trial-level company_name data is available.
    """
    names = [t.get("company_name", "") for t in (trials or [])]
    names = [n for n in names if n and n != "Unknown"]
    if names:
        try:
            return _stat_mode(names)
        except Exception:
            from collections import Counter
            return Counter(names).most_common(1)[0][0]
    if molecule:
        return molecule.get("sponsor", "Unknown")
    return "Unknown"
