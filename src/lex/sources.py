"""
Source adapters — where does a directive come from?

## The Protocol pattern

A `Source` is anything with an `async fetch(celex_id, language)` that
returns raw XML bytes. By defining this as a `typing.Protocol`, any
object matching that shape works — no inheritance required. We get two
concrete implementations here:

- `CellarRest`: hits EUR-Lex's CELLAR API. Production path.
- `LocalFile`:  reads a pre-downloaded file off disk. Used for offline
                development, tests, and reproducible demos.

Swapping sources is a one-line change in the ingestion handler, which
means you can debug the parser without needing network — just point at
a saved XML file and iterate.

## CELLAR, for the uninitiated

CELLAR is the Publications Office's public repository of EU legal
content. Every act has a persistent URI of the form:

    http://publications.europa.eu/resource/celex/{CELEX_ID}

where CELEX is a code like `32018L1972` (3 = Directive Category,
2018 = year, L = directive, 1972 = serial). A GET against that URI
with the right `Accept` header returns the document in whatever format
you want — PDF, HTML, Formex XML, metadata notice, ...

For Formex specifically, the cleanest pattern we've found is:

    1. GET {base}/{CELEX}.{LANG3}.fmx4
       → returns an RDF manifest listing every file in the manifestation.

    2. Parse the RDF, pick the "enacting act" DOC_N entry.

    3. GET that DOC URL.
       → returns the Formex4 XML we actually want to parse.

Why two hops? Because long directives (like the EECC, ~180 pages) are
split into N separate Formex files: one for the act proper, one for
each annex. The `.fmx4` URL doesn't return a concatenated whole; it
returns an RDF index of the parts. We could fetch and concatenate them
all, but v1 keeps things simple and grabs just the enacting act. Annex
ingestion is a future extension.

## Language codes

The rest of LEX uses 2-letter ISO 639-1 codes (`en`, `fr`, `lv`).
EUR-Lex URLs use 3-letter ISO 639-2/B codes (`ENG`, `FRA`, `LAV`).
We keep 2-letter everywhere internally and translate at the edge —
right here — so the rest of the system doesn't know the difference.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import httpx
import structlog
from lxml import etree

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

class Source(Protocol):
    """Anything that can produce Formex XML bytes for a given CELEX ID."""

    async def fetch(self, celex_id: str, language: str) -> bytes: ...


# ---------------------------------------------------------------------------
# RDF manifest parsing
#
# CELLAR returns RDF/XML describing the manifestation's items. We only
# need a tiny slice of that: the URL of each item and its MIME type.
# This helper walks the RDF and picks the right DOC to fetch.
# ---------------------------------------------------------------------------

# RDF/XML namespaces used by CELLAR.
_RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_CMR_NS = "{http://publications.europa.eu/ontology/cdm/cmr#}"

# MIME type that identifies Formex4 items in the manifest.
_FMX4_MIME = "application/xml;type=fmx4"


def _pick_enacting_doc_url(rdf_bytes: bytes) -> str | None:
    """
    From a CELLAR RDF manifest, pick the URL of the main enacting act XML.

    The RDF describes every manifestation "item" — typically named
    `.../DOC_1`, `.../DOC_2`, ..., `.../DOC_N`. Each entry declares its
    MIME type. We filter for Formex4 items, then apply a heuristic to
    pick the one containing the enacting act (as opposed to an annex).

    Heuristic: sort items by their numeric DOC_N suffix. DOC_1 is
    usually an OJ cover / metadata page; the enacting act is typically
    the next one (DOC_2 for the EECC). If only one Formex item exists,
    return it. If none exist, return None.

    This is a pragmatic v1 approach. A more robust implementation would
    HEAD-request each candidate and pick by content-length (the main
    act is always the largest file), or use the RDF's `stream_label`
    field if present. We'll upgrade if this heuristic ever breaks on
    a real directive.
    """
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    root = etree.fromstring(rdf_bytes, parser=parser)

    candidates: list[str] = []
    for desc in root.iterfind(f"{_RDF_NS}Description"):
        mime_el = desc.find(f"{_CMR_NS}manifestationMimeType")
        if mime_el is None or (mime_el.text or "").strip() != _FMX4_MIME:
            continue
        about = desc.get(f"{_RDF_NS}about")
        if about:
            candidates.append(about)

    if not candidates:
        return None

    def _doc_index(url: str) -> int:
        # Extract the integer N from ".../DOC_N". Items without that
        # suffix sort to the end.
        m = re.search(r"/DOC_(\d+)$", url)
        return int(m.group(1)) if m else 10_000

    candidates.sort(key=_doc_index)

    # Skip DOC_1 (cover page) if there's at least one other candidate.
    if len(candidates) > 1 and _doc_index(candidates[0]) == 1:
        return candidates[1]
    return candidates[0]


# ---------------------------------------------------------------------------
# CELLAR — EUR-Lex's REST endpoint
# ---------------------------------------------------------------------------

class CellarRest:
    """
    Fetches Formex XML from CELLAR.

    The flow is two HTTP calls:

        GET .../celex/{CELEX}.{LANG3}.fmx4   → RDF manifest
        GET {url-picked-from-manifest}       → Formex4 XML

    See the module docstring for why this two-step approach.
    """

    BASE_URL = "http://publications.europa.eu/resource/celex"

    # ISO 639-1 → ISO 639-2/B (bibliographic variant).
    # EUR-Lex uses the "B" variant for German and a few others — "DEU"
    # not "GER", "FRA" not "FRE". This dict is the single place we
    # hardcode that mapping.
    _LANG_MAP = {
        "bg": "BUL", "cs": "CES", "da": "DAN", "de": "DEU", "el": "ELL",
        "en": "ENG", "es": "SPA", "et": "EST", "fi": "FIN", "fr": "FRA",
        "ga": "GLE", "hr": "HRV", "hu": "HUN", "it": "ITA", "lt": "LIT",
        "lv": "LAV", "mt": "MLT", "nl": "NLD", "pl": "POL", "pt": "POR",
        "ro": "RON", "sk": "SLK", "sl": "SLV", "sv": "SWE",
    }

    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    async def fetch(self, celex_id: str, language: str) -> bytes:
        lang3 = self._LANG_MAP.get(language.lower())
        if lang3 is None:
            raise SourceUnavailable(
                f"No EUR-Lex language mapping for '{language}'. "
                f"Known: {sorted(self._LANG_MAP)}"
            )

        # Step 1: fetch the RDF manifest.
        rdf_url = f"{self.BASE_URL}/{celex_id}.{lang3}.fmx4"
        log.info("cellar.rdf.fetch", celex_id=celex_id, language=language, url=rdf_url)

        async with httpx.AsyncClient(timeout=self._timeout_s, follow_redirects=True) as client:
            rdf_resp = await client.get(rdf_url)
            if rdf_resp.status_code == 404:
                # 404 at the manifest URL means no Formex4 exists in
                # this language. Typically pre-2004 docs, or certain
                # consolidated versions.
                raise SourceUnavailable(
                    f"No Formex4 manifestation for {celex_id} in {language} "
                    f"(404 at {rdf_url}). Pre-Formex documents and some "
                    f"consolidated versions don't have .fmx4 variants."
                )
            rdf_resp.raise_for_status()

            # Step 2: parse the RDF, pick the enacting act item URL.
            doc_url = _pick_enacting_doc_url(rdf_resp.content)
            if doc_url is None:
                raise SourceUnavailable(
                    f"RDF manifest for {celex_id}/{language} lists no "
                    f"{_FMX4_MIME} items. Nothing to fetch."
                )

            # Step 3: fetch the actual Formex XML.
            log.info("cellar.doc.fetch", url=doc_url)
            doc_resp = await client.get(doc_url)
            doc_resp.raise_for_status()

        log.info(
            "cellar.fetch.ok",
            celex_id=celex_id,
            language=language,
            bytes=len(doc_resp.content),
        )
        return doc_resp.content


# ---------------------------------------------------------------------------
# Local — a file on disk
# ---------------------------------------------------------------------------

class LocalFile:
    """
    Reads Formex XML from a directory on disk.

    Expects files named `{celex_id}.{language}.xml`, e.g.
    `32018L1972.en.xml`. This is how we keep ingestion tests offline
    and deterministic — download once, commit to `tests/fixtures/`,
    iterate on parser logic without touching the network.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    async def fetch(self, celex_id: str, language: str) -> bytes:
        path = self._dir / f"{celex_id}.{language}.xml"
        if not path.exists():
            raise SourceUnavailable(f"No local file at {path}")
        return path.read_bytes()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SourceUnavailable(RuntimeError):
    """Raised when a source can't produce the requested document."""
    pass
