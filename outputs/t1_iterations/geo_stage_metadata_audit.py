"""Build a stage-screened GEO sample manifest from metadata only.

This never fetches expression or the mixed-stage GSE76118_RAW.tar archive.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76118/miniml/GSE76118_family.xml.tgz"
HERE = Path(__file__).resolve().parent
OUT = HERE / "GSE76118_WT_STAGE_MANIFEST.json"
NS = {"g": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
ALLOWED = {"e8.5", "e9.5"}
MAX_ARCHIVE_BYTES = 1_000_000
MAX_XML_BYTES = 12_000_000


def clean(value: str | None) -> str:
    return " ".join((value or "").split())


def main() -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "virtualembryo-stage-metadata-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("Unexpectedly large GEO metadata archive")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != 1 or members[0].name != "GSE76118_family.xml":
            raise ValueError("Unexpected GEO metadata archive members")
        if members[0].size > MAX_XML_BYTES:
            raise ValueError("Unexpectedly large GEO metadata XML")
        source = bundle.extractfile(members[0])
        if source is None:
            raise ValueError("GEO metadata XML missing")
        xml = source.read(MAX_XML_BYTES + 1)
    if len(xml) > MAX_XML_BYTES:
        raise ValueError("Unexpectedly large GEO metadata XML")
    root = ET.fromstring(xml)
    allow: dict[str, list[str]] = {"E8.5": [], "E9.5": []}
    excluded = Counter()
    regions: dict[str, Counter[str]] = {"E8.5": Counter(), "E9.5": Counter()}
    backgrounds: dict[str, Counter[str]] = {"E8.5": Counter(), "E9.5": Counter()}
    all_accessions: set[str] = set()
    separate_b6_controls: list[str] = []
    for sample in root.findall("g:Sample", NS):
        accession = clean(sample.findtext("g:Accession", namespaces=NS))
        title = clean(sample.findtext("g:Title", namespaces=NS)).lower()
        channel = sample.find("g:Channel", NS)
        if not accession.startswith("GSM") or accession in all_accessions or channel is None:
            raise ValueError("Missing or duplicate GEO sample accession/channel")
        all_accessions.add(accession)
        source_name = clean(channel.findtext("g:Source", namespaces=NS)).lower()
        chars = [clean(c.text).lower() for c in channel.findall("g:Characteristics", NS)]
        stage = chars[1] if len(chars) > 1 else "unknown"
        if stage not in ALLOWED:
            excluded[stage] += 1
            continue
        if not title.startswith(stage + "_") or not source_name.startswith(stage + " "):
            raise ValueError(f"Ambiguous stage metadata for {accession}")
        if len(chars) < 3 or chars[0] not in {"cd1", "b6"}:
            raise ValueError(f"Unexpected genotype/background for {accession}")
        if any(term in title + " " + source_name for term in
               ("knockout", "knock-out", "-/-", "cre;", "nkx2.5mut", "nkx2.5 mut")):
            excluded["mutant_" + stage] += 1
            continue
        if chars[0] != "cd1":
            if chars[0] == "b6" and "nkx2.5wt" in title and "nkx2.5 wt" in source_name:
                separate_b6_controls.append(accession)
                excluded["separate_b6_wt_cohort"] += 1
                continue
            raise ValueError(f"Unresolved non-CD1 background for {accession}")
        stage_key = stage.upper()
        allow[stage_key].append(accession)
        backgrounds[stage_key][chars[0]] += 1
        regions[stage_key][chars[2]] += 1
    if len(all_accessions) != 3241 or len(allow["E8.5"]) != 143 or len(allow["E9.5"]) != 999:
        raise ValueError("GEO metadata sample counts changed; review before use")
    if excluded["e10.5"] != 1536 or excluded["embryoid body"] != 274:
        raise ValueError("Protected or non-embryo exclusion counts changed")
    if excluded["mutant_e9.5"] != 127 or len(separate_b6_controls) != 162:
        raise ValueError("Mutant exclusion counts changed")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Metadata-only sample accession allowlist. No expression files or mixed-stage RAW.tar acquired.",
        "accession": "GSE76118",
        "metadata_url": URL,
        "metadata_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "metadata_xml_sha256": hashlib.sha256(xml).hexdigest(),
        "total_samples": len(all_accessions),
        "excluded_counts": dict(sorted(excluded.items())),
        "allowed_sample_ids": {stage: sorted(ids) for stage, ids in allow.items()},
        "allowed_counts": {stage: len(ids) for stage, ids in allow.items()},
        "separate_b6_wt_sample_ids": sorted(separate_b6_controls),
        "separate_b6_wt_policy": "162 E9.5 WT controls from the B6 perturbation cohort; eligible stage but not pooled into the primary CD1 temporal cohort without assay/background checks.",
        "backgrounds": {stage: dict(sorted(counts.items())) for stage, counts in backgrounds.items()},
        "anatomical_regions": {stage: dict(sorted(counts.items())) for stage, counts in regions.items()},
        "access_policy": "For any later expression acquisition, download only individually allowlisted E8.5/E9.5 GSM files; never bulk-download mixed-stage RAW.tar. E9.5 cannot train an E8.5-to-E9.5 backtest. Reverify stage/genotype, license, and challenge disclosure before modeling.",
        "qc_note": "Primary candidate WT/CD1 counts143 E8.5 and999 E9.5. Of289 B6 E9.5 samples,127 are explicit Nkx2.5 mutants and162 WT controls retained as a separate cohort. Published analyzed CD1 counts118/949 still require QC. Old stage-only allowlist superseded for acquisition; no expression used.",
        "training_approved": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_archive_sha256": result["metadata_archive_sha256"], "allowed_counts": result["allowed_counts"], "excluded_counts": result["excluded_counts"], "training_approved": False}))


if __name__ == "__main__":
    main()
