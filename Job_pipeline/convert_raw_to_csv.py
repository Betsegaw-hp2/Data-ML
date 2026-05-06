"""Convert raw JSON files (jobs_has_link.json, no_link_jobs.json) to pipeline-ready CSV.

Maps scraper field names -> pipeline expected field names:
  title, entity_name, created_at, description, city, country, source

Skips records already in processed CSVs (by job_id) when --skip-processed is given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── field mapping helpers ────────────────────────────────────────────────────

def parse_location(loc_str: str) -> tuple[str, str]:
    """Split 'City, Country' string into (city, country)."""
    if not loc_str:
        return "", ""
    parts = [p.strip() for p in loc_str.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return parts[0], ""


def flatten_jobs_has_link(record: dict) -> Dict[str, str]:
    """Map jobs_has_link.json fields -> pipeline CSV fields."""
    city, country = parse_location(record.get("location") or "")
    skills_raw = record.get("skills", [])
    if isinstance(skills_raw, list):
        skills_str = ", ".join(str(s) for s in skills_raw)
    else:
        skills_str = str(skills_raw)
    return {
        "title": record.get("job_title") or "",
        "entity_name": record.get("company_name") or "",
        "created_at": record.get("posted_date") or record.get("scraped_at") or "",
        "description": record.get("description") or "",
        "city": city,
        "country": country,
        "source": "afriwork",
        # pass-through extras (pipeline ignores unknown keys)
        "job_type_raw": record.get("job_type") or "",
        "job_site": record.get("job_site") or "",
        "skills_raw": skills_str,
        "education_qualification": record.get("education_qualification") or "",
        "experience_level": record.get("experience_level") or "",
        "url": record.get("url") or "",
    }


def flatten_no_link_jobs(record: dict) -> Dict[str, str]:
    """Map no_link_jobs.json (Telegram) fields -> pipeline CSV fields."""
    city, country = parse_location(record.get("work_location") or "")
    # Combine description + raw_text for richer content
    desc = record.get("description") or ""
    raw_text = record.get("raw_text") or ""
    combined_desc = desc if desc else raw_text
    return {
        "title": record.get("job_title") or "",
        "entity_name": record.get("company") or "",
        "created_at": record.get("date") or "",
        "description": combined_desc,
        "city": city,
        "country": country,
        "source": f"telegram_{record.get('channel') or 'unknown'}",
        "job_type_raw": record.get("job_type") or "",
        "job_site": "",
        "skills_raw": "",
        "education_qualification": "",
        "experience_level": "",
        "url": "",
    }


CSV_FIELDS = [
    "title", "entity_name", "created_at", "description",
    "city", "country", "source",
    "job_type_raw", "job_site", "skills_raw",
    "education_qualification", "experience_level", "url",
]


# ── processed ID tracking ────────────────────────────────────────────────────

def load_processed_ids(processed_dir: Path) -> set:
    """Load all job_ids from existing processed CSVs."""
    ids = set()
    for csv_file in processed_dir.glob("*.csv"):
        try:
            with csv_file.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    jid = row.get("job_id", "").strip()
                    if jid:
                        ids.add(jid)
        except Exception as e:
            logger.warning("Could not read %s: %s", csv_file, e)
    logger.info("Loaded %d already-processed job_ids", len(ids))
    return ids


# ── main converter ───────────────────────────────────────────────────────────

def convert_file(
    json_path: Path,
    out_csv: Path,
    flatten_fn,
    skip_ids: Optional[set] = None,
) -> tuple[int, int]:
    """Convert a JSON file to pipeline-ready CSV. Returns (written, skipped)."""
    logger.info("Converting %s -> %s", json_path, out_csv)
    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records from %s", len(records), json_path.name)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            flat = flatten_fn(rec)
            # Optionally skip already-processed records
            if skip_ids is not None:
                # Re-compute job_id the same way the pipeline does (title|entity|date_ym|source)
                title = re.sub(r"[^\w\s]", "", (flat.get("title") or "").lower()).strip()
                entity = re.sub(r"[^\w\s]", "", (flat.get("entity_name") or "").lower()).strip()
                date_raw = flat.get("created_at") or ""
                ym = date_raw[:7] if len(date_raw) >= 7 else date_raw
                source = (flat.get("source") or "").lower().strip()
                sig = f"{title}|{entity}|{ym}|{source}"
                jid = hashlib.sha256(sig.encode()).hexdigest()[:16]
                if jid in skip_ids:
                    skipped += 1
                    continue
            writer.writerow({k: flat.get(k, "") for k in CSV_FIELDS})
            written += 1

    logger.info("%s: written=%d skipped=%d", json_path.name, written, skipped)
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw JSON to pipeline-ready CSV.")
    parser.add_argument("--raw-dir", default="Job_pipeline/data/raw")
    parser.add_argument("--out-dir", default="Job_pipeline/data/raw")
    parser.add_argument("--processed-dir", default="Job_pipeline/data/processed",
                        help="Processed dir to check already-done IDs.")
    parser.add_argument("--skip-processed", action="store_true",
                        help="Skip records matching already-processed job_ids.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    skip_ids = None
    if args.skip_processed:
        skip_ids = load_processed_ids(Path(args.processed_dir))

    jobs_has_link = raw_dir / "jobs_has_link.json"
    no_link_jobs = raw_dir / "no_link_jobs.json"

    if jobs_has_link.exists():
        w, s = convert_file(
            jobs_has_link,
            out_dir / "jobs_has_link.csv",
            flatten_jobs_has_link,
            skip_ids,
        )
        print(f"jobs_has_link.json -> jobs_has_link.csv  written={w}  skipped={s}")
    else:
        print(f"WARNING: {jobs_has_link} not found")

    if no_link_jobs.exists():
        w, s = convert_file(
            no_link_jobs,
            out_dir / "no_link_jobs.csv",
            flatten_no_link_jobs,
            skip_ids,
        )
        print(f"no_link_jobs.json -> no_link_jobs.csv  written={w}  skipped={s}")
    else:
        print(f"WARNING: {no_link_jobs} not found")


if __name__ == "__main__":
    main()
