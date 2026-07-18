"""Resolve model DATE labels to SDATE / EDATE within project entries."""

from __future__ import annotations

import re


def resolve_dates_to_sdate_edate(labels: list[str], project_segments: list[dict]) -> list[str]:
    resolved: list[str] = []
    dates_in_entry: list[int] = []

    for i, lbl in enumerate(labels):
        seg = project_segments[i]

        if lbl == "PROJECT_NAME":
            for idx in dates_in_entry:
                resolved[idx] = "SDATE"
            dates_in_entry = []

        if lbl == "DATE":
            dates_in_entry.append(i)
            clean_txt = seg.get("text", "").strip()
            if re.search(r"\b(?:present|date|till)\b", clean_txt, re.IGNORECASE):
                resolved.append("EDATE")
                if len(dates_in_entry) > 1:
                    resolved[dates_in_entry[-2]] = "SDATE"
                try:
                    dates_in_entry.remove(i)
                except ValueError:
                    pass
            elif seg.get("_is_split_date_range"):
                resolved.append("EDATE" if seg.get("_is_split_right") else "SDATE")
                try:
                    dates_in_entry.remove(i)
                except ValueError:
                    pass
            else:
                resolved.append("SDATE")
        else:
            resolved.append(lbl)

        if len(dates_in_entry) == 2:
            resolved[dates_in_entry[0]] = "SDATE"
            resolved[dates_in_entry[1]] = "EDATE"
            dates_in_entry = []

    for idx in dates_in_entry:
        resolved[idx] = "SDATE"

    return resolved
