import csv
import os

import numpy as np


def collect_fieldnames(rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def normalize_cell(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    return value


def write_rows_to_csv(path, rows, fieldnames=None):
    rows = rows or []
    fieldnames = fieldnames or collect_fieldnames(rows)
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_cell(row.get(key, "")) for key in fieldnames})


def read_rows_from_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_single_row_csv(path, row):
    write_rows_to_csv(path, [row], fieldnames=list(row.keys()))


def write_matrix_to_csv(path, matrix, row_names=None, col_names=None, row_label="Row"):
    matrix = np.asarray(matrix)
    row_names = row_names or [str(i) for i in range(matrix.shape[0])]
    col_names = col_names or [str(i) for i in range(matrix.shape[1])]

    rows = []
    for row_name, values in zip(row_names, matrix):
        row = {row_label: row_name}
        for col_name, value in zip(col_names, values):
            row[col_name] = float(value)
        rows.append(row)
    write_rows_to_csv(path, rows, fieldnames=[row_label] + list(col_names))


def merge_rows_on_key(base_rows, summary_rows, key="Method"):
    if not base_rows:
        return []
    summary_map = {row[key]: row for row in summary_rows if key in row}
    merged_rows = []
    for row in base_rows:
        merged = dict(row)
        summary = summary_map.get(row.get(key))
        if summary:
            for summary_key, summary_value in summary.items():
                if summary_key == key:
                    continue
                merged[summary_key] = summary_value
        merged_rows.append(merged)
    return merged_rows


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan", "NaN", "None", "Failed"}:
        return True
    return False


def to_float(value, default=np.nan):
    if is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_ignore_nan(values):
    values = np.asarray([to_float(v) for v in values], dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def std_ignore_nan(values):
    values = np.asarray([to_float(v) for v in values], dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanstd(values))


def filter_numeric_rows(rows, *columns):
    filtered = []
    for row in rows:
        keep = True
        for column in columns:
            if np.isnan(to_float(row.get(column))):
                keep = False
                break
        if keep:
            filtered.append(row)
    return filtered


def sort_rows(rows, *sort_keys):
    def _key(row):
        key_values = []
        for sort_key in sort_keys:
            value = row.get(sort_key)
            numeric = to_float(value)
            key_values.append(numeric if not np.isnan(numeric) else value)
        return tuple(key_values)

    return sorted(rows, key=_key)


def group_rows_by(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get(key), []).append(row)
    return grouped
