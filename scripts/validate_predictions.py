"""Validate competition CSVs without needing image or training dependencies."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def validate(path, num_classes, expected_rows=None, test_root=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError('CSV is empty')
    names, labels = [], []
    for i, row in enumerate(rows, 1):
        if len(row) != 2:
            raise ValueError(f'Row {i}: require exactly two fields')
        name, label = row
        if not name or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Row {i}: require a bare image filename')
        if not re.fullmatch(r'[0-9]{4}', label) or int(label) >= num_classes:
            raise ValueError(f'Row {i}: invalid label {label!r}')
        names.append(name); labels.append(label)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image filenames')
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, got {len(rows)}')
    if test_root:
        ext = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
        actual = {p.name for p in Path(test_root).iterdir() if p.is_file() and p.suffix.lower() in ext}
        if actual != set(names):
            raise ValueError(f'Test filenames differ: missing={len(actual-set(names))}, extra={len(set(names)-actual)}')
    counts = Counter(labels)
    return {'file':str(path), 'rows':len(rows), 'predicted_classes':len(counts),
            'num_classes':num_classes, 'duplicate_filenames':0, 'format_valid':True,
            'note':'Format and label coverage only; no ground truth is available.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv',type=Path)
    ap.add_argument('--num-classes',type=int,required=True)
    ap.add_argument('--expected-rows',type=int)
    ap.add_argument('--test-root',type=Path)
    args=ap.parse_args()
    if not 1 <= args.num_classes <= 10000:
        raise ValueError('num-classes must be between 1 and 10000')
    print(json.dumps(validate(args.csv,args.num_classes,args.expected_rows,args.test_root),indent=2))


if __name__=='__main__': main()
