import json
import csv
import os
import sys
from pathlib import Path
import importlib.util

ROW_NUM = 6000
def _load_models_cascade():
    # Try to import MODELS_CASCADE from generator_groq.config; fallback to loading by path
    try:
        from generator_groq.config import MODELS_CASCADE
        return MODELS_CASCADE
    except Exception:
        # Fallback: locate config.py relative to this file
        cfg_path = Path(__file__).resolve().parent / 'generator_groq' / 'config.py'
        if cfg_path.exists():
            spec = importlib.util.spec_from_file_location('generator_groq.config', str(cfg_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, 'MODELS_CASCADE', [])
        return []


def jsonl_dir_to_csv(input_dir, output_filepath):
    label_map = {
        "ham": "0",
        "spam": "1",
        "phishing": "1"
    }

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input path {input_dir} is not a directory")

    # Collect records per model
    records_by_model = {}
    jsonl_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix == '.jsonl'])

    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"JSON parse error in file {file_path}: {line}")
                        continue

                    email_text = record.get('input', '')
                    raw_label = record.get('output', '').strip().lower()
                    model_name = record.get('generated_by') or record.get('model') or record.get('model_name') or 'unknown'

                    email_text = email_text.replace(';', '；')
                    mapped_label = label_map.get(raw_label, raw_label)

                    records_by_model.setdefault(model_name, []).append((email_text, mapped_label))
        except Exception as e:
            print(f"Błąd czytania pliku {file_path}: {e}")

    # Build ordered model list based on config
    cascade = _load_models_cascade() or []
    ordered_models = []
    for m in cascade:
        if m in records_by_model:
            ordered_models.append(m)

    # append remaining models that were present but not in cascade, preserving natural order
    for m in records_by_model.keys():
        if m not in ordered_models:
            ordered_models.append(m)

    # Write CSV
    with open(output_filepath, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writerow(['email_text', 'label'])
        curr_rows=0
        for model in ordered_models:
            rows = records_by_model.get(model, [])
            for email_text, mapped_label in rows:
                writer.writerow([email_text, mapped_label])
                curr_rows+=1
            if(curr_rows>=ROW_NUM):
                print("6k rows, exiting early for higher average quality")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Map JSONL files in a directory to a single CSV, prioritising models by MODELS_CASCADE order from config.')
    parser.add_argument('input_dir', nargs='?', default='.', help='Directory containing .jsonl files')
    parser.add_argument('output_file', nargs='?', default='generated_dataset_v1.csv', help='Output CSV file path')

    args = parser.parse_args()

    try:
        jsonl_dir_to_csv(args.input_dir, args.output_file)
        print(f"Zakończono! Dane zostały zmapowane i zapisane do pliku {args.output_file}.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")