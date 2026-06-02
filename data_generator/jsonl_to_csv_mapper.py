import json
import csv

def jsonl_to_csv_mapper(input_filepath, output_filepath):
    label_map = {
        "ham": "0",
        "spam": "1",
        "phishing": "1"
    }

    with open(input_filepath, 'r', encoding='utf-8') as jsonl_file, \
         open(output_filepath, 'w', encoding='utf-8', newline='') as csv_file:
        
        writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_ALL)
        
        writer.writerow(['email_text', 'label'])
        
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                email_text = record.get('input', '')
                raw_label = record.get('output', '').strip().lower()
                
                email_text = email_text.replace(';', '；')
                
                mapped_label = label_map.get(raw_label, raw_label)
                
                writer.writerow([email_text, mapped_label])
                
            except json.JSONDecodeError:
                print(f"Błąd parsowania JSON dla linii: {line}")
            except KeyError as e:
                print(f"Brak wymaganego klucza {e} w rekordzie: {line}")

if __name__ == "__main__":

    INPUT_FILE = "phishing_dataset_v3.jsonl"
    OUTPUT_FILE = "generated_dataset_v1.csv"
    
    jsonl_to_csv_mapper(INPUT_FILE, OUTPUT_FILE)
    print(f"Zakończono! Dane zostały zmapowane i zapisane do pliku {OUTPUT_FILE}.")