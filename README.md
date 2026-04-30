# Eksploracja_danych_projekt---detekcja-spamu

## Pobranie datasetu Kaggle

Kod pobierania znajduje sie w notebooku `spam_dataset_quality_analysis.ipynb`.
Uzywa `kagglehub` do pobrania najnowszej wersji datasetu
`ssssws/spam-email-detection-dataset-clean-and-ml-ready` i zapisuje kopie w
`data/spam_email_dataset.csv`.

## Dodawanie wielu datasetow

Datasety konfiguruje sie w liscie `DATASET_CONFIGS` w notebooku. Dla kazdego
zbioru mozna wskazac zrodlo, plik, kolumne z trescia (`text_col`), kolumne z
klasa (`label_col`) oraz mapowanie etykiet przez `spam_values` i `ham_values`.
Dzieki temu obslugiwane sa etykiety liczbowe (`1`/`0`), logiczne
(`true`/`false`) oraz tekstowe (`spam`/`ham`). Po wczytaniu kazdy zbior ma
ujednolicone kolumny `text`, `label` oraz boolowska flage `is_spam`.

Kolejne zbiory z Kaggle dodaje sie przez dopisanie nowego slownika z
`source: "kaggle"`, `kaggle_dataset`, `file_name`, `text_col`, `label_col`,
`spam_values` i `ham_values`. Wszystkie tabelki analityczne iteruja po
`loaded_datasets`, wiec nowy zbior automatycznie pojawi sie w wynikach.
