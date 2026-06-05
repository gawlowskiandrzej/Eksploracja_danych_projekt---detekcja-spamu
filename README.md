# Eksploracja_danych_projekt---detekcja-spamu

## Instalacja zaleznosci

Zainstaluj pakiety w tym samym interpreterze Pythona, ktorego uzywasz do
uruchamiania projektu:

```powershell
python -m pip install -r requirements.txt
```

Jesli blad `No module named peft` pojawia sie w notebooku, sprawdz interpreter
aktywny w kernelu:

```python
import sys
print(sys.executable)
```

Nastepnie uruchom instalacje przez ten sam interpreter albo zmien kernel
notebooka na srodowisko, w ktorym zaleznosci sa juz zainstalowane.

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

## Trening modelu LLaMA / QLoRA

W pliku `classifier/main.py` znajduje sie logika trenowania i ładowania modelu.
Przed uruchomieniem treningu wykonaj następujące kroki:

1. Ustaw lokalną ścieżkę do modelu LLaMA w `llama_cfg.local_dir`.
2. Ustaw ścieżkę do pliku datasetu w `config1.csv_path`.
3. Określ rozkład danych treningowych i testowych poprzez parametr
   `config1.test_size` (np. `0.2` = 20% danych testowych).
4. Odkomentuj linię:
   `train(llama, prompt_builder, df_train, df_test, llama_train_cfg, config_setup)`
   aby uruchomić trening.
5. Po zakończeniu treningu zakomentuj powyższą linię i upewnij się, że
   `adapter_path` w wywołaniu `classifier.fit(..., adapter_path=...)`
   wskazuje na katalog z zapisanym adapterem/fintunowanym modelem.
6. Próbkę testową do oceny definiuje zmienna `sample` w `main()`.

