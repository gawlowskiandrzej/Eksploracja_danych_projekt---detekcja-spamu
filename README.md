# Eksploracja_danych_projekt---detekcja-spamu

## Zainstaluj pakiety

```powershell
python -m pip install -r requirements.txt
```

## Zbiory danych
Zbiory danych są zapisane w `data`
- jakościowy 7000 danych
- jakościowy pochodzący z innego źródła 100
- niejakościowy

## Dokumentacja i wyniki trenowania
Zapisane sa w `documentation`

## Analiza eksploracyjna
Uruchamiana jest w notatnika jupyter notebook. Skonfigurowane sa 3 datasety:
- jakościowy 7000 danych
- jakościowy pochodzący z innego źródła 100
- niejakościowy

W celu wykonania analizy uruchom notatnik powinny pokazać się tabelę z wynikami miar

## Generacja danych jakościowych
Spójrz w README w `data_generator`

## Trening modelu LLaMA / QLoRA

W pliku `classifier/run_llama_classifier.py` znajduje sie logika trenowania, ładowania oraz jego testowania.
Przed uruchomieniem treningu wykonaj następujące kroki:

1. Ustaw parametry aplikacji takie jak ścieżka do modelu LLaMA oraz parametry trenowania w `classifier\config.py`.
2. Upewnij się że odpowiedni zbiór danych jest ustawiony jako zbiór treningowy i testowy również sprawdź jego podział `classifier\config.py`.
3. Uruchom `classifier\run_llama_classifier.py` komendą `python -X utf8 .\classifier\run_llama_classifier.py` rozpocznie się w ten sposób proces uczenia
4. Po zakończeniu treningu model zapiszę się pod ścieżka określoną w pliku `classifier\config.py` llama_train_cfg.output_dir
5. Następnie uruchomi się proces testowania 
6. Po zakończonych testach pojawią się miary pozwalające określić dokładność modelu

