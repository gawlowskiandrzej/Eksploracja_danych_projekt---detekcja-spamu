## Jak odpalić

1. (zainicjalizować) i uruchomić venv (ja venva mam w katalogu całego repo)
`python -m venv venv`
`venv\Scripts\activate` (Windows)
2. `pip install -r requirements.txt`
3. skopuj plik .env.example do pliku .env i uzupełnij własnymi zmiennymi (klucz api groq) (żeby zobaczyć czy klucz api działa można odpalić groq_available_models.py)
4. upewnić się że ma się aktualną wersję brancha. uruchom skrypt. skrypt appenduje nowe wiersze do już istniejącego pliku. można na spokojnie przerywać skrypt, nic się nie stanie

## Jak generować dane wspólnie

Widzi mi się to że każdy z nas odpali generator raz dziennie, on sobie zużyje tokeny, a to co wygeneruje do pliku byśmy łączyli za pomocą pull requestów w całość. Więc wygenerowane wyniki byśmy commitowali na nowym/prywatnym do tego celu branchu i byśmy tworzyli wtedy pull requesty do maina. Wtedy nie będzie problemów z konfliktami znaczących