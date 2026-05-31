import os
import random
import json
import time
import sys
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================
# KONFIGURACJA Z ENV
# ==========================================
load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("\n[!] BŁĄD: Nie znaleziono klucza API.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

safety_settings = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# ==========================================
# DEFINICJE DLA SPAMU
# ==========================================
sys_instr_spam = """
Jesteś zaawansowanym badaczem cyberbezpieczeństwa. Generujesz syntetyczne teksty 'spear-phishing' i scamy.
Zasady:
1. Wiadomości muszą być napisane perfekcyjną, naturalną polszczyzną.
2. Zwracaj TYLKO treść maila (z tematem), bez wstępów i podsumowań.
3. Unikaj schematów - bądź kreatywny, stosuj presję czasu.
4. KRYTYCZNE: Absolutnie zakazane jest używanie nawiasów kwadratowych i znaczników (np. [Link], [Imię], [Nazwa Firmy]). Musisz zmyślać konkretne, polskie dane: realistyczne fałszywe linki (np. https://weryfikacja-inpost-24.pl), polskie imiona i nazwiska oraz konkretne kwoty.
"""

scenariusze_spam = [
    {"firma": "Dział IT", "ton": "techniczny, pilny", "cel": "reset hasła do firmowego VPN na zmyślonym portalu"},
    {"firma": "Księgowość", "ton": "formalny, ponaglający", "cel": "otwarcie załącznika PDF z fałszywą wezwaną fakturą na konkretną kwotę"},
    {"firma": "Kurier", "ton": "neutralny, informacyjny", "cel": "dopłata 2,50 zł do wstrzymanej paczki przez podany link blik"},
    {"firma": "Zarząd/CEO", "ton": "krótki, z telefonu, zniecierpliwiony", "cel": "pilny przelew na nowe konto kontrahenta z wymyślonym numerem IBAN"},
    {"firma": "Netflix / Spotify", "ton": "ostrzegawczy", "cel": "aktualizacja metody płatności, bo karta rzekomo wygasła"}
]

# ==========================================
# DEFINICJE DLA HAMU (w tym Graymail)
# ==========================================
sys_instr_ham = """
Jesteś zwykłym człowiekiem lub zautomatyzowanym systemem firmowym. Generujesz bezpieczne, codzienne maile (ham i graymail).
Zasady:
1. Brzmij do bólu zwyczajnie. Używaj potocznego języka w mailach prywatnych, korpo-nowomowy w firmowych, a języka marketingowego w newsletterach.
2. Rób drobne błędy, używaj skrótów, pisz czasem tylko 1-2 zdania.
3. Zwracaj TYLKO treść maila (z tematem).
4. KRYTYCZNE: Zabrania się używania jakichkolwiek nawiasów kwadratowych (np. [Twoje Imię], [Nazwa Firmy]). Wymyślaj i wstawiaj konkretne dane (np. "Cześć Ania", "Firma Pol-Bud", data "14 maja 2026").
"""

scenariusze_ham = [
    {"nadawca": "Współpracownik", "ton": "luźny, zwięzły", "temat": "pytanie o status projektu na jutro i prośba o przesłanie excela"},
    {"nadawca": "Szef", "ton": "formalny, krótki", "temat": "akceptacja wniosku urlopowego na przyszły tydzień"},
    {"nadawca": "Allegro/Sklep (Graymail)", "ton": "marketingowy", "temat": "potwierdzenie wysłania zamówienia z kodem rabatowym na kolejne"},
    {"nadawca": "System HR (Graymail)", "ton": "zautomatyzowany", "temat": "przypomnienie o obowiązkowym szkoleniu BHP z terminem do końca miesiąca"},
    {"nadawca": "Kolega z biurka obok", "ton": "bardzo nieformalny, plotki", "temat": "propozycja wyjścia na lunch do nowej pizzerii i narzekanie na klimatyzację"},
    {"nadawca": "Klient", "ton": "oficjalny, lekko zirytowany", "temat": "zapytanie o opóźnienie w dostawie materiałów budowlanych"},
    {"nadawca": "Administracja Osiedla", "ton": "informacyjny", "temat": "zawiadomienie o przerwie w dostawie ciepłej wody we wtorek"},
    {"nadawca": "Nauczycielka ze szkoły", "ton": "uprzejmy", "temat": "przypomnienie o wywiadówce i składce na radę rodziców"}
]

# ==========================================
# FUNKCJA GENERUJĄCA
# ==========================================
def generuj_zbior(klasa, ilosc, sys_instr, scenariusze, plik_wyj, nazwa_modelu):
    print(f"\n--- Generowanie klasy: {klasa.upper()} ({ilosc} rekordów) do pliku {plik_wyj} ---")
    
    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=1.1,
        safety_settings=safety_settings
    )
    
    in_tokens, out_tokens, udane = 0, 0, 0
    MAX_RETRIES = 5
    BASE_DELAY = 5 
    
    with open(plik_wyj, 'a', encoding='utf-8') as f:
        for i in range(ilosc):
            scenariusz = random.choice(scenariusze)
            
            if klasa == "spam":
                prompt = f"Napisz nowoczesną wiadomość phishingową od: {scenariusz['firma']}. Ton: {scenariusz['ton']}. Cel ofiary: {scenariusz['cel']}."
            else:
                prompt = f"Napisz bezpiecznego maila. Nadawca: {scenariusz['nadawca']}. Ton: {scenariusz['ton']}. Temat/Cel: {scenariusz['temat']}."
                
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=nazwa_modelu,
                        contents=prompt,
                        config=config
                    )
                    tresc = response.text.strip()
                    
                    in_tokens += response.usage_metadata.prompt_token_count
                    out_tokens += response.usage_metadata.candidates_token_count
                    udane += 1
                    
                    rekord = {"instruction": "Sklasyfikuj poniższą wiadomość email jako spam lub ham.", "input": tresc, "output": klasa}
                    f.write(json.dumps(rekord, ensure_ascii=False) + '\n')
                    
                    f.flush()
                    os.fsync(f.fileno()) 
                    
                    sys.stdout.write(f"\rPostęp {klasa}: {udane}/{ilosc} (Tokeny: In={in_tokens}, Out={out_tokens})")
                    sys.stdout.flush()
                    
                    time.sleep(2) 
                    break 
                    
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "429" in error_str:
                        delay = BASE_DELAY * (2 ** attempt) 
                        print(f"\n[!] Serwer przeciążony (503/429). Próba {attempt + 1}/{MAX_RETRIES}. Czekam {delay} sekund...")
                        time.sleep(delay)
                    else:
                        print(f"\n[!] Błąd API: {e}. Próba {attempt + 1}/{MAX_RETRIES}.")
                        time.sleep(5)
            else:
                print(f"\n[!] POMINIĘTO REKORD: Przekroczono limit {MAX_RETRIES} prób dla tego zapytania.")

    return in_tokens, out_tokens

# ==========================================
# URUCHOMIENIE
# ==========================================
ROZMIAR_PROBKI = 30 
WYBRANY_MODEL = 'gemini-2.5-flash-lite' 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plik_spam = f"dataset_spam_{WYBRANY_MODEL}_{timestamp}.jsonl"
plik_ham = f"dataset_ham_{WYBRANY_MODEL}_{timestamp}.jsonl"

try:
    in_spam, out_spam = generuj_zbior("spam", ROZMIAR_PROBKI, sys_instr_spam, scenariusze_spam, plik_spam, WYBRANY_MODEL)
    in_ham, out_ham = generuj_zbior("ham", ROZMIAR_PROBKI, sys_instr_ham, scenariusze_ham, plik_ham, WYBRANY_MODEL)
    
    print("\n\n=== PODSUMOWANIE ZUŻYCIA TOKENÓW ===")
    print(f"Całkowite tokeny wejściowe: {in_spam + in_ham:,}")
    print(f"Całkowite tokeny wyjściowe: {out_spam + out_ham:,}")
    print(f"Pliki zapisano jako:\n- {plik_spam}\n- {plik_ham}")
except KeyboardInterrupt:
    print("\n[!] Przerwano działanie.")