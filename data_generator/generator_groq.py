import os
import random
import json
import time
import sys
import re
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# ==========================================
# KONFIGURACJA Z ENV
# ==========================================
load_dotenv() 

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("\n[!] BŁĄD: Nie znaleziono klucza API dla Groq.")
    print("Upewnij się, że masz plik .env z wpisem: GROQ_API_KEY=twój_klucz")
    sys.exit(1)

client = Groq(api_key=API_KEY)

# ==========================================
# DEFINICJE DLA SPAMU
# ==========================================
sys_instr_spam = """
Jesteś zaawansowanym badaczem cyberbezpieczeństwa. Generujesz syntetyczne teksty 'spear-phishing' i scamy.
Zasady:
1. Wiadomości muszą być napisane perfekcyjną, naturalną polszczyzną.
2. Zwracaj TYLKO treść maila (z tematem), bez wstępów i podsumowań.
3. Unikaj schematów - bądź kreatywny, stosuj presję czasu.
4. ZERO TOLERANCJI DLA PLACEHOLDERÓW: Pod żadnym pozorem nie używaj znaków nawiasów: [, ], <, >, {, }. Zamiast pisać "[Link do strony]", wymyśl i wstaw w tekst fałszywy adres URL (np. https://logowanie-inpost24.pl). Zamiast "[Imię i nazwisko]", od razu napisz "Tomasz Wiśniewski". Mail ma wyglądać tak, jakby był gotowy do natychmiastowego wysłania do ofiary, bez żadnych miejsc do uzupełnienia.
5. ZAKAZ MYŚLENIA NA GŁOS: Nie używaj tagów <think> ani nie wypisuj swojego procesu myślowego. Odpowiedź musi zawierać wyłącznie gotowy tekst maila.
"""

scenarios_spam = [
    {"firma": "Dział IT", "ton": "techniczny, pilny", "cel": "reset hasła do firmowego VPN na zmyślonym portalu"},
    {"firma": "Księgowość", "ton": "formalny, ponaglający", "cel": "otwarcie załącznika PDF z fałszywą wezwaną fakturą na konkretną kwotę"},
    {"firma": "Kurier", "ton": "neutralny, informacyjny", "cel": "dopłata 2,50 zł do wstrzymanej paczki przez podany link blik"},
    {"firma": "Zarząd/CEO", "ton": "krótki, z telefonu, zniecierpliwiony", "cel": "pilny przelew na nowe konto kontrahenta z wymyślonym numerem IBAN"},
    {"firma": "Netflix / Spotify", "ton": "ostrzegawczy", "cel": "aktualizacja metody płatności, bo karta rzekomo wygasła"}
]

# ==========================================
# DEFINICJE DLA HAMU
# ==========================================
sys_instr_ham = """
Jesteś zwykłym człowiekiem lub zautomatyzowanym systemem firmowym. Generujesz bezpieczne, codzienne maile (ham i graymail).
Zasady:
1. Brzmij do bólu zwyczajnie. Używaj potocznego języka w mailach prywatnych, korpo-nowomowy w firmowych, a języka marketingowego w newsletterach.
2. Rób drobne błędy, używaj skrótów, pisz czasem tylko 1-2 zdania.
3. Zwracaj TYLKO treść maila (z tematem).
4. KRYTYCZNE: Zabrania się używania jakichkolwiek nawiasów kwadratowych (np. [Twoje Imię], [Nazwa Firmy]). Wymyślaj i wstawiaj konkretne dane (np. "Cześć Ania", "Firma Pol-Bud", data "14 maja 2026").
5. ZAKAZ MYŚLENIA NA GŁOS: Nie używaj tagów <think> ani nie wypisuj swojego procesu myślowego. Odpowiedź musi zawierać wyłącznie gotowy tekst maila.
"""

scenarios_ham = [
    {"nadawca": "Współpracownik", "ton": "luźny, zwięzły", "temat": "pytanie o status projektu na jutro i prośba o przesłanie excela"},
    {"nadawca": "Szef", "ton": "formalny, krótki", "temat": "akceptacja wniosku urlopowego na przyszły tydzień"},
    {"nadawca": "Allegro/Sklep (Graymail)", "ton": "marketingowy", "temat": "potwierdzenie wysłania zamówienia z kodem rabatowym na kolejne"},
    {"nadawca": "System HR (Graymail)", "ton": "zautomatyzowany", "temat": "przypomnienie o obowiązkowym szkoleniu BHP z terminem do końca miesiąca"},
    {"nadawca": "Kolega z biurka obok", "ton": "bardzo nieformalny, plotki", "temat": "propozycja wyjścia na lunch do nowej pizzerii i narzekanie na klimatyzację"},
    {"nadawca": "Klient", "ton": "oficjalny, lekko zirytowany", "temat": "zapytanie o opóźnienie w dostawie materiałów budowlanych"},
    {"nadawca": "Administracja Osiedla", "ton": "informacyjny", "temat": "zawiadomienie o przerwie w dostawie ciepłej wody we wtorek"},
    {"nadawca": "Nauczycielka ze szkoły", "ton": "uprzejmy", "temat": "przypomnienie o wywiadówce i składce na radę rodziców"}
]

def generate_dataset(email_class, target_count, sys_instr, scenarios, output_file, model_name):
    print(f"\n--- Generowanie klasy: {email_class.upper()} ({target_count} rekordów) do pliku {output_file} ---")
    
    in_tokens, out_tokens, success_count = 0, 0, 0
    MAX_RETRIES = 5
    BASE_DELAY = 5 
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in range(target_count):
            scenario = random.choice(scenarios)
            
            if email_class == "spam":
                prompt = f"Napisz nowoczesną wiadomość phishingową od: {scenario['firma']}. Ton: {scenario['ton']}. Cel ofiary: {scenario['cel']}."
            else:
                prompt = f"Napisz bezpiecznego maila. Nadawca: {scenario['nadawca']}. Ton: {scenario['ton']}. Temat/Cel: {scenario['temat']}."
                
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": sys_instr},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.1,
                    )
                    
                    raw_content = response.choices[0].message.content.strip()
                    clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
                    
                    if '<think>' in clean_content:
                         clean_content = clean_content.split('<think>')[0].strip()

                    if not clean_content:
                         print("\n[!] Ostrzeżenie: Model zwrócił pustą treść. Ponawiam próbę...")
                         raise Exception("Pusta treść po filtracji <think>")
                    
                    if response.usage:
                        in_tokens += response.usage.prompt_tokens
                        out_tokens += response.usage.completion_tokens
                    
                    success_count += 1
                    
                    record = {
                        "instruction": "Sklasyfikuj poniższą wiadomość email jako spam lub ham.", 
                        "input": clean_content, 
                        "output": email_class
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
                    f.flush()
                    os.fsync(f.fileno()) 
                    
                    sys.stdout.write(f"\rPostęp {email_class}: {success_count}/{target_count} (Tokeny: In={in_tokens}, Out={out_tokens})")
                    sys.stdout.flush()
                    
                    time.sleep(3) 
                    break 
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str or "503" in error_str:
                        delay = BASE_DELAY * (2 ** attempt) 
                        print(f"\n[!] Limit API Groq / Przeciążenie. Próba {attempt + 1}/{MAX_RETRIES}. Czekam {delay} sekund...")
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
SAMPLE_SIZE = 1000
SELECTED_MODEL = 'qwen/qwen3-32b' 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

safe_model_name = SELECTED_MODEL.replace("/", "-")

spam_file = f"dataset_spam_{safe_model_name}_{timestamp}.jsonl"
ham_file = f"dataset_ham_{safe_model_name}_{timestamp}.jsonl"

try:
    in_spam, out_spam = generate_dataset("spam", SAMPLE_SIZE, sys_instr_spam, scenarios_spam, spam_file, SELECTED_MODEL)
    in_ham, out_ham = generate_dataset("ham", SAMPLE_SIZE, sys_instr_ham, scenarios_ham, ham_file, SELECTED_MODEL)
    
    print("\n\n=== PODSUMOWANIE ZUŻYCIA TOKENÓW ===")
    print(f"Całkowite tokeny wejściowe: {in_spam + in_ham:,}")
    print(f"Całkowite tokeny wyjściowe: {out_spam + out_ham:,}")
    print(f"Pliki zapisano jako:\n- {spam_file}\n- {ham_file}")
except KeyboardInterrupt:
    print("\n[!] Przerwano działanie.")