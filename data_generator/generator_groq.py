import os
import random
import time
import sys
import re
import json
import csv
from dotenv import load_dotenv
from groq import Groq, RateLimitError

# ==============================================================================
# 1. KONFIGURACJA I INICJALIZACJA API
# ==============================================================================
# Ładowanie zmiennych środowiskowych z pliku .env (np. GROQ_API_KEY=gsk_...)
load_dotenv() 

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("\n[!] BŁĄD: Nie znaleziono klucza API dla Groq w pliku .env.")
    print("Upewnij się, że plik .env zawiera linię: GROQ_API_KEY=twój_klucz_api")
    sys.exit(1)

# Inicjalizacja oficjalnego klienta Groq
client = Groq(api_key=API_KEY)

# ==============================================================================
# 2. USTAWIENIA PROJEKTU
# ==============================================================================
TARGET_PER_CLASS = 500         # Docelowa liczba rekordów dla KAŻDEJ z klas (razem 1000)
# SELECTED_MODEL = 'qwen/qwen3-32b' # Model LLM używany do generowania danych
SELECTED_MODEL = 'openai/gpt-oss-120b' # Model LLM używany do generowania danych
OUTPUT_FILE = "phishing_dataset_test.jsonl" # Plik wyjściowy w formacie JSON Lines

# ==============================================================================
# 3. ZAAWANSOWANE PROMPTY SYSTEMOWE (SPAM VS TRUDNY HAM)
# ==============================================================================
# Słowniki powitań - całkowicie uniezależnione od klas (spam/ham czy ai/human)
POWITANIA_FORMALNE = [
    "Szanowny Panie,", 
    "Szanowna Pani,", 
    "Szanowni Państwo,", 
    "Dzień dobry,"
]

POWITANIA_POTOCZNE = [
    "Cześć,", 
    "Hej,", 
    "Cześć! Szybka sprawa,", 
    "Siemanko,"
]
# KLASA: WYRAFINOWANY SPAM (Phishing/Social Engineering)
# Cel: Ma brzmieć jak normalny mail, unikać słów-kluczy, manipulować kontekstem.
sys_instr_spam = """
Jesteś audytorem bezpieczeństwa. Tworzysz zaawansowany przykład wiadomości phishingowej (klasa: SPAM).
ZASADY STYLU:
1. Pisz naturalnym, biznesowym lub prywatnym językiem. Unikaj agresywnego marketingu.
2. BEZWZGLĘDNY ZAKAZ używania słów: "wygrałeś", "promocja", "zarób", "!!!", "darmowy", "loteria".
3. Kluczem do rozpoznania spamu jest INTENCJA (nakłonienie do kliknięcia w złośliwy link), a nie oczywiste słowa-klucze.
4. Link wklej jako surowy tekst (ZAKAZ formatowania Markdown typu [Link](url)).
5. ZERO PLACEHOLDERÓW: Wstawiaj losowe, realistyczne polskie imiona i nazwiska (unikaj Kowalski/Nowak).
6. Zwracaj TYLKO treść maila, zacznij od powitania.
"""

# KLASA: TRUDNY HAM (Legalne wiadomości, które systemy antyspamowe często mylą ze spamem)
# Cel: Naszpikować maila "podejrzanymi" słowami, ale intencja i struktura muszą być w 100% bezpieczne i legalne.
sys_instr_ham = """
Jesteś generatorem danych. Tworzysz legalną, prawdziwą i bezpieczną wiadomość e-mail (klasa: HAM), na którą odbiorca czeka lub się zapisał.
ZASADY STYLU:
1. Wiadomość MUSI zawierać słowa techniczne lub alarmujące (np. "faktura", "wyciąg", "autoryzacja", "aktywacja", "alert bezpieczeństwa", "zmień hasło", "nie odpowiadaj na tego maila").
2. Wiadomość musi kierować do bezpiecznej, oficjalnej domeny instytucji (np. poprzez instrukcję "zaloguj się w aplikacji banku" lub link do oficjalnego portalu).
3. Styl ma być wysoce profesjonalny, automatyczny lub korporacyjny.
4. Link wklej jako surowy tekst (ZAKAZ formatowania Markdown).
5. ZERO PLACEHOLDERÓW: Wstawiaj losowe, realistyczne dane.
6. Zwracaj TYLKO treść maila, zacznij od powitania.
"""

# ==============================================================================
# 4. MATRYCA KONTEKSTOWA (GENERATOR RÓŻNORODNOŚCI)
# ==============================================================================
BRANZE = ["Bankowość/Finanse", "E-commerce/Zakupy online", "Logistyka/Kurierzy", "IT/Administracja", "HR/Rekrutacja", "Telekomunikacja", "Urzędy/Podatki", "Służba zdrowia"]

# Scenariusze dla Spamu (Manipulacje)
CONTEXTS_SPAM = [
    "fałszywa prośba od szefa o pilne zweryfikowanie faktury kosztowej",
    "podszycie się pod dział IT proszący o aktualizację tokenu dostępu z powodu rzekomej awarii serwera",
    "informacja o rzekomej blokadzie konta z powodu podejrzanego logowania z zagranicy",
    "zawiadomienie o konieczności dopłaty do przesyłki, która rzekomo utknęła na cle",
    "fałszywe zapytanie ofertowe od nowego klienta z prośbą o kliknięcie w specyfikację zamówienia"
]

# Scenariusze dla Hamu (Trudne, ale legalne wiadomości)
CONTEXTS_HAM = [
    "oficjalny alert bezpieczeństwa o poprawnym zalogowaniu na konto z nowego urządzenia (np. w roku 2026)",
    "automatyczne powiadomienie systemowe o wystawieniu comiesięcznej faktury i terminie płatności",
    "legalny newsletter z kodem rabatowym dla lojalnych klientów",
    "prośba z działu HR o wypełnienie obowiązkowego kwestionariusza BHP do końca tygodnia",
    "potwierdzenie zmiany regulaminu świadczenia usług drogą elektroniczną z linkiem do pełnej treści"
]

length_modifiers = [
    "Wiadomość musi być BARDZO KRÓTKA (maksymalnie 3 zdania).",
    "Wiadomość ma być ŚREDNIEJ DŁUGOŚCI (około 2 krótkie akapity).",
    "Wiadomość musi być DŁUGA i formalna (minimum 3 akapity, dużo szczegółów proceduralnych)."
]

# ==============================================================================
# 5. LOGIKA WZNAWIANIA PRACY (AUTO-RESUME)
# ==============================================================================
def get_current_counts(filename):
    """
    Funkcja skanuje istniejący plik datasetu (.jsonl), linia po linii,
    i zlicza ile rekordów danej klasy zostało już pomyślnie zapisanych.
    Umożliwia to zatrzymanie i wznowienie skryptu w dowolnym momencie.
    """
    ham_count = 0
    spam_count = 0
    
    if not os.path.exists(filename):
        return ham_count, spam_count
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("output") == "ham":
                            ham_count += 1
                        elif record.get("output") == "spam":
                            spam_count += 1
                    except json.JSONDecodeError:
                        pass # Ignoruj uszkodzone/niepełne linie w pliku
    except Exception as e:
        print(f"[!] Ostrzeżenie przy odczycie pliku bazy: {e}")
        
    return ham_count, spam_count

# ==============================================================================
# 6. GŁÓWNA PĘTLA GENERUJĄCA (GENEROWANIE NAPRZEMIENNE)
# ==============================================================================
def generate_alternating_dataset():
    # Pobierz aktualny stan licznika z pliku (Auto-Resume)
    ham_count, spam_count = get_current_counts(OUTPUT_FILE)
    
    print(f"\n==================================================")
    print(f"   STAN ZBIORU DANYCH (WZNOWIENIE PROCESU)")
    print(f"==================================================")
    print(f" -> AI Spam:    {ham_count}/{TARGET_PER_CLASS}")
    print(f" -> Human Spam: {spam_count}/{TARGET_PER_CLASS}")
    print(f" Plik docelowy: {OUTPUT_FILE}\n")
    
    # Jeśli obie klasy osiągnęły limit, zakończ działanie
    if ham_count >= TARGET_PER_CLASS and spam_count >= TARGET_PER_CLASS:
        print("[+] Zbiór danych jest już w pełni kompletny!")
        return

    # Liczniki tokenów zużytych TYLKO w bieżącej sesji
    in_tokens, out_tokens = 0, 0
    MAX_RETRIES = 5   # Maksymalna liczba prób ponowienia przy błędzie API
    BASE_DELAY = 5    # Bazowy czas oczekiwania (w sekundach) dla algorytmu Exponential Backoff
    
    # Otwarcie pliku w trybie 'a' (append) - dopisywanie na końcu pliku bez nadpisywania
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        try:
            # Pętla działa dopóki choć jedna klasa nie osiągnie celu
            while ham_count < TARGET_PER_CLASS or spam_count < TARGET_PER_CLASS:
                
                if ham_count <= spam_count and ham_count < TARGET_PER_CLASS:
                    current_label = "ham"  # <--- Zmiana etykiety
                    current_sys_instr = sys_instr_ham
                    current_temp = 0.7
                    # Losujemy kontekst bezpieczny
                    wybrany_kontekst = f"Branża: {random.choice(BRANZE)}. Sytuacja: {random.choice(CONTEXTS_HAM)}."
                    fake_url = f"https://www.oficjalny-portal-{random.randint(10,99)}.pl/login"
                else:
                    current_label = "spam" # <--- Zmiana etykiety
                    current_sys_instr = sys_instr_spam
                    current_temp = 0.9     # Wyższa temperatura dla spamu, żeby był bardziej kreatywny
                    # Losujemy kontekst niebezpieczny
                    wybrany_kontekst = f"Branża: {random.choice(BRANZE)}. Sytuacja: {random.choice(CONTEXTS_SPAM)}."
                    fake_url = f"https://weryfikacja-konta-{random.randint(100,999)}.secure-auth-update.com"

                current_length_instruction = random.choice(length_modifiers)

                # Budowanie dynamicznego promptu
                prompt = (
                    f"Wygeneruj treść e-maila w języku polskim.\n"
                    f"KONTEKST SYTUACYJNY: {wybrany_kontekst}\n"
                    f"WYMÓG STRUKTURALNY: {current_length_instruction}\n"
                    f"Wklej w odpowiednim miejscu ten adres URL jako surowy tekst: {fake_url}"
                )
                
                print(f"Generowanie: [{current_label.upper()}] ... ", end="")
                sys.stdout.flush()
                
                # Pętla obsługi błędów API (Retries)
                for attempt in range(MAX_RETRIES):
                    try:
                        # Wysłanie zapytania do API Groq
                        response = client.chat.completions.create(
                            model=SELECTED_MODEL,
                            messages=[
                                {"role": "system", "content": current_sys_instr},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=current_temp,
                        )
                        
                        # Pobranie i czyszczenie tekstu z ewentualnych tagów myślenia (<think>)
                        raw_content = response.choices[0].message.content.strip()
                        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

                        if "I’m sorry" in clean_content or "I can't help" in clean_content:
                            print("\n[!] Model zablokował odpowiedź (Guardrail). Ponawiam próbę...")
                            print(clean_content)
                            continue # Wymusza ponowne odpytanie API w ramach pętli prób (MAX_RETRIES)
                        
                        if '<think>' in clean_content:
                             clean_content = clean_content.split('<think>')[0].strip()

                        # Jeśli odpytanie zwróciło pusty ciąg, zgłoś błąd i ponów próbę
                        if not clean_content:
                             raise Exception("Model zwrócił pustą treść po odfiltrowaniu procesu myślowego.")
                        
                        # Zliczanie tokenów (jeśli API zwraca obiekt usage)
                        if response.usage:
                            in_tokens += response.usage.prompt_tokens
                            out_tokens += response.usage.completion_tokens
                        
                        # Przygotowanie struktury rekordu pod fine-tuning (format Alpaca/Instruct)
                        record = {
                            "instruction": "Sklasyfikuj poniższą wiadomość email. Zdecyduj, czy jest ona zwykłą wiadomością (ham) czy phishingową (spam).", 
                            "input": clean_content, 
                            "output": current_label
                        }
                        
                        # Bezpieczny zapis linii do pliku JSONL
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        
                        # Wymuszenie natychmiastowego zapisu z bufora pamięci na dysk (ochrona przed utratą danych)
                        f.flush()
                        os.fsync(f.fileno()) 
                        
                        # Aktualizacja lokalnych liczników stanów
                        if current_label == "ham":
                            ham_count += 1
                        else:
                            spam_count += 1
                            
                        print(f"SUKCES! (Postęp ogólny: AI={ham_count}/{TARGET_PER_CLASS}, Human={spam_count}/{TARGET_PER_CLASS})")
                        
                        # Krótka przerwa między zapytaniami, aby nie przekroczyć limitów TPM (Tokens Per Minute)
                        time.sleep(2.5) 
                        break # Wyjście z pętli ponowień (attempt) – przechodzimy do kolejnego rekordu
                        
                    except RateLimitError as e:
                        # Pobranie nagłówków z odpowiedzi API
                        headers = e.response.headers
                        rem_requests = headers.get('x-ratelimit-remaining-requests')
                        rem_tokens = headers.get('x-ratelimit-remaining-tokens')
                        retry_after = headers.get('retry-after')

                        # Bezpieczne parsowanie na int (w razie nietypowych wartości w nagłówkach)
                        def safe_int(val):
                            try: return int(val)
                            except (TypeError, ValueError): return 1 # Zwraca > 0 w przypadku błędu

                        # Identyfikacja, który z limitów spadł do zera
                        limit_type = "Rate Limit (Nieokreślony)"
                        if rem_tokens is not None and safe_int(rem_tokens) <= 0:
                            limit_type = "Tokens Per Minute (TPM)"
                        elif rem_requests is not None and safe_int(rem_requests) <= 0:
                            limit_type = "Requests Per Day (RPD)"

                        # Wykorzystanie dynamicznego czasu 'retry-after', jeśli to możliwe
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = BASE_DELAY * (2 ** attempt) # Fallback do backoffu
                        else:
                            delay = BASE_DELAY * (2 ** attempt)

                        print(f"\n[!] Przekroczono limit: {limit_type}. Próba {attempt + 1}/{MAX_RETRIES}. Oczekiwanie {delay}s...")
                        time.sleep(delay)

                    except Exception as e:
                        error_str = str(e).lower()
                        # Oddzielna obsługa błędów przeciążenia serwerów Groq (HTTP 503)
                        if "503" in error_str or "service unavailable" in error_str:
                            delay = BASE_DELAY * (2 ** attempt)
                            print(f"\n[!] Przeciążenie serwera (HTTP 503). Próba {attempt + 1}/{MAX_RETRIES}. Oczekiwanie {delay}s...")
                            time.sleep(delay)
                        else:
                            print(f"\n[!] Błąd komunikacji z API: {e}. Próba {attempt + 1}/{MAX_RETRIES} za 5 sekund...")
                            time.sleep(5)
                else:
                    # Wykonuje się tylko, gdy pętla 'for attempt' nie zostanie przerwana przez 'break' (wyczerpanie prób)
                    print(f"\n[!] KRYTYCZNY BŁĄD: Przekroczono limit {MAX_RETRIES} prób dla jednego rekordu. Przerywam sesję w celu ochrony limitów.")
                    return

        except KeyboardInterrupt:
            # Łagodne obsłużenie kombinacji Ctrl+C przez użytkownika
            print("\n[!] Wykryto przerwanie ręczne (Ctrl+C). Zamykanie sesji... Postęp został bezpiecznie zachowany.")
            
        print(f"\n==================================================")
        print(f"   PODSUMOWANIE ZAKOŃCZONEJ SESJI")
        print(f"==================================================")
        print(f" -> Zużyte tokeny wejściowe (Prompt): {in_tokens:,}")
        print(f" -> Zużyte tokeny wyjściowe (Completion): {out_tokens:,}")
        print(f" -> Aktualny stan pliku: AI={ham_count}, Human={spam_count}")

# ==============================================================================
# 7. URUCHOMIENIE PROGRAMU
# ==============================================================================
if __name__ == "__main__":
    generate_alternating_dataset()