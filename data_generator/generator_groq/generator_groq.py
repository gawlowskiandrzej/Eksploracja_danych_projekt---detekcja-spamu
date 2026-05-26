# main.py
import os
import random
import time
import sys
import re
import json
from dotenv import load_dotenv
from groq import Groq, RateLimitError

# Import konfiguracji
import config as config

# ==============================================================================
# 1. INICJALIZACJA API
# ==============================================================================
load_dotenv() 
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("\n[!] BŁĄD: Nie znaleziono klucza API dla Groq w pliku .env.")
    sys.exit(1)

client = Groq(api_key=API_KEY)

# ==============================================================================
# 2. POMOCNICZA FUNKCJA MINIFIKACJI PROMPTU
# ==============================================================================
def minify_prompt(text):
    """Usuwa zbędne wielokrotne spacje i nowe linie, oszczędzając tokeny wejściowe."""
    return re.sub(r'\s+', ' ', text).strip()

# ==============================================================================
# 3. LOGIKA AUTO-RESUME
# ==============================================================================
def get_current_counts(filename):
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
                        pass
    except Exception as e:
        print(f"[!] Ostrzeżenie przy odczycie pliku bazy: {e}")
    return ham_count, spam_count

# ==============================================================================
# 4. GŁÓWNA PĘTLA BATCHINGU
# ==============================================================================
def generate_dataset():
    ham_count, spam_count = get_current_counts(config.OUTPUT_FILE)
    
    print(f"\n==================================================")
    print(f"   ZBUDOWANY SYSTEM BATCHINGU Z LIVE LICZNIKIEM")
    print(f"==================================================")
    print(f" -> Język: {config.LANGUAGE}")
    print(f" -> Ham:   {ham_count}/{config.TARGET_PER_CLASS}")
    print(f" -> Spam:  {spam_count}/{config.TARGET_PER_CLASS}")
    print(f" Plik:     {config.OUTPUT_FILE}\n")
    
    in_tokens, out_tokens = 0, 0
    MAX_RETRIES = 5   
    BASE_DELAY = 5    
    
    with open(config.OUTPUT_FILE, 'a', encoding='utf-8') as f:
        try:
            while ham_count < config.TARGET_PER_CLASS or spam_count < config.TARGET_PER_CLASS:
                
                if ham_count <= spam_count and ham_count < config.TARGET_PER_CLASS:
                    current_label = "ham"
                    raw_sys_instr = config.sys_instr_ham
                    current_temp = 0.6
                    current_batch_size = min(config.BATCH_SIZE, config.TARGET_PER_CLASS - ham_count)
                    branza = random.choice(config.BRANZE)
                    scenariusz = random.choice(config.CONTEXTS_HAM)
                    wybrany_kontekst = f"Branża: {branza}. Sytuacja: {scenariusz}."
                    fake_url = f"https://www.oficjalny-portal-{random.randint(10,99)}.com/verification-secure"
                else:
                    current_label = "spam"
                    raw_sys_instr = config.sys_instr_spam
                    current_temp = 0.85
                    current_batch_size = min(config.BATCH_SIZE, config.TARGET_PER_CLASS - spam_count)
                    branza = random.choice(config.BRANZE)
                    scenariusz = random.choice(config.CONTEXTS_SPAM)
                    wybrany_kontekst = f"Branża: {branza}. Sytuacja: {scenariusz}."
                    fake_url = f"https://weryfikacja-konta-{random.randint(100,999)}.secure-auth-update.net"

                current_length_instruction = random.choice(config.length_modifiers)

                raw_user_prompt = (
                    f"Wygeneruj dokładnie {current_batch_size} unikalnych przykładów e-mail jako obiekt JSON. "
                    f"KONTEKST: {wybrany_kontekst} STRUKTURA: {current_length_instruction} "
                )

                sys_prompt = minify_prompt(raw_sys_instr)
                user_prompt = minify_prompt(raw_user_prompt)

                # \r pozwala nadpisać tę samą linijkę w konsoli
                sys.stdout.write(f"\r>> Pobieranie paczki [{current_label.upper()}] (Rozmiar: {current_batch_size}) ... Czekam na API...")
                sys.stdout.flush()
                
                for attempt in range(MAX_RETRIES):
                    try:
                        # Używamy with_raw_response aby mieć dostęp do ukrytych nagłówków HTTP z limitami
                        raw_response = client.chat.completions.with_raw_response.create(
                            model=config.SELECTED_MODEL,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=current_temp,
                            max_tokens=4000,
                            response_format={"type": "json_object"}
                        )
                        
                        # Parsowanie odpowiedzi z surowego obiektu
                        response = raw_response.parse()
                        headers = raw_response.headers
                        
                        # Pobieranie limitów z nagłówków (Tokeny oraz Requesty)
                        # Uwaga: w zależności od Twojego planu w Groq (Free/Paid), API zwraca limity na minutę (TPM) lub na dzień (TPD).
                        rem_tokens = headers.get('x-ratelimit-remaining-tokens', 'Brak danych')
                        rem_requests = headers.get('x-ratelimit-remaining-requests', 'Brak danych')

                        raw_content = response.choices[0].message.content.strip()
                        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

                        if "I’m sorry" in clean_content or "I can't help" in clean_content:
                            sys.stdout.write(f"\n[!] Wykryto blokadę bezpieczeństwa (Guardrail). Ponawiam paczkę...\n")
                            continue 
                        
                        if response.usage:
                            in_tokens += response.usage.prompt_tokens
                            out_tokens += response.usage.completion_tokens
                        
                        parsed_json = json.loads(clean_content)
                        email_list = parsed_json.get("emails", [])

                        if not email_list:
                            raise Exception("Zwrócona lista 'emails' jest pusta.")

                        saved_in_batch = 0
                        for email_text in email_list:
                            if "I’m sorry" in email_text or "I can't help" in email_text or not email_text.strip():
                                continue
                                
                            record = {
                                "instruction": "Sklasyfikuj poniższą wiadomość email. Zdecyduj, czy jest ona zwykłą wiadomością (ham) czy phishingową (spam).", 
                                "input": email_text.strip(), 
                                "output": current_label
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            saved_in_batch += 1

                        f.flush()
                        os.fsync(f.fileno()) 
                        
                        if current_label == "ham":
                            ham_count += saved_in_batch
                        else:
                            spam_count += saved_in_batch
                            
                        # LIVE LICZNIK W KONSOLI (Nadpisuje linijkę "Pobieranie...")
                        # Wyświetla: Postęp, Zużyte Tokeny w sesji oraz Pozostały limit API z nagłówków
                        sys.stdout.write(f"\r[OK] Zapisano: {saved_in_batch} szt. (Ham: {ham_count}, Spam: {spam_count}) | Sesja: {in_tokens+out_tokens:,} tok. | Zapas API: {rem_tokens} tok. / {rem_requests} zapytań\n")
                        sys.stdout.flush()
                        
                        time.sleep(3.0) 
                        break 
                        
                    except json.JSONDecodeError:
                        sys.stdout.write("\n[!] Błąd parsowania JSON. Serwer zwrócił niepoprawny format. Ponawiam próbę...\n")
                        time.sleep(2)
                    except RateLimitError as e:
                        headers = e.response.headers
                        retry_after = headers.get('retry-after')
                        delay = float(retry_after) if retry_after else BASE_DELAY * (2 ** attempt)
                        sys.stdout.write(f"\n[!] Rate Limit (TPM/RPD). Czekam {delay}s...\n")
                        time.sleep(delay)
                    except Exception as e:
                        sys.stdout.write(f"\n[!] Błąd: {e}. Próba {attempt + 1}/{MAX_RETRIES}...\n")
                        time.sleep(5)
                else:
                    print("\n[!] KRYTYCZNY BŁĄD: Paczka porzucona po wyczerpaniu prób.")
                    return

        except KeyboardInterrupt:
            print("\n[!] Przerwano ręcznie. Postęp bezpieczny na dysku.")
            
        print(f"\n==================================================")
        print(f"   STATYSTYKI ZAKOŃCZONEJ SESJI BATCHINGU")
        print(f"==================================================")
        print(f" -> Tokeny wejściowe w tej sesji:  {in_tokens:,}")
        print(f" -> Tokeny wyjściowe w tej sesji: {out_tokens:,}")
        print(f" -> Całkowity koszt sesji:        {in_tokens+out_tokens:,} tokenów")

if __name__ == "__main__":
    generate_dataset()