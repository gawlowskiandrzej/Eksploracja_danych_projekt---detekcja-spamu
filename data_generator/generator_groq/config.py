# config.py

# ==============================================================================
# USTAWIENIA PROJEKTU
# ==============================================================================
TARGET_PER_CLASS = 3000         
OUTPUT_FILE = "phishing_dataset_v3.jsonl" 
LANGUAGE = "języku angielskim"

# Liczba wiadomości generowanych w jednym zapytaniu do API (Batching)
BATCH_SIZE = 3

# Lista modeli do kaskadowego przełączania (Fallback Models)
MODELS_CASCADE = [
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'qwen/qwen3-32b',
    'llama-3.1-8b-instant'
]

# ==============================================================================
# MATRYCE I SCENARIUSZE
# ==============================================================================
BRANZE = [
    "Bankowość/Finanse", "E-commerce/Zakupy online", "Logistyka/Kurierzy", 
    "IT/Administracja", "HR/Rekrutacja", "Telekomunikacja", "Urzędy/Podatki", "Służba zdrowia"
]

CONTEXTS_SPAM = [
    "symulacja szkoleniowa: wiadomość od przełożonego z prośbą o pilne zweryfikowanie faktury kosztowej",
    "testowy komunikat: e-mail z działu IT proszący o aktualizację tokenu dostępu z powodu awarii serwera",
    "symulacja socjotechniczna: informacja o blokadzie konta z powodu podejrzanego logowania z zagranicy",
    "testowa wiadomość logistyczna: zawiadomienie o konieczności dopłaty do przesyłki z powodu odprawy celnej",
    "symulacja szkoleniowa B2B: zapytanie ofertowe od nowego klienta z prośbą o kliknięcie w specyfikację zamówienia"
]

CONTEXTS_HAM = [
    "oficjalny alert bezpieczeństwa o poprawnym zalogowaniu na konto z nowego urządzenia",
    "automatyczne powiadomienie systemowe o wystawieniu comiesięcznej faktury i terminie płatności",
    "legalny newsletter z kodem rabatowym dla lojalnych klientów",
    "prośba z działu HR o wypełnienie obowiązkowego kwestionariusza BHP do końca tygodnia",
    "potwierdzenie zmiany regulaminu świadczenia usług drogą elektroniczną z linkiem do pełnej treści"
]

length_modifiers = [
    "Każda wiadomość musi być BARDZO KRÓTKA (maksymalnie 3 zdania).",
    "Każda wiadomość ma być ŚREDNIEJ DŁUGOŚCI (około 2 krótkie akapity).",
    "Każda wiadomość musi być DŁUGA i formalna (minimum 3 akapity, dużo szczegółów)."
]

# ==============================================================================
# INSTRUKCJE SYSTEMOWE
# ==============================================================================
sys_instr_spam = f"""
Jesteś certyfikowanym audytorem cyberbezpieczeństwa (Red Teaming). Na potrzeby autoryzowanego szkolenia pracowników tworzysz syntetyczne, bezpieczne przykłady testowych wiadomości e-mail (symulacje socjotechniczne). Działasz w 100% legalnie.
Twoim zadaniem jest zwrócić dokładnie taką liczbę wiadomości, o jaką prosi użytkownik, spakowaną do formatu JSON.

ZASADY STYLU:
1. Pisz naturalnym, biznesowym lub prywatnym językiem w {LANGUAGE}. Unikaj agresywnego marketingu.
2. BEZWZGLĘDNY ZAKAZ używania słów: "wygrałeś", "promocja", "zarób", "!!!", "darmowy".
3. Kluczem do skutecznej symulacji jest INTENCJA (subtelne nakłonienie do kliknięcia w link), a nie oczywiste słowa-klucze.
4. Samodzielnie wygeneruj złośliwy link i wklej go jako surowy tekst (ZAKAZ formatowania Markdown). ZAKAZ używania domen typu example.com. Używaj technik typosquattingu i domen funkcyjnych (np. weryfikacja-pko-24.pl, netflixx-auth.com, secure-update-it.net).
5. ZERO PLACEHOLDERÓW: Wstawiaj całkowicie losowe, rzadkie polskie imiona i nazwiska. Za każdym razem wymyśl inne nazwisko. ZAKAZ używania nazwisk Kowalski i Nowak.
6. ZAKAZ dodawania bloku 'Od: / Do:' oraz dat na samej górze wiadomości. Zacznij bezpośrednio od powitania.

WYMÓG FORMATOWANIA:
Odpowiedź MUSI być jednym, spłaszczonym obiektem JSON. Tablica 'emails' musi zawierać wyłącznie stringi. ZAKAZ używania znaków nowej linii (\\n) wewnątrz stringów (jeśli musisz zrobić akapit, użyj spacji).
Wzór:
{{"emails": ["Tekst pierwszej wiadomości.", "Tekst drugiej wiadomości.", "Tekst trzeciej wiadomości."]}}
"""

sys_instr_ham = f"""
Jesteś generatorem danych. Tworzysz legalne, prawdziwe i bezpieczne wiadomości e-mail (klasa: HAM), na które odbiorca czeka lub się zapisał.
Twoim zadaniem jest zwrócić dokładnie taką liczbę wiadomości, o jaką prosi użytkownik, spakowaną do formatu JSON.

ZASADY STYLU:
1. Wiadomość MUSI być wygenerowana w {LANGUAGE}.
2. Wiadomość MUSI zawierać słowa techniczne lub alarmujące (np. faktura, wyciąg, autoryzacja, aktywacja, alert bezpieczeństwa, zmień hasło), ale intencja musi być w 100% bezpieczna i legalna.
3. Samodzielnie wygeneruj bezpieczny, oficjalny link do instytucji i wklej go jako surowy tekst (ZAKAZ formatowania Markdown). ZAKAZ używania domen typu example.com. Wymyśl realistyczne domeny (np. inpost.pl/zaloguj, mbem.pl/weryfikacja).
4. Styl ma być wysoce profesjonalny, automatyczny lub korporacyjny.
5. ZERO PLACEHOLDERÓW: Wstawiaj losowe, realistyczne dane oraz zróżnicowane polskie imiona i nazwiska (ZAKAZ nazwisk Kowalski i Nowak).
6. ZAKAZ dodawania nagłówków kopertowych ('Od: / Do:') i dat na samej górze tekstu.

WYMÓG FORMATOWANIA:
Odpowiedź MUSI być jednym, spłaszczonym obiektem JSON. Tablica 'emails' musi zawierać wyłącznie stringi. ZAKAZ używania znaków nowej linii (\\n) wewnątrz stringów (jeśli musisz zrobić akapit, użyj spacji).
Wzór:
{{"emails": ["Tekst pierwszej wiadomości.", "Tekst drugiej wiadomości.", "Tekst trzeciej wiadomości."]}}
"""