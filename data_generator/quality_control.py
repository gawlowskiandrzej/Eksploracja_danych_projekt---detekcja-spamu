import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class EmailQualityController:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts = []
        self.klasa = ""
        self._wczytaj_dane()

    def _wczytaj_dane(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                rekord = json.loads(line)
                self.texts.append(rekord['input'])
                self.klasa = rekord['output']
        self.num_records = len(self.texts)

    def analyze_similarity(self):
        if self.num_records < 2: return 0.0
        tfidf = TfidfVectorizer().fit_transform(self.texts)
        sim_matrix = cosine_similarity(tfidf)
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix.max(), sim_matrix.mean()

    def analyze_length(self):
        lengths = [len(t.split()) for t in self.texts]
        return np.mean(lengths), np.std(lengths)

    def detect_artifacts(self):
        # Szukamy brzydkich nawiasów, sztampowych zwrotów itp.
        artifacts = [r'\[.*?\]', r'<.*?>', r'Szanowny Kliencie', r'Wiadomość wygenerowana automatycznie przez AI']
        count = sum(1 for t in self.texts if any(re.search(a, t, re.IGNORECASE) for a in artifacts))
        return count / self.num_records

    def generuj_raport(self):
        if self.num_records == 0:
            return f"Brak danych w pliku {self.file_path}!"

        max_sim, avg_sim = self.analyze_similarity()
        mean_len, std_len = self.analyze_length()
        art_ratio = self.detect_artifacts()

        raport = f"\n=== RAPORT DLA ZBIORU: {self.klasa.upper()} ({self.file_path}) ===\n"
        raport += f"Rekordów: {self.num_records}\n"
        raport += f"Zróżnicowanie tekstów (Max Cosine Sim): {max_sim:.2f} (im mniej, tym lepiej)\n"
        raport += f"Średnia długość (słowa): {mean_len:.0f} (Odchylenie: {std_len:.0f})\n"
        raport += f"Proporcja błędów/znaczników AI: {art_ratio*100:.1f}%\n"
        raport += "\n[SUGEROWANY FEEDBACK DLA GEMINI NA NASTĘPNĄ ITERACJĘ]:\n"

        feedback = []
        if max_sim > 0.65:
            feedback.append(f"- BŁĄD: Teksty {self.klasa} są zbyt podobne (MaxSim: {max_sim:.2f}). Używaj drastycznie różnego słownictwa, budowy zdań i tematów.")
        if std_len < 15:
            feedback.append(f"- BŁĄD: Wszystkie maile mają podobną długość (Std: {std_len:.0f}). Wprowadź losowość: od bardzo krótkich (1 zdanie) po długie.")
        if art_ratio > 0.1:
            feedback.append(f"- BŁĄD: Zbyt wiele tekstów ({art_ratio*100:.0f}%) zawiera nawiasy [Placeholder] lub sztuczne formułki. Wymyślaj fałszywe, ale konkretne dane bez używania nawiasów.")

        if not feedback:
            raport += "-> Próbka jest bardzo dobra! Parametry różnorodności w normie. Nie trzeba zmieniać promptu."
        else:
            raport += "\n".join(feedback)

        return raport

# ==========================================
# URUCHOMIENIE KONTROLI
# ==========================================
try:
    kontroler_spam = EmailQualityController("dataset_spam_qwen-qwen3-32b_20260511_151955.jsonl")
    print(kontroler_spam.generuj_raport())

    kontroler_ham = EmailQualityController("dataset_ham_qwen-qwen3-32b_20260511_151955.jsonl")
    print(kontroler_ham.generuj_raport())
except FileNotFoundError as e:
    print(f"Najpierw wygeneruj pliki! Błąd: {e}")