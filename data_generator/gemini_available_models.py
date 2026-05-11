import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("BŁĄD: Nie znaleziono klucza API. Upewnij się, że masz plik .env")
    exit()

client = genai.Client(api_key=API_KEY)

print("Pobieram listę dostępnych modeli...\n")
print("-" * 50)

try:
    for model in client.models.list():
        print(f"Nazwa do skryptu: '{model.name}'")
        print(f"Wyświetlana nazwa: {model.display_name}")
        print("-" * 50)
            
except Exception as e:
    print(f"Wystąpił błąd podczas pobierania listy: {e}")

