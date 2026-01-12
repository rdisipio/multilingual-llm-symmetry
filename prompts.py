
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Example prompts for quick testing (replace with dataset-driven prompts later)
PROMPTS = [
    # Factual
    {
        "id": "fact_1",
        "en": "Who discovered penicillin?",
        "fr": "Qui a découvert la pénicilline ?",
        "type": "factual",
    },
    {
        "id": "fact_2",
        "en": "What is the capital of Japan?",
        "fr": "Quelle est la capitale du Japon ?",
        "type": "factual",
    },
    {
    "id": "fact_3",
    "en": "What is the largest planet in our solar system?",
    "fr": "Quelle est la plus grande planète de notre système solaire ?",
    "type": "factual",
    },
    # Open-ended / intent-like
    {
    "id": "open_1",
    "en": "Describe a simple way to make a morning routine more enjoyable.",
    "fr": "Décrivez une façon simple de rendre une routine matinale plus agréable.",
    "type": "open",
    },
    {
        "id": "open_2",
        "en": "What is one small habit that can help someone stay organized?",
        "fr": "Quelle petite habitude peut aider quelqu’un à rester organisé ?",
        "type": "open",
    },
    {
        "id": "open_3",
        "en": "If you could suggest a relaxing activity for a quiet afternoon, what would it be?",
        "fr": "Si vous pouviez suggérer une activité relaxante pour un après-midi tranquille, quelle serait-elle ?",
        "type": "open",
    },
]
