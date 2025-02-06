from transformers import pipeline
import deepl

from bot.config import DEEPL_API_KEY

mood_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

def analyze_mood(text: str) -> str:
    translated_text = translate_text(text)

    result = mood_analyzer(translated_text)
    sentiment = result[0]['label']

    if sentiment == "joy":
        return "Щасливий"
    elif sentiment == "anger":
        return "Злий"
    elif sentiment == "sadness":
        return "Сумний"
    elif sentiment == "fear":
        return "Наляканий"
    elif sentiment == "surprise":
        return "Здивований"
    elif sentiment == "disgust":
        return "Обурений"
    else:
        return "Нейтральний"

def translate_text(text: str) -> str:
    translator = deepl.Translator(DEEPL_API_KEY)

    result = translator.translate_text(text, target_lang="EN-US")
    return result.text

