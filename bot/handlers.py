from telegram import Update
from telegram.ext import CallbackContext

from bot.mood import analyze_mood
from bot.prompts import send_user_text


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        'Привіт! Я бот-психолог. Як я можу тобі допомогти? Опиши мені свій настрій, і ми поговоримо з тобою про це.'
    )


async def handle_text(update: Update, context: CallbackContext):
    user_text = update.message.text.lower()
    mood = analyze_mood(user_text)

    response = await send_user_text(user_text,mood)
    text = (f"Зараз твій настрій {mood}. "
            f"{response}")

    await update.message.reply_text(text)

async def stop(update: Update, context: CallbackContext) -> None:
    context.user_data.clear()
    await update.message.reply_text("Сесія завершена. Якщо потрібно, звертайтеся знову.")