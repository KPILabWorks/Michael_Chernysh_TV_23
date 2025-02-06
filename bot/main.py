from telegram.ext import Application, CommandHandler, MessageHandler, filters
from handlers import start, stop, handle_text  # додали handle_text для обробки текстових повідомлень
from config import TG_API_KEY


def main():
    application = Application.builder().token(TG_API_KEY).build()

    # Обробник для стартової команди
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    # Обробник для текстових повідомлень (після /start)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling()


if __name__ == '__main__':
    main()
