"""
Telegram-бот с короткой памятью (history buffer)
Использует aiogram 3.x и OpenAI ChatCompletion API
"""

import os
import asyncio
import logging
from collections import defaultdict, deque
from typing import Dict, Deque

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message
from openai import AsyncOpenAI
from dotenv import load_dotenv  # Импорт библиотеки
load_dotenv()  # Загрузка .env файла

BOT_TOKEN = os.getenv("BOT_TOKEN")  # Теперь будет работать
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение токенов из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в переменных окружения!")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения!")

# Инициализация OpenAI клиента
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Хранилище истории сообщений: user_id -> deque последних N сообщений
# Используем deque для автоматического удаления старых сообщений
HISTORY_SIZE = 10
user_histories: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

# Создание роутера для обработчиков
router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start
    Приветствует пользователя и очищает историю
    """
    user_id = message.from_user.id
    
    # Очищаем историю пользователя при старте
    user_histories[user_id].clear()
    
    await message.answer(
        "👋 Привет! Я бот с короткой памятью.\n\n"
        "Я запоминаю последние 10 сообщений нашего диалога.\n"
        "Просто напиши мне что-нибудь, и я отвечу! 💬"
    )
    logger.info(f"Пользователь {user_id} запустил бота")


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    """
    Обработчик команды /clear
    Очищает историю диалога пользователя
    """
    user_id = message.from_user.id
    user_histories[user_id].clear()
    
    await message.answer("🗑️ История диалога очищена!")
    logger.info(f"Пользователь {user_id} очистил историю")


@router.message(F.text)
async def handle_message(message: Message):
    """
    Обработчик текстовых сообщений
    Сохраняет сообщение в историю, отправляет в OpenAI и возвращает ответ
    """
    user_id = message.from_user.id
    user_text = message.text
    
    logger.info(f"Получено сообщение от пользователя {user_id}: {user_text}")
    
    # Добавляем сообщение пользователя в историю
    user_histories[user_id].append({
        "role": "user",
        "content": user_text
    })
    
    # Отправляем индикатор "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Подготавливаем историю для отправки в OpenAI
        # Добавляем системное сообщение в начало
        messages = [
            {
                "role": "system",
                "content": "Ты дружелюбный AI-ассистент. Отвечай кратко и по существу."
            }
        ]
        
        # Добавляем историю диалога пользователя
        messages.extend(list(user_histories[user_id]))
        
        logger.info(f"Отправка запроса в OpenAI (история: {len(user_histories[user_id])} сообщений)")
        
        # Отправляем запрос в OpenAI
        response = await openai_client.chat.completions.create(
           # model="gpt-3.5-turbo",  # Можно заменить на "gpt-4"
            model = os.getenv("OPENAI_MODEL"),
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        # Получаем ответ от модели
        bot_reply = response.choices[0].message.content
        
        logger.info(f"Получен ответ от OpenAI: {bot_reply[:50]}...")
        
        # Добавляем ответ бота в историю
        user_histories[user_id].append({
            "role": "assistant",
            "content": bot_reply
        })
        
        # Отправляем ответ пользователю
        await message.answer(bot_reply)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer(
            "❌ Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /clear для очистки истории."
        )


async def main():
    """
    Главная функция для запуска бота
    """
    # Инициализация бота и диспетчера
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    
    # Регистрация роутера
    dp.include_router(router)
    
    logger.info("Бот запущен!")
    
    try:
        # Удаляем старые обновления и запускаем polling
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
