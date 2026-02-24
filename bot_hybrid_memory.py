"""
Telegram-бот с гибридной памятью (короткая + долгая)
- Короткая память: последние N сообщений диалога (RAM)
- Долгая память: документы → эмбеддинги → ChromaDB (persistent)
Использует aiogram 3.x, OpenAI API и ChromaDB
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Deque
from collections import defaultdict, deque
import io
import time

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message, Document
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение токенов из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в .env файле!")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в .env файле!")

# Инициализация OpenAI клиента
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ==================== НАСТРОЙКИ КОРОТКОЙ ПАМЯТИ ====================
HISTORY_SIZE = 10  # Количество запоминаемых сообщений диалога
user_histories: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

# ==================== НАСТРОЙКИ ДОЛГОЙ ПАМЯТИ ====================
CHUNK_SIZE = 500  # Размер чанка в символах
CHUNK_OVERLAP = 50  # Перекрытие между чанками

# Инициализация ChromaDB (векторная база данных)
MEMORY_DIR = "./memory"
Path(MEMORY_DIR).mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=MEMORY_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Создание/получение коллекции для хранения эмбеддингов
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"description": "User uploaded documents with embeddings"}
)

# Создание роутера для обработчиков
router = Router()


# ==================== ФУНКЦИИ РАБОТЫ С ДОКУМЕНТАМИ (ДОЛГАЯ ПАМЯТЬ) ====================

def load_document(file_content: bytes, filename: str) -> str:
    """
    Загружает и конвертирует документ в текст
    Поддерживает: TXT, PDF, DOCX
    """
    file_ext = Path(filename).suffix.lower()
    
    try:
        if file_ext == '.txt':
            return file_content.decode('utf-8')
        
        elif file_ext == '.pdf':
            import PyPDF2
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif file_ext in ['.docx', '.doc']:
            import docx
            doc_file = io.BytesIO(file_content)
            doc = docx.Document(doc_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа: {e}")
        raise


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Разбивает текст на части (chunks) с перекрытием
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start = end - overlap
    
    logger.info(f"Текст разбит на {len(chunks)} частей")
    return chunks


async def embed_chunks(user_id: int, chunks: List[str], filename: str):
    """
    Создает эмбеддинги для чанков и сохраняет в ChromaDB
    """
    logger.info(f"Создание эмбеддингов для {len(chunks)} частей...")
    
    try:
        embeddings_response = await openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=chunks
        )
        
        embeddings = [item.embedding for item in embeddings_response.data]
        
        # Уникальные ID с временной меткой
        timestamp = int(time.time() * 1000)
        ids = [f"user_{user_id}_file_{filename}_ts_{timestamp}_chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [
            {
                "user_id": str(user_id),
                "filename": filename,
                "chunk_index": i,
                "timestamp": timestamp
            }
            for i in range(len(chunks))
        ]
        
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Успешно сохранено {len(chunks)} эмбеддингов в базу данных")
        
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддингов: {e}")
        raise


async def retrieve_context(user_id: int, query: str, n_results: int = 3) -> List[Dict]:
    """
    Ищет релевантные фрагменты документа по запросу
    """
    logger.info(f"Поиск контекста в документах для запроса: {query}")
    
    try:
        query_embedding_response = await openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=[query]
        )
        query_embedding = query_embedding_response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": str(user_id)}
        )
        
        if results['documents'] and results['documents'][0]:
            context_items = []
            for i, doc in enumerate(results['documents'][0]):
                context_items.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
            
            logger.info(f"Найдено {len(context_items)} релевантных фрагментов в документах")
            return context_items
        else:
            logger.info("Релевантные фрагменты в документах не найдены")
            return []
    
    except Exception as e:
        logger.error(f"Ошибка при поиске контекста: {e}")
        return []


# ==================== ГИБРИДНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА ====================

async def generate_response(user_id: int, user_message: str) -> str:
    """
    Генерирует ответ используя ОБЕ памяти:
    - Короткую память (история диалога)
    - Долгую память (контекст из документов)
    """
    
    # 1. Проверяем, есть ли документы у пользователя (долгая память)
    try:
        user_docs = collection.get(where={"user_id": str(user_id)})
        has_documents = bool(user_docs['ids'])
    except:
        has_documents = False
    
    # 2. Ищем релевантный контекст в документах (если есть)
    document_context = []
    if has_documents:
        document_context = await retrieve_context(user_id, user_message, n_results=3)
    
    # 3. Формируем системный промпт
    system_prompt = """Ты дружелюбный AI-ассистент с гибридной памятью.

У тебя есть:
1. КОРОТКАЯ ПАМЯТЬ - история текущего диалога
2. ДОЛГАЯ ПАМЯТЬ - загруженные документы пользователя

Правила:
- Используй информацию из ОБЕИХ памятей для ответа
- Если есть информация в документах - ссылайся на неё
- Если информации нет в документах, но есть в истории диалога - используй её
- Отвечай четко и по существу
- Не придумывай информацию, которой нет ни в документах, ни в истории"""

    # 4. Формируем сообщения для модели
    messages = [{"role": "system", "content": system_prompt}]
    
    # Добавляем контекст из документов (если есть)
    if document_context:
        doc_text = "\n\n---\n\n".join([item["text"] for item in document_context])
        context_message = f"""📄 КОНТЕКСТ ИЗ ДОКУМЕНТОВ:

{doc_text}

---
Используй этот контекст для ответа, если он релевантен вопросу."""
        messages.append({"role": "system", "content": context_message})
    
    # Добавляем историю диалога (короткая память)
    messages.extend(list(user_histories[user_id]))
    
    # Добавляем текущее сообщение пользователя
    messages.append({"role": "user", "content": user_message})
    
    # 5. Генерируем ответ
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=1000
        )
        
        answer = response.choices[0].message.content
        logger.info("Ответ успешно сгенерирован (использована гибридная память)")
        return answer
        
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        raise


# ==================== ОБРАБОТЧИКИ КОМАНД БОТА ====================

@router.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start
    """
    user_id = message.from_user.id
    user_histories[user_id].clear()
    
    await message.answer(
        "🤖 <b>Привет! Я бот с гибридной памятью.</b>\n\n"
        "🧠 <b>Мои возможности:</b>\n\n"
        "💭 <b>КОРОТКАЯ ПАМЯТЬ</b>\n"
        "• Помню последние 10 сообщений диалога\n"
        "• Поддерживаю контекст разговора\n\n"
        "📚 <b>ДОЛГАЯ ПАМЯТЬ</b>\n"
        "• Загружаю и анализирую документы (PDF, TXT, DOCX)\n"
        "• Отвечаю на вопросы по документам\n"
        "• Сохраняю информацию между сеансами\n\n"
        "🎯 <b>Как использовать:</b>\n"
        "1. Просто общайся со мной - я запомню контекст\n"
        "2. Загрузи документ - я буду использовать его в ответах\n"
        "3. Задавай вопросы - я использую ОБЕ памяти!\n\n"
        "📋 <b>Команды:</b>\n"
        "/clear - Очистить историю диалога\n"
        "/clear_docs - Удалить все документы\n"
        "/info - Информация о памяти",
        parse_mode="HTML"
    )
    logger.info(f"Пользователь {user_id} запустил бота")


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    """
    Очистка КОРОТКОЙ памяти (истории диалога)
    """
    user_id = message.from_user.id
    message_count = len(user_histories[user_id])
    user_histories[user_id].clear()
    
    await message.answer(
        f"💭 <b>Короткая память очищена!</b>\n\n"
        f"Удалено {message_count} сообщений из истории диалога.\n"
        f"Документы сохранены. Для удаления документов используй /clear_docs",
        parse_mode="HTML"
    )
    logger.info(f"Пользователь {user_id} очистил короткую память")


@router.message(Command("clear_docs"))
async def cmd_clear_docs(message: Message):
    """
    Очистка ДОЛГОЙ памяти (документов)
    """
    user_id = message.from_user.id
    
    try:
        results = collection.get(where={"user_id": str(user_id)})
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            await message.answer(
                f"📚 <b>Долгая память очищена!</b>\n\n"
                f"Удалено {len(results['ids'])} фрагментов из документов.\n"
                f"История диалога сохранена.",
                parse_mode="HTML"
            )
            logger.info(f"Пользователь {user_id} очистил долгую память ({len(results['ids'])} фрагментов)")
        else:
            await message.answer("📭 Долгая память пуста - нет загруженных документов.")
    
    except Exception as e:
        logger.error(f"Ошибка при очистке документов: {e}")
        await message.answer("❌ Ошибка при очистке документов")


@router.message(Command("info"))
async def cmd_info(message: Message):
    """
    Информация о состоянии ОБЕИХ памятей
    """
    user_id = message.from_user.id
    
    # Информация о короткой памяти
    short_memory_count = len(user_histories[user_id])
    
    # Информация о долгой памяти
    try:
        results = collection.get(where={"user_id": str(user_id)})
        
        if results['ids']:
            from datetime import datetime
            files_info = {}
            
            for metadata in results['metadatas']:
                if 'filename' in metadata:
                    filename = metadata['filename']
                    timestamp = metadata.get('timestamp', 0)
                    
                    if filename not in files_info or timestamp > files_info[filename]['timestamp']:
                        files_info[filename] = {'timestamp': timestamp, 'count': 0}
            
            for metadata in results['metadatas']:
                filename = metadata.get('filename')
                timestamp = metadata.get('timestamp', 0)
                if filename in files_info and timestamp == files_info[filename]['timestamp']:
                    files_info[filename]['count'] += 1
            
            info_text = f"🧠 <b>Состояние гибридной памяти:</b>\n\n"
            info_text += f"💭 <b>КОРОТКАЯ ПАМЯТЬ</b>\n"
            info_text += f"📝 Сообщений в истории: {short_memory_count}/{HISTORY_SIZE}\n\n"
            
            info_text += f"📚 <b>ДОЛГАЯ ПАМЯТЬ</b>\n"
            info_text += f"📄 Уникальных файлов: {len(files_info)}\n"
            info_text += f"🧩 Всего фрагментов: {len(results['ids'])}\n\n"
            
            if files_info:
                info_text += "<b>Файлы:</b>\n"
                for filename, info in sorted(files_info.items()):
                    if info['timestamp']:
                        upload_time = datetime.fromtimestamp(info['timestamp'] / 1000)
                        time_str = upload_time.strftime("%d.%m.%Y %H:%M")
                        info_text += f"• {filename}\n  📊 {info['count']} фрагментов | ⏰ {time_str}\n"
            
            await message.answer(info_text, parse_mode="HTML")
        else:
            await message.answer(
                f"🧠 <b>Состояние гибридной памяти:</b>\n\n"
                f"💭 <b>КОРОТКАЯ ПАМЯТЬ</b>\n"
                f"📝 Сообщений в истории: {short_memory_count}/{HISTORY_SIZE}\n\n"
                f"📚 <b>ДОЛГАЯ ПАМЯТЬ</b>\n"
                f"📭 Документов пока нет\n\n"
                f"Отправьте файл для создания долгой памяти!",
                parse_mode="HTML"
            )
    
    except Exception as e:
        logger.error(f"Ошибка при получении информации: {e}")
        await message.answer("❌ Ошибка при получении информации")


@router.message(F.document)
async def handle_document(message: Message, bot: Bot):
    """
    Обработчик загруженных документов (ДОЛГАЯ ПАМЯТЬ)
    """
    user_id = message.from_user.id
    document: Document = message.document
    
    filename = document.file_name
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in ['.txt', '.pdf', '.docx', '.doc']:
        await message.answer(
            "❌ Неподдерживаемый формат файла!\n\n"
            "Поддерживаются: TXT, PDF, DOCX"
        )
        return
    
    await message.answer("⏳ Обрабатываю документ, подождите...")
    logger.info(f"Получен документ от пользователя {user_id}: {filename}")
    
    try:
        file = await bot.get_file(document.file_id)
        file_content = await bot.download_file(file.file_path)
        file_bytes = file_content.read()
        
        text = load_document(file_bytes, filename)
        
        if not text.strip():
            await message.answer("❌ Документ пустой или не удалось извлечь текст")
            return
        
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            await message.answer("❌ Не удалось разбить документ на части")
            return
        
        await embed_chunks(user_id, chunks, filename)
        
        await message.answer(
            f"✅ <b>Документ добавлен в долгую память!</b>\n\n"
            f"📄 Файл: {filename}\n"
            f"📊 Размер: {len(text)} символов\n"
            f"🧩 Фрагментов: {len(chunks)}\n\n"
            f"💬 Теперь я буду использовать этот документ в ответах!",
            parse_mode="HTML"
        )
        
        logger.info(f"Документ {filename} успешно обработан для пользователя {user_id}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        await message.answer(
            f"❌ Ошибка при обработке документа:\n{str(e)}\n\n"
            "Попробуйте другой файл или формат."
        )


@router.message(F.text)
async def handle_message(message: Message):
    """
    Обработчик текстовых сообщений
    Использует ГИБРИДНУЮ ПАМЯТЬ (короткая + долгая)
    """
    user_id = message.from_user.id
    user_text = message.text
    
    logger.info(f"Получено сообщение от пользователя {user_id}: {user_text}")
    
    # Добавляем сообщение пользователя в короткую память
    user_histories[user_id].append({
        "role": "user",
        "content": user_text
    })
    
    # Отправляем индикатор "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Генерируем ответ используя гибридную память
        bot_reply = await generate_response(user_id, user_text)
        
        # Добавляем ответ бота в короткую память
        user_histories[user_id].append({
            "role": "assistant",
            "content": bot_reply
        })
        
        # Отправляем ответ пользователю
        await message.answer(bot_reply)
        
        logger.info(f"Отправлен ответ пользователю {user_id}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer(
            "❌ Произошла ошибка при обработке вашего сообщения.\n"
            "Попробуйте еще раз или используйте:\n"
            "/clear - очистить историю диалога\n"
            "/clear_docs - очистить документы"
        )


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

async def main():
    """
    Главная функция для запуска бота
    """
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    
    dp.include_router(router)
    
    logger.info("🚀 Бот с гибридной памятью запущен!")
    logger.info(f"📁 База данных документов: {MEMORY_DIR}")
    logger.info(f"💭 Размер короткой памяти: {HISTORY_SIZE} сообщений")
    logger.info(f"🤖 Модель чата: {OPENAI_MODEL}")
    logger.info(f"🔤 Модель эмбеддингов: {EMBED_MODEL}")
    
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
