"""
Telegram-бот с долгой памятью (документы → эмбеддинги → ChromaDB)
Использует aiogram 3.x, OpenAI Embeddings и ChromaDB
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
import io

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

# Настройки для разбиения документов
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


# ==================== ФУНКЦИИ РАБОТЫ С ДОКУМЕНТАМИ ====================

def load_document(file_content: bytes, filename: str) -> str:
    """
    Загружает и конвертирует документ в текст
    Поддерживает: TXT, PDF, DOCX
    """
    file_ext = Path(filename).suffix.lower()
    
    try:
        if file_ext == '.txt':
            # Текстовый файл - просто декодируем
            return file_content.decode('utf-8')
        
        elif file_ext == '.pdf':
            # PDF файл - используем PyPDF2
            import PyPDF2
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif file_ext in ['.docx', '.doc']:
            # DOCX файл - используем python-docx
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
        
        # Пропускаем пустые чанки
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
        # Создаем эмбеддинги через OpenAI
        embeddings_response = await openai_client.embeddings.create(
            model=EMBED_MODEL,  # Модель для эмбеддингов из .env
            input=chunks
        )
        
        # Извлекаем векторы эмбеддингов
        embeddings = [item.embedding for item in embeddings_response.data]
        
        # Создаем уникальные ID для каждого чанка с временной меткой
        import time
        timestamp = int(time.time() * 1000)  # Миллисекунды для уникальности
        ids = [f"user_{user_id}_file_{filename}_ts_{timestamp}_chunk_{i}" for i in range(len(chunks))]
        
        # Метаданные для каждого чанка
        metadatas = [
            {
                "user_id": str(user_id),
                "filename": filename,
                "chunk_index": i,
                "timestamp": timestamp
            }
            for i in range(len(chunks))
        ]
        
        # Сохраняем в ChromaDB
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
    Использует семантический поиск через эмбеддинги
    """
    logger.info(f"Поиск контекста для запроса: {query}")
    
    try:
        # Создаем эмбеддинг для запроса
        query_embedding_response = await openai_client.embeddings.create(
            model=EMBED_MODEL,  # Модель для эмбеддингов из .env
            input=[query]
        )
        query_embedding = query_embedding_response.data[0].embedding
        
        # Ищем похожие фрагменты в ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": str(user_id)}  # Фильтруем по пользователю
        )
        
        # Формируем результат
        if results['documents'] and results['documents'][0]:
            context_items = []
            for i, doc in enumerate(results['documents'][0]):
                context_items.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
            
            logger.info(f"Найдено {len(context_items)} релевантных фрагментов")
            return context_items
        else:
            logger.info("Релевантные фрагменты не найдены")
            return []
    
    except Exception as e:
        logger.error(f"Ошибка при поиске контекста: {e}")
        return []


async def answer_question(query: str, context_items: List[Dict]) -> str:
    """
    Генерирует ответ на основе найденного контекста
    Использует ChatCompletion API
    """
    # Формируем контекст из найденных фрагментов
    context_text = "\n\n---\n\n".join([item["text"] for item in context_items])
    
    # Формируем промпт для модели
    system_prompt = """Ты AI-ассистент, который отвечает на вопросы на основе предоставленных документов.

ВАЖНО:
- Отвечай ТОЛЬКО на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи: "В загруженных документах нет информации для ответа на этот вопрос"
- НЕ придумывай информацию
- Отвечай четко и по существу
- Если нужно, цитируй документ"""

    user_prompt = f"""Контекст из документа:
{context_text}

Вопрос пользователя: {query}

Ответь на вопрос на основе контекста выше."""
    
    try:
        # Отправляем запрос к OpenAI
        # Используем max_completion_tokens для новых моделей (gpt-4, gpt-5)
        # Для gpt-5-mini не указываем temperature, т.к. поддерживается только значение по умолчанию (1)
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=1000  # Для новых моделей используем max_completion_tokens
        )
        
        answer = response.choices[0].message.content
        logger.info("Ответ успешно сгенерирован")
        return answer
    
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        raise


# ==================== ОБРАБОТЧИКИ КОМАНД БОТА ====================

@router.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start
    Приветствует пользователя и объясняет возможности
    """
    await message.answer(
        "🤖 <b>Привет! Я бот с долгой памятью.</b>\n\n"
        "📄 <b>Что я умею:</b>\n"
        "• Загружать документы (PDF, TXT, DOCX)\n"
        "• Запоминать содержимое документов\n"
        "• Отвечать на вопросы по загруженным документам\n\n"
        "📤 <b>Как использовать:</b>\n"
        "1. Отправь мне документ\n"
        "2. Задай вопрос по документу\n"
        "3. Получи точный ответ!\n\n"
        "🗑️ <b>Команды:</b>\n"
        "/clear - Очистить мою память\n"
        "/info - Информация о загруженных документах",
        parse_mode="HTML"
    )
    logger.info(f"Пользователь {message.from_user.id} запустил бота")


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    """
    Обработчик команды /clear
    Удаляет все документы пользователя из базы данных
    """
    user_id = message.from_user.id
    
    try:
        # Получаем все документы пользователя
        results = collection.get(where={"user_id": str(user_id)})
        
        if results['ids']:
            # Удаляем все документы пользователя по ID (более надежный способ)
            collection.delete(ids=results['ids'])
            await message.answer(f"🗑️ Удалено {len(results['ids'])} фрагментов из памяти!")
            logger.info(f"Пользователь {user_id} очистил память ({len(results['ids'])} фрагментов)")
        else:
            await message.answer("📭 Память уже пуста!")
    
    except Exception as e:
        logger.error(f"Ошибка при очистке памяти: {e}")
        await message.answer("❌ Ошибка при очистке памяти")


@router.message(Command("info"))
async def cmd_info(message: Message):
    """
    Обработчик команды /info
    Показывает информацию о загруженных документах
    """
    user_id = message.from_user.id
    
    try:
        # Получаем все документы пользователя
        results = collection.get(where={"user_id": str(user_id)})
        
        if results['ids']:
            # Группируем файлы по имени и находим последнюю версию каждого
            from datetime import datetime
            files_info = {}
            
            for metadata in results['metadatas']:
                if 'filename' in metadata:
                    filename = metadata['filename']
                    timestamp = metadata.get('timestamp', 0)
                    
                    if filename not in files_info or timestamp > files_info[filename]['timestamp']:
                        files_info[filename] = {
                            'timestamp': timestamp,
                            'count': 0
                        }
            
            # Подсчитываем фрагменты для каждого файла (только последней версии)
            for metadata in results['metadatas']:
                filename = metadata.get('filename')
                timestamp = metadata.get('timestamp', 0)
                if filename in files_info and timestamp == files_info[filename]['timestamp']:
                    files_info[filename]['count'] += 1
            
            info_text = f"📚 <b>Информация о памяти:</b>\n\n"
            info_text += f"📄 Уникальных файлов: {len(files_info)}\n"
            info_text += f"🧩 Всего фрагментов: {len(results['ids'])}\n\n"
            
            if files_info:
                info_text += "<b>Файлы:</b>\n"
                for filename, info in sorted(files_info.items()):
                    # Форматируем дату загрузки
                    if info['timestamp']:
                        upload_time = datetime.fromtimestamp(info['timestamp'] / 1000)
                        time_str = upload_time.strftime("%d.%m.%Y %H:%M")
                        info_text += f"• {filename}\n  📊 {info['count']} фрагментов | ⏰ {time_str}\n"
                    else:
                        info_text += f"• {filename} ({info['count']} фрагментов)\n"
            
            await message.answer(info_text, parse_mode="HTML")
        else:
            await message.answer("📭 В памяти пока нет документов.\n\nОтправьте мне файл!")
    
    except Exception as e:
        logger.error(f"Ошибка при получении информации: {e}")
        await message.answer("❌ Ошибка при получении информации")


@router.message(F.document)
async def handle_document(message: Message, bot: Bot):
    """
    Обработчик загруженных документов
    Сохраняет документ, создает эмбеддинги и сохраняет в ChromaDB
    """
    user_id = message.from_user.id
    document: Document = message.document
    
    # Проверяем расширение файла
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
        # Скачиваем файл
        file = await bot.get_file(document.file_id)
        file_content = await bot.download_file(file.file_path)
        file_bytes = file_content.read()
        
        # Загружаем и конвертируем документ в текст
        text = load_document(file_bytes, filename)
        
        if not text.strip():
            await message.answer("❌ Документ пустой или не удалось извлечь текст")
            return
        
        # Разбиваем на чанки
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            await message.answer("❌ Не удалось разбить документ на части")
            return
        
        # Создаем эмбеддинги и сохраняем в базу
        await embed_chunks(user_id, chunks, filename)
        
        await message.answer(
            f"✅ <b>Документ успешно обработан!</b>\n\n"
            f"📄 Файл: {filename}\n"
            f"📊 Размер: {len(text)} символов\n"
            f"🧩 Фрагментов: {len(chunks)}\n\n"
            f"💬 Теперь можете задавать вопросы по документу!",
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
async def handle_question(message: Message):
    """
    Обработчик текстовых сообщений (вопросов)
    Ищет контекст в базе и генерирует ответ
    """
    user_id = message.from_user.id
    question = message.text
    
    logger.info(f"Получен вопрос от пользователя {user_id}: {question}")
    
    # Проверяем, есть ли документы у пользователя
    try:
        user_docs = collection.get(where={"user_id": str(user_id)})
        
        if not user_docs['ids']:
            await message.answer(
                "📭 Сначала загрузите документ!\n\n"
                "Отправьте мне файл (PDF, TXT, DOCX), "
                "и я смогу отвечать на вопросы по нему."
            )
            return
    except Exception as e:
        logger.error(f"Ошибка при проверке документов: {e}")
    
    # Отправляем индикатор "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Ищем релевантный контекст
        context_items = await retrieve_context(user_id, question, n_results=3)
        
        if not context_items:
            await message.answer(
                "🤔 Не нашел релевантной информации в загруженных документах.\n\n"
                "Попробуйте переформулировать вопрос или загрузите нужный документ."
            )
            return
        
        # Генерируем ответ
        answer = await answer_question(question, context_items)
        
        # Отправляем ответ
        await message.answer(
            f"💡 <b>Ответ:</b>\n\n{answer}",
            parse_mode="HTML"
        )
        
        logger.info(f"Отправлен ответ пользователю {user_id}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        await message.answer(
            "❌ Произошла ошибка при обработке вашего вопроса.\n"
            "Попробуйте еще раз или используйте /clear для очистки памяти."
        )


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

async def main():
    """
    Главная функция для запуска бота
    """
    # Инициализация бота и диспетчера
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    
    # Регистрация роутера
    dp.include_router(router)
    
    logger.info("🚀 Бот с долгой памятью запущен!")
    logger.info(f"📁 База данных: {MEMORY_DIR}")
    logger.info(f"🤖 Модель чата: {OPENAI_MODEL}")
    logger.info(f"🔤 Модель эмбеддингов: {EMBED_MODEL}")
    
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
        logger.info("Бот остановлен пользователем")
