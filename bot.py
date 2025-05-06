import discord
import openai
from openai import ChatCompletion
import torch
import asyncio
import random
from discord import utils
import logging
import time
import html
import re
import itertools
import datetime
from collections import deque
from functools import wraps
import aiohttp
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

#########################################
# Настройка логирования
#########################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация клиента OpenAI (если используется OpenAI режим)
if config.GENERATOR_MODE == "openai":
    openai.api_key = config.OPENAI_API_KEY
else:
    # Загрузка модели и токенизатора (только если используется локальная модель)
    MODEL_NAME = config.LOCAL_MODEL_PATH
    logger.info(f"Загрузка токенизатора {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Выбор устройства для модели: MODEL_DEVICE задаётся в конфиге ("cuda" или "cpu")
    device = config.MODEL_DEVICE if hasattr(config, "MODEL_DEVICE") else "cpu"
    if device.lower() == "cuda":
        device_map = "auto"
        dtype = torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32

    logger.info(f"Загрузка модели {MODEL_NAME} на устройстве {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        use_safetensors=True
    )
    logger.info("Модель загружена.")


#########################################
# Функции загрузки токенов и прокси
#########################################
def load_tokens_from_file(filepath="discord_tokens.txt"):
    """Загружает токены ботов из файла."""
    try:
        with open(filepath, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        if not tokens:
            logger.warning("Файл discord_tokens.txt пуст или не содержит валидных токенов.")
            return None
        logger.info(f"Загружено {len(tokens)} токенов из {filepath}")
        return tokens
    except FileNotFoundError:
        logger.error(f"Файл {filepath} не найден. Убедитесь, что файл discord_tokens.txt существует в директории с ботом.")
        return None

def load_proxies_from_file(filepath="proxies.txt"):
    """Загружает прокси из файла."""
    try:
        with open(filepath, 'r') as f:
            proxies = [line.strip() for line in f if line.strip()]
        if proxies:
            logger.info(f"Загружено {len(proxies)} прокси из {filepath}")
            return proxies
        else:
            logger.info(f"Файл {filepath} пуст, прокси не будут использоваться.")
            return None
    except FileNotFoundError:
        logger.warning(f"Файл {filepath} не найден. Прокси не будут использоваться.")
        return None

#########################################
# Список user agents для рандомизации
#########################################
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.126 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/116.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/115.0.1901.183 Safari/537.36"
]

old_get_info = utils._get_info

async def new_get_info(*args, **kwargs):
    sp, encoded = await old_get_info(*args, **kwargs)
    sp['browser_user_agent'] = random.choice(USER_AGENTS)
    return sp, encoded

utils._get_info = new_get_info


# Загрузка токенов и прокси при запуске бота
TOKEN_LIST = load_tokens_from_file()
PROXY_LIST = load_proxies_from_file()

if TOKEN_LIST is None:
    exit(1)  # Если токены не загружены, бот не может работать

TOKEN_CYCLE = itertools.cycle(TOKEN_LIST)
PROXY_CYCLE = itertools.cycle(PROXY_LIST) if PROXY_LIST else itertools.cycle([None])

# Глобальные настройки
MAX_HISTORY_SIZE = 0  # Количество пар «Пользователь–Ответ» в истории
MAX_REPLY_AGE = datetime.timedelta(minutes=1)  # Максимальный возраст входящего сообщения (1 минута)

#########################################
# Telegram-уведомления
#########################################
async def send_telegram_notification(text: str):
    if config.TELEGRAM_LOGGING_ENABLED:
        try:
            proxy = next(PROXY_CYCLE)
            proxy_config = {'proxy': proxy} if proxy else {}
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {
                    'chat_id': config.TELEGRAM_CHAT_ID,
                    'text': text,
                    'parse_mode': 'HTML'
                }
                async with session.post(url, data=payload, **proxy_config) as response:
                    if response.status != 200:
                        logger.error(f"Ошибка отправки уведомления в Telegram: {response.status}, {await response.text()}")
        except Exception:
            logger.exception("Ошибка при отправке Telegram-уведомления:")

#########################################
# Вспомогательная функция для получения настроек канала
#########################################
def get_channel_config(guild_id, channel_id):
    """
    Получает конфигурацию канала для заданного guild_id и channel_id.
    Функция пытается найти настройки, используя ключ channel_id как число и как строку.
    """
    guild_config = config.GUILDS.get(guild_id)
    if guild_config is None:
        return None, None
    channel_config = guild_config.get(channel_id) or guild_config.get(str(channel_id))
    if channel_config is None:
        return None, None
    mode = channel_config.get("mode")
    if mode == "reply":
        mode_settings = config.REPLY_SETTINGS
    elif mode == "active":
        mode_settings = config.ACTIVE_SETTINGS
    elif mode == "both":
        mode_settings = config.BOTH_SETTINGS
    else:
        mode_settings = {}
    return channel_config, mode_settings

#########################################
# Вспомогательная функция для проверки наличия роли
#########################################
def has_any_role(member, roles_list):
    return any(role.id in roles_list for role in member.roles)

#########################################
# Функция нормализации текста
#########################################
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

#########################################
# Класс AsyncQueue
#########################################
class AsyncQueue:
    def __init__(self, process_batch, batch_size=4, delay=0.1):
        self.queue = deque()
        self.batch_size = batch_size
        self.delay = delay
        self.process_batch = process_batch
        self.is_processing = False

    async def add(self, item):
        self.queue.append(item)
        if not self.is_processing:
            asyncio.create_task(self.process())

    async def process(self):
        self.is_processing = True
        while self.queue:
            batch = []
            while len(batch) < self.batch_size and self.queue:
                batch.append(self.queue.popleft())
            if batch:
                await self.process_batch(batch)
                await asyncio.sleep(self.delay)
        self.is_processing = False

#########################################
# Декоратор rate_limited
#########################################
def rate_limited(max_calls, period):
    def decorator(func):
        calls = []
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal calls
            now = time.time()
            calls = [t for t in calls if t > now - period]
            if len(calls) >= max_calls:
                await asyncio.sleep(period - (now - calls[0]))
            calls.append(time.time())
            return await func(*args, **kwargs)
        return wrapper
    return decorator

#########################################
# Класс DiscordBot
#########################################
class DiscordBot(discord.Client):
    def __init__(self, token, config, proxy=None, *args, **kwargs):
        if hasattr(discord, 'Intents'):
            intents = kwargs.pop("intents", discord.Intents.all())
            super().__init__(self_bot=True, proxy=proxy, intents=intents, *args, **kwargs)
        else:
            super().__init__(self_bot=True, proxy=proxy, *args, **kwargs)
        self.token = token
        self.config = config
        self.proxy = proxy  # Сохраняем прокси для логов
        self.token_index = 0  # Будет установлен извне
        self.bot_id = None    # Будет установлен после on_ready
        self.dialog_context = {}
        self.active_guild_id = None
        self.active_channel_id = None
        # Ключи для задач теперь включают bot_id для изоляции
        self.active_message_tasks = {}  # Ключ: (bot_id, guild_id, channel_id)
        self.priority_queues = {}       # Ключ: (bot_id, guild_id, channel_id)
        self.priority_tasks = {}        # Ключ: (bot_id, guild_id, channel_id)
        self.request_queue = AsyncQueue(self.process_and_send_response, batch_size=self.config.BATCH_SIZE)
        # Обновляем user agent через super_properties
        if hasattr(self, "http") and hasattr(self.http, "super_properties"):
            try:
                self.http.super_properties["browser_user_agent"] = random.choice(USER_AGENTS)
            except Exception as e:
                logger.exception("Ошибка при установке user agent: %s", e)

    async def run_bot(self, token, startup_delay):
        logger.info(f"DEBUG: run_bot: Функция run_bot вызвана для бота #{self.token_index + 1}, токен ...{self.token[-4:]}, задержка: {startup_delay} секунд")
        logger.info(f"🔑 Бот #{self.token_index + 1} ждет {startup_delay} секунд перед входом...")
        await asyncio.sleep(startup_delay)
        logger.info(f"🔑 Бот #{self.token_index + 1} пытается войти в Discord ...")
        try:
            await self.start(self.token)
        except discord.LoginFailure:
            logger.error(f"❌ Ошибка авторизации с токеном index {self.token_index}. Проверьте токен в discord_tokens.txt.")
        except aiohttp.ClientProxyConnectionError as e:
            logger.error(f"❌ Ошибка подключения через прокси {self.proxy}: {e}")
            logger.error("Возможно, прокси не работает или заблокирован. Проверьте прокси или запустите без прокси.")
        except Exception as e:
            logger.exception(f"🚨 Непредвиденная ошибка при запуске клиента с токеном index {self.token_index}:")

    @rate_limited(*config.RATE_LIMIT)
    async def generate_stream(self, prompt, settings, use_openai=False, history=None):
        if use_openai:
            try:
                proxy = next(PROXY_CYCLE)
                proxy_config = {'proxy': proxy} if proxy else {}

                # Формируем сообщение
                messages = [{"role": "system", "content": settings.get("system_prompt", "")}]
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": prompt})

                # Отправка запроса через прокси
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f'Bearer {config.OPENAI_API_KEY}'
                    }
                    url = "https://api.openai.com/v1/chat/completions"
                    payload = {
                        'model': config.OPENAI_ENGINE,
                        'messages': messages,
                        **config.OPENAI_MODEL_SETTINGS
                    }
                    async with session.post(url, json=payload, headers=headers, **proxy_config) as response:
                        if response.status != 200:
                            logger.error(f"Ошибка OpenAI API: {response.status}, {await response.text()}")
                            return None

                        response_json = await response.json()
                        return response_json['choices'][0]['message']['content']
            except Exception as e:
                logger.error(f"Ошибка при обращении к OpenAI API: {e}")
                return None
        else:
            # Локальный режим
            inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
            with torch.no_grad():
                generation_params = {**settings, "pad_token_id": tokenizer.eos_token_id}
                outputs = model.generate(**inputs, **generation_params)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def improved_add_typo(self, text, num_typos=4):
        if len(text) < 2:
            return text
        new_text = text
        for _ in range(num_typos):
            if len(new_text) < 2:
                break
            typo_type = random.choice(['swap', 'miss', 'repeat', 'keyboard'])
            pos = random.randint(0, len(new_text) - 2)
            if typo_type == 'swap' and len(new_text) > pos + 1:
                new_text = new_text[:pos] + new_text[pos+1] + new_text[pos] + new_text[pos+2:]
            elif typo_type == 'miss':
                new_text = new_text[:pos] + new_text[pos+1:]
            elif typo_type == 'repeat':
                new_text = new_text[:pos] + new_text[pos] + new_text[pos] + new_text[pos+1:]
            elif typo_type == 'keyboard':
                keyboard_errors = {
                    'q': 'w', 'w': 'e', 'e': 'r', 'r': 't', 't': 'y',
                    'a': 's', 's': 'd', 'd': 'f', 'f': 'g', 'g': 'h',
                    'z': 'x', 'x': 'c', 'c': 'v', 'v': 'b', 'b': 'n'
                }
                char = new_text[pos].lower()
                new_text = new_text[:pos] + keyboard_errors.get(char, char) + new_text[pos+1:]
        return new_text

    async def send_with_retry(self, channel, content, max_retries=3, reference=None):
        for attempt in range(max_retries):
            try:
                return await channel.send(content, reference=reference)
            except discord.HTTPException as e:
                if e.status == 429:
                    retry_after = e.retry_after or 5
                    await asyncio.sleep(retry_after)
                else:
                    raise
        raise Exception("Не удалось отправить сообщение после попыток")

    def check_message_allowed(self, guild_id, channel_id, message: str) -> bool:
        logger.debug(f"check_message_allowed: Вызвана для канала {channel_id}, сообщение: '{message}'")

        if guild_id in self.config.GUILDS:
            guild_config = self.config.GUILDS[guild_id]
            # Пытаемся найти channel_id как число И как строку в конфигурации guild_config
            channel_config = guild_config.get(channel_id) or guild_config.get(str(channel_id))

            if channel_config: # Проверяем, что channel_config был найден
                normalized_msg = normalize_text(message)
                negative_phrases = channel_config.get("negative_prompt", [])
                logger.debug(f"check_message_allowed: Негативные промпты для канала {channel_id}: {negative_phrases}")
                for phrase in negative_phrases:
                    normalized_phrase = normalize_text(phrase)
                    if re.search(rf'\b{re.escape(normalized_phrase)}\b', normalized_msg):
                        logger.info(f"Сообщение отклонено, содержит запрещённое слово: '{phrase}'")
                        return False
            else:
                logger.debug(f"check_message_allowed: Конфигурация канала {channel_id} не найдена.") # Лог, если config не найден

        logger.debug(f"check_message_allowed: Сообщение разрешено.")
        return True

    def clean_generated_text(self, text):
        unwanted_patterns = [
            r"</?think>",
            r"</?silence>",
        ]
        for pattern in unwanted_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def generate_response(self, guild_id, channel_id, user_message=None, response_type="reply"):
        try:
            channel_config, _ = get_channel_config(guild_id, channel_id)
            if not channel_config:
                return None

            if response_type == "reply":
                system_prompt = channel_config.get("prompt_reply", "")
            elif response_type == "active":
                system_prompt = channel_config.get("prompt_active", "")
            else:
                system_prompt = ""

            history_openai = []
            history_str = ""
            if user_message:
                context_key = (guild_id, channel_id, user_message.author.id)
                if context_key in self.dialog_context:
                    history_lines = self.dialog_context.get(context_key, [])[-MAX_HISTORY_SIZE:]
                    history_str = "\n".join(history_lines)
                    for line in history_lines:
                        if line.startswith("Пользователь:"):
                            history_openai.append({"role": "user", "content": line[len("Пользователь:"):].strip()})
                        elif line.startswith("Ответ:"):
                            history_openai.append({"role": "assistant", "content": line[len("Ответ:"):].strip()})
                full_prompt = f"{system_prompt}\n{history_str}Пользователь: {user_message.content}\nОтвет:"
            else:
                full_prompt = f"{system_prompt}\nОтвет:"

            logger.info(f"Промпт для генерации: {full_prompt}")

            if config.GENERATOR_MODE == "openai":
                response_text = await self.generate_stream(
                    prompt=user_message.content if user_message else full_prompt,
                    settings={"system_prompt": system_prompt},
                    use_openai=True,
                    history=history_openai
                )
            else:
                response_text = await self.generate_stream(full_prompt, config.MODEL_SETTINGS)

            if "Ответ:" in response_text:
                clean_response = response_text.split("Ответ:")[-1].strip()
            else:
                if response_text.startswith(system_prompt):
                    clean_response = response_text[len(system_prompt):].strip()
                else:
                    clean_response = response_text.strip()

            clean_response = self.clean_generated_text(clean_response)
            return clean_response
        except Exception:
            logger.exception("Ошибка при генерации ответа:")
            return None

    def update_dialog_context(self, guild_id, channel_id, user_id, user_message, bot_response):
        context_key = (guild_id, channel_id, user_id)
        if context_key not in self.dialog_context:
            self.dialog_context[context_key] = []
        self.dialog_context[context_key].extend([
            f"Пользователь: {user_message}",
            f"Ответ: {bot_response}"
        ])
        max_length = MAX_HISTORY_SIZE * 2
        if len(self.dialog_context[context_key]) > max_length:
            self.dialog_context[context_key] = self.dialog_context[context_key][-max_length:]
        if max_length == 0:
            self.dialog_context[context_key].clear()

    async def server_switcher(self):
        active_servers_list = list(self.config.GUILDS.keys())
        if not active_servers_list:
            logger.warning("Нет серверов для переключения (список серверов пуст). Server Switcher остановлен.")
            return

        num_cycles = self.config.CHANNEL_SWITCH_CYCLES
        if num_cycles <= 0:
            logger.warning("CHANNEL_SWITCH_CYCLES должен быть больше 0. Server Switcher остановлен.")
            return

        logger.info(f"Server Switcher запущен на {num_cycles} циклов.")
        cycle_count = 0
        while cycle_count < num_cycles:
            await asyncio.sleep(2)
            cycle_count += 1
            logger.info(f"🔄 Цикл смены серверов №{cycle_count}/{num_cycles} начат.")
            for guild_id in active_servers_list:
                try:
                    guild_id_int = int(guild_id)
                except:
                    continue
                if guild_id not in self.config.GUILDS:
                    logger.warning(f"Сервер {guild_id} не найден в конфигурации GUILDS, пропускаем.")
                    continue
                channels = self.config.GUILDS[guild_id]
                if not channels:
                    logger.warning(f"Нет каналов для сервера {guild_id}, пропускаем сервер.")
                    continue
                active_channel_time_min, active_channel_time_max = self.config.CHANNEL_ACTIVE_TIME
                active_time = random.uniform(active_channel_time_min, active_channel_time_max) * 60
                for channel_id_str, channel_config in channels.items():
                    try:
                        channel_id = int(channel_id_str)
                    except:
                        continue
                    mode = channel_config.get("mode")
                    if mode in ["active", "both", "reply"]:
                        task_key = (self.bot_id, guild_id, channel_id)
                        if task_key in self.active_message_tasks:
                            logger.info(f"⏹️ Завершение предыдущей active_message_task для канала {channel_id} перед переключением.")
                            self.active_message_tasks[task_key].cancel()
                            self.active_message_tasks.pop(task_key, None)
                        logger.info(f"⚙️ Запуск active_message_task для канала {channel_id} в Server Switcher.")
                        _, mode_settings = get_channel_config(guild_id, channel_id)
                        self.active_message_tasks[task_key] = asyncio.create_task(
                            self.active_message_task(guild_id, channel_id, channel_config, mode_settings)
                        )
                    logger.info(f"Сервер {guild_id} канал {channel_id} активен в течение {active_time / 60:.2f} минут.")
                    self.active_guild_id = guild_id
                    self.active_channel_id = channel_id
                    logger.info(f"✅ Активный сервер: {self.active_guild_id}, активный канал: {self.active_channel_id}")
                    await asyncio.sleep(active_time)
            logger.info(f"🔄 Цикл смены серверов №{cycle_count}/{num_cycles} завершен.")
        logger.info(f"Server Switcher завершил {num_cycles} циклов и останавливается.")

    async def process_and_send_response(self, batch, skip_priority_delay=False):
        logger.info(">> process_and_send_response: Начало обработки пакета сообщений.")
        logger.debug(f"  - Пакет сообщений для обработки: {batch}")
        for item in batch:
            message = item['message']
            guild_id = item['guild_id']
            channel_id = item['channel_id']
            logger.debug(f"  - Обработка сообщения от {message.author} в канале {channel_id}.")
            now = datetime.datetime.now(datetime.timezone.utc)
            message_age = now - message.created_at
            is_priority_reply = False
            if message.reference and message.reference.resolved:
                original_message = message.reference.resolved
                if original_message.author == self.user:
                    channel_config, mode_settings = get_channel_config(guild_id, channel_id)
                    mode = channel_config.get("mode") if channel_config else None
                    if (mode == "reply" and mode_settings.get("priority_replies", False)) or \
                       (mode == "both" and mode_settings.get("reply_priority", False)):
                        is_priority_reply = True
                        logger.debug("- Сообщение является приоритетным ответом.")
            if message_age > MAX_REPLY_AGE and not is_priority_reply:
                logger.info(f"⏳ Пропускаем устаревшее сообщение от {message.author} (возраст: {message_age}).")
                continue
            if message.guild and message.guild.id != self.active_guild_id:
                logger.debug(f"  - Сообщение с другого сервера (guild_id: {message.guild.id}, active_guild_id: {self.active_guild_id}), пропуск.")
                continue
            channel_config, mode_settings = get_channel_config(guild_id, channel_id)
            if not channel_config or not mode_settings:
                logger.debug(f"  - Конфигурация канала или mode_settings отсутствуют для {channel_id}, пропуск.")
                continue
            mode = channel_config.get("mode")
            if mode == "active":
                logger.debug("  - Канал в режиме 'active', process_and_send_response ничего не делает.")
                continue
            response_type = "reply"
            min_delay = mode_settings.get("min_delay", self.config.BOTH_SETTINGS["min_delay"])
            max_delay = mode_settings.get("max_delay", self.config.BOTH_SETTINGS["max_delay"])
            delay = random.uniform(min_delay, max_delay) * 60
            if is_priority_reply:
                if not skip_priority_delay:
                    task_key = (self.bot_id, guild_id, channel_id)
                    active_task = self.active_message_tasks.get(task_key)
                    if active_task:
                        logger.info(f"❌ Отменяем active_message_task в {channel_id} из-за приоритетного ответа.")
                        active_task.cancel()
                        self.active_message_tasks.pop(task_key, None)
                    else:
                        logger.info(f"  active_message_task для {channel_id} не найдена при приоритетном ответе.")
                    reply_priority_delay = mode_settings.get("reply_priority_delay", self.config.BOTH_SETTINGS["reply_priority_delay"]) * 60
                    logger.info(f"🕒 Ждём {reply_priority_delay / 60:.2f} минут перед отправкой приоритетного ответа.")
                    await asyncio.sleep(reply_priority_delay)
                else:
                    logger.info("skip_priority_delay=True, пропускаем ожидание задержки и отмену active_message_task для приоритетного ответа.")
            else:
                logger.info(f"🕒 Обычный reply, ждём задержку {delay/60:.2f} минут перед ответом.")
                await asyncio.sleep(delay)
            response = await self.generate_response(guild_id, channel_id, user_message=message, response_type=response_type)
            if response:
                if random.random() < self.config.TYPO_SETTINGS.get("typo_chance", 0):
                    typo_response = self.improved_add_typo(response)
                    sent_message = await self.send_with_retry(message.channel, typo_response, reference=message)
                    await asyncio.sleep(self.config.TYPO_SETTINGS.get("correction_delay", 0))
                    await sent_message.edit(content=response)
                else:
                    await self.send_with_retry(message.channel, response, reference=message)
                self.update_dialog_context(guild_id, channel_id, message.author.id, message.content, response)
                if self.config.TELEGRAM_NOTIFICATIONS.get("important_replies", False) and \
                   has_any_role(message.author, self.config.TELEGRAM_NOTIFICATIONS.get("important_roles", [])):
                    # Формируем уведомление, экранируя все динамические данные
                    notify_text = (
                        f"<b>Важный ответ!</b>\n"
                        f"Сервер: {html.escape(message.guild.name)}\n"
                        f"Канал: {html.escape(message.channel.name)}\n"
                        f"Пользователь: {html.escape(str(message.author))}\n"
                        f"<b>Сообщение:</b> {html.escape(message.content)}\n"
                        f"<b>Ответ:</b> {html.escape(response)}"
                    )
                    asyncio.create_task(send_telegram_notification(notify_text))
        logger.info("<< process_and_send_response: Завершение обработки пакета сообщений.")

    async def active_message_task(self, guild_id, channel_id, channel_config, mode_settings):
        task_key = (self.bot_id, guild_id, channel_id)
        mode = channel_config.get("mode")
        if mode not in ["active", "both", "reply"]:
            logger.warning(f"active_message_task запущена для канала {channel_id} в режиме {mode}, который не поддерживается.")
            return
        min_delay = mode_settings.get("min_delay", self.config.BOTH_SETTINGS["min_delay"])
        max_delay = mode_settings.get("max_delay", self.config.BOTH_SETTINGS["max_delay"])
        while True:
            delay = random.uniform(min_delay, max_delay) * 60
            logger.info(f"🚀 Запуск active_message_task для канала {channel_id}.")
            logger.info(f"Сообщение будет отправлено через {delay/60:.2f} минут.")
            try:
                await asyncio.sleep(delay)
                logger.debug(f"⏱️ Задержка {delay/60:.2f} минут истекла для канала {channel_id}.")
                if mode == "both":
                    action_type = random.choice(["active_message", "reply"])
                    logger.debug(f"active_message_task: выбран action_type = {action_type} для канала {channel_id} (режим both).")
                elif mode == "reply":
                    action_type = "reply"
                    logger.debug(f"active_message_task: режим reply, action_type = reply для канала {channel_id}.")
                else:
                    action_type = "active_message"
                    logger.debug(f"active_message_task: режим active, action_type = active_message для канала {channel_id}.")

                async def send_response(target_channel, response, reference_msg=None):
                    typo_chance = self.config.TYPO_SETTINGS.get("typo_chance", 0)
                    correction_delay = self.config.TYPO_SETTINGS.get("correction_delay", 0)
                    max_typos = self.config.TYPO_SETTINGS.get("max_typos", 2)
                    if random.random() < typo_chance:
                        num_typos = random.randint(1, max_typos)
                        typo_response = self.improved_add_typo(response, num_typos)
                        sent_msg = await self.send_with_retry(target_channel, typo_response, reference=reference_msg)
                        logger.info(f"Сообщение отправлено с опечаткой (num_typos={num_typos}). Исправление через {correction_delay} сек.")
                        await asyncio.sleep(correction_delay)
                        await sent_msg.edit(content=response)
                    else:
                        await self.send_with_retry(target_channel, response, reference=reference_msg)

                if action_type == "active_message":
                    prompt_active = channel_config.get("prompt_active", "")
                    if prompt_active:
                        response = await self.generate_response(guild_id, channel_id, response_type="active")
                        if response:
                            channel = self.get_channel(channel_id)
                            if channel:
                                try:
                                    await send_response(channel, response)
                                    self.update_dialog_context(guild_id, channel_id, self.user.id, "", response)
                                    logger.info(f"Активное сообщение отправлено в канал {channel_id}: {response}")
                                except Exception:
                                    logger.exception("🚨 Ошибка при отправке активного сообщения:")
                            else:
                                logger.warning(f"⚠️ Канал {channel_id} не найден, активное сообщение не отправлено.")
                        else:
                            logger.warning(f"⚠️ prompt_active не настроен для канала {channel_id}, активное сообщение не отправлено.")
                    else:
                        logger.debug(f"⚠️ Отсутствует prompt_active для канала {channel_id}.")
                elif action_type == "reply":
                    logger.debug(f"active_message_task: попытка отправить ответ в канале {channel_id} из набора сообщений.")
                    valid_messages = []
                    try:
                        channel = self.get_channel(channel_id)
                        if channel is None:
                            logger.warning(f"⚠️ Канал {channel_id} не найден при попытке получить историю сообщений.")
                        else:
                            async for msg in channel.history(limit=20):
                                if msg.author == self.user:
                                    continue
                                if (datetime.datetime.now(datetime.timezone.utc) - msg.created_at) > MAX_REPLY_AGE:
                                    continue
                                if self.config.IGNORE_MESSAGE_ROLES_ENABLED and has_any_role(msg.author, self.config.IGNORE_MESSAGE_ROLES):
                                    logger.debug(f"active_message_task: сообщение от {msg.author} игнорируется по IGNORE_MESSAGE_ROLES.")
                                    continue
                                if msg.reference and msg.reference.resolved and msg.reference.resolved.author == self.user:
                                    if self.config.IGNORE_REPLY_ROLES_ENABLED and has_any_role(msg.author, self.config.IGNORE_REPLY_ROLES):
                                        logger.debug(f"active_message_task: сообщение от {msg.author} игнорируется по IGNORE_REPLY_ROLES.")
                                        continue
                                if not self.check_message_allowed(guild_id, channel_id, msg.content):
                                    logger.debug("active_message_task: сообщение отклонено check_message_allowed.")
                                    continue
                                valid_messages.append(msg)
                    except Exception as e:
                        logger.exception(f"🚨 Ошибка при получении истории сообщений канала {channel_id}: {e}")

                    if valid_messages:
                        chosen_msg = random.choice(valid_messages)
                        logger.debug(f"active_message_task: выбрано сообщение от {chosen_msg.author} для ответа.")
                        response = await self.generate_response(guild_id, channel_id, chosen_msg, response_type="reply")
                        if response:
                            channel = self.get_channel(channel_id)
                            if channel:
                                try:
                                    await send_response(chosen_msg.channel, response, reference_msg=chosen_msg)
                                    self.update_dialog_context(guild_id, channel_id, chosen_msg.author.id, chosen_msg.content, response)
                                    logger.info(f"Ответ отправлен в канал {channel_id} на сообщение пользователя {chosen_msg.author}: {response}")
                                except Exception:
                                    logger.exception("🚨 Ошибка при отправке ответа:")
                            else:
                                logger.warning(f"⚠️ Канал {channel_id} не найден, ответ не отправлен.")
                        else:
                            logger.warning("⚠️ Не удалось сгенерировать ответ, отправка ответа отменена.")
                    else:
                        if mode == "both":
                            logger.debug(f"active_message_task: подходящих сообщений для ответа не найдено в канале {channel_id}, отправляем активное сообщение (fallback).")
                            prompt_active = channel_config.get("prompt_active", "")
                            if prompt_active:
                                response = await self.generate_response(guild_id, channel_id, response_type="active")
                                if response:
                                    channel = self.get_channel(channel_id)
                                    if channel:
                                        try:
                                            await send_response(channel, response)
                                            self.update_dialog_context(guild_id, channel_id, self.user.id, "", response)
                                            logger.info(f"Активное сообщение (fallback) отправлено в канал {channel_id}: {response}")
                                        except Exception:
                                            logger.exception("🚨 Ошибка при отправке fallback активного сообщения:")
                                    else:
                                        logger.warning(f"⚠️ Канал {channel_id} не найден, fallback активное сообщение не отправлено.")
                                else:
                                    logger.warning(f"⚠️ prompt_active не настроен для канала {channel_id}, fallback активное сообщение не отправлено.")
                        else:
                            logger.debug(f"active_message_task: в режиме reply подходящих сообщений для ответа не найдено, отправка пропущена.")
                logger.info(f"⏹️ Завершение цикла active_message_task для канала {channel_id}.")
            except asyncio.CancelledError:
                logger.info(f"⏹️ Завершение active_message_task для канала {channel_id} (отмена).")
                break
            except Exception:
                logger.exception(f"🚨 Ошибка в active_message_task для канала {channel_id}:")
                await asyncio.sleep(60)
        logger.info(f"⚫ active_message_task завершена для канала {channel_id}.")

    async def on_message(self, message):
        if message.guild and message.guild.id != self.active_guild_id:
            return
        if message.author == self.user:
            return
        guild_id = message.guild.id
        channel_id = message.channel.id
        logger.debug(f">> on_message: Получено сообщение от {message.author}#{message.author.discriminator} в канале {channel_id}: {message.content}")
        channel_config, mode_settings = get_channel_config(guild_id, channel_id)
        logger.debug(f" - on_message: get_channel_config вернула: channel_config={channel_config}, mode_settings={mode_settings}")
        if not channel_config:
            return
        mode = channel_config.get("mode")
        if mode == "active":
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        message_age = now - message.created_at
        is_priority_reply = False
        if message.reference and message.reference.resolved:
            original_message = message.reference.resolved
            if original_message.author == self.user:
                if (mode == "reply" and mode_settings.get("priority_replies", False)) or \
                   (mode == "both" and mode_settings.get("reply_priority", False)):
                    is_priority_reply = True
        if message_age > MAX_REPLY_AGE and not is_priority_reply:
            logger.info(f"Сообщение от {message.author} слишком старое ({message_age}), пропускаем.")
            return
        if not self.check_message_allowed(guild_id, channel_id, message.content):
            return
        if message.reference and message.reference.resolved:
            original_message = message.reference.resolved
            if original_message.author == self.user:
                if self.config.IGNORE_REPLY_ROLES_ENABLED and has_any_role(message.author, self.config.IGNORE_REPLY_ROLES):
                    if self.config.TELEGRAM_NOTIFICATIONS.get("important_replies", False) and \
                       has_any_role(message.author, self.config.TELEGRAM_NOTIFICATIONS.get("important_roles", [])):
                        notify_text = (
                            f"<b>Важный ответ!</b>\n"
                            f"Сервер: {html.escape(message.guild.name)}\n"
                            f"Канал: {html.escape(message.channel.name)}\n"
                            f"Пользователь: {html.escape(str(message.author))}\n"
                            f"Сообщение: {html.escape(message.content)}"
                        )
                        asyncio.create_task(send_telegram_notification(notify_text))
                    return
        if self.config.IGNORE_MESSAGE_ROLES_ENABLED and has_any_role(message.author, self.config.IGNORE_MESSAGE_ROLES):
            return
        key = (self.bot_id, guild_id, channel_id)
        if is_priority_reply:
            priority_ignore_chance = mode_settings.get("ignore_chance", 0)
            if random.random() < priority_ignore_chance:
                logger.info(f"Приоритетное сообщение от {message.author} проигнорировано (ignore_chance сработал).")
                return
            if key not in self.priority_queues:
                self.priority_queues[key] = asyncio.Queue()
            await self.priority_queues[key].put({
                'message': message,
                'guild_id': guild_id,
                'channel_id': channel_id
            })
            logger.info(f"Приоритетное сообщение от {message.author} добавлено в очередь для канала {channel_id}.")
            if key not in self.priority_tasks or self.priority_tasks[key].done():
                if key in self.active_message_tasks:
                    self.active_message_tasks[key].cancel()
                    self.active_message_tasks.pop(key, None)
                    logger.info(f"active_message_task для канала {channel_id} отменён для приоритетной обработки.")
                self.priority_tasks[key] = asyncio.create_task(
                    self.process_priority_queue(guild_id, channel_id, channel_config, mode_settings)
                )
            return
        ignore_chance = mode_settings.get("ignore_chance", 0)
        if mode in ["reply", "both"] and random.random() < ignore_chance:
            return
        #await self.request_queue.add({
        #    'message': message,
        #    'guild_id': guild_id,
        #    'channel_id': channel_id
        #})

    async def process_priority_queue(self, guild_id, channel_id, channel_config, mode_settings):
        key = (self.bot_id, guild_id, channel_id)
        queue = self.priority_queues.get(key)
        if not queue:
            return
        reply_priority_delay = mode_settings.get("reply_priority_delay", self.config.BOTH_SETTINGS.get("reply_priority_delay", 1)) * 60
        while not queue.empty():
            item = await queue.get()
            logger.info(f"Ждем {reply_priority_delay/60:.2f} минут перед ответом на приоритетное сообщение в канале {channel_id}.")
            await asyncio.sleep(reply_priority_delay)
            await self.process_and_send_response([item], skip_priority_delay=True)
            logger.info(f"Обработано приоритетное сообщение для канала {channel_id}.")
        self.priority_tasks.pop(key, None)
        logger.info(f"Очередь приоритетных сообщений для канала {channel_id} пуста.")
        if key not in self.active_message_tasks:
            self.active_message_tasks[key] = asyncio.create_task(
                self.active_message_task(guild_id, channel_id, channel_config, mode_settings)
            )
            logger.info(f"active_message_task для канала {channel_id} перезапущен после обработки очереди приоритетных сообщений.")

#########################################
# Функция запуска бота с токеном
#########################################
async def start_bot_with_token(token_index, token, startup_delay):
    logger.info(f"DEBUG: start_bot_with_token: Запуск для бота #{token_index + 1}, токен ...{token[-4:]}, задержка: {startup_delay} секунд")
    current_proxy = next(PROXY_CYCLE)
    proxy_config_for_client = {'proxy': current_proxy} if current_proxy else {}
    client = DiscordBot(token, config=config, **proxy_config_for_client)
    client.token_index = token_index
    client.dialog_context = {}
    client.config = config
    client.request_queue = AsyncQueue(client.process_and_send_response, batch_size=config.BATCH_SIZE)
    client.active_message_tasks = {}
    client.priority_queues = {}
    client.priority_tasks = {}
    client.active_guild_id = None
    client.active_channel_id = None

    @client.event
    async def on_ready():
        client.bot_id = client.user.id  # Сохраняем уникальный идентификатор бота
        logger.info(f"✅ Аккаунт #{client.token_index + 1} (токен index {client.token_index}) запущен как {client.user.name}#{client.user.discriminator} {'с прокси' if client.proxy else 'без прокси'}")
        asyncio.create_task(client.server_switcher())
        asyncio.create_task(client.request_queue.process())
        logger.info(f"Бот #{client.token_index + 1} готов к работе.")

    await client.run_bot(token, startup_delay)

#########################################
# Главный цикл запуска ботов
#########################################
async def main_bot_loop():
    if not TOKEN_LIST:
        logger.error("Нет токенов для запуска бота. Выход.")
        return
    startup_delay_base = 10  # базовая задержка в секундах
    tasks = []
    for i, token in enumerate(TOKEN_LIST):
        startup_delay = startup_delay_base * i
        logger.info(f"Бот #{i + 1} будет запущен с задержкой {startup_delay} секунд.")
        tasks.append(asyncio.create_task(start_bot_with_token(i, token, startup_delay)))
    await asyncio.gather(*tasks)
    logger.info("✅ Задачи запуска всех ботов инициированы. Боты работают в фоне.")

asyncio.run(main_bot_loop())
