##############################
# 1. ОБЩИЕ НАСТРОЙКИ
##############################

# Режим генерации текста:
#   "local"   – Локальная модель 
#   "openai"  – API OpenAI
GENERATOR_MODE = "openai"

# Глобальные лимиты и параметры запросов: Этот пункт можете не трогать
RATE_LIMIT = (5, 1)        # 5 запросов в секунду
BATCH_SIZE = 4             # Размер батча для обработки
RETRY_SETTINGS = {         # Повторы при ошибках
    'max_retries': 3,
    'delay': [1, 5, 10]  # Задержка (в секундах)
}

# --- Общие параметры активности ---
CHANNEL_ACTIVE_TIME = (5, 15)  # Бот активен в одном канале 5-15 минут
CHANNEL_SWITCH_CYCLES = 3      # Количество циклов смены каналов

##############################
# 2. НАСТРОЙКИ РЕЖИМОВ
##############################

# --- Режим "reply" (только отвечает) ---
REPLY_SETTINGS = {
    "min_delay": 0.30,        # Минимальная задержка перед ответом (минуты)
    "max_delay": 0.40,        # Максимальная задержка перед ответом (минуты)
    "priority_replies": True,  # True/False. Если True - Отвечает в первую очередь на реплаи записывая их в очередь
    "reply_priority_delay": 1,  # Задержка для приоритетного ответа на реплай (в минутах)
    "ignore_chance": 0.1  # Вероятность игнорирования (10%)
}

# --- Режим "active" (только пишет сам) ---
ACTIVE_SETTINGS = {
    "min_delay": 0.30,  # Минимальная задержка перед новым сообщением (минуты)
    "max_delay": 0.40  # Максимальная задержка перед новым сообщением (минуты)
}

# --- Режим "both" (и отвечает, и пишет сам) ---
BOTH_SETTINGS = {
    "min_delay": 2,        # Минимальная задержка для случайных действий (в минутах)
    "max_delay": 3,        # Максимальная задержка для случайных действий (в минутах)
    "reply_priority": True,   # True/False. Если True - Отвечает в первую очередь на реплаи записывая их в очередь
    "reply_priority_delay": 1,  # Задержка для приоритетного ответа на реплай (в минутах)
    "ignore_chance": 0.1  # Вероятность игнорирования (10%)
}

##############################
# 3. ТЕЛЕГРАМ-УВЕДОМЛЕНИЯ
##############################

TELEGRAM_LOGGING_ENABLED = False # True/False
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN" # Можете создать через @BotFather. Не забудьте в боте нажать /start
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID" # ID вашего аккаунта на который будут приходить уведомления. Можете получить через специальных ботов

TELEGRAM_NOTIFICATIONS = {
    "dm_messages": True,   #True/False Личные сообщения в Telegram
    "important_replies": True,  #True/False Ответы от важных пользователей в Telegram
    "important_roles": [111111111111111111, 222222222222222222]  # ID ролей от которых вы хотите получать уведомления в Telegram
}
IGNORE_REPLY_ROLES_ENABLED = True #True/False
IGNORE_REPLY_ROLES = [123456789012345678, 987654321098765432]  # Если пользователь с одной из этих ролей отвечает на наше сообщение, бот не отвечает

IGNORE_MESSAGE_ROLES_ENABLED = True #True/False
IGNORE_MESSAGE_ROLES = [111111111111111111, 222222222222222222]  # Если сообщение пришло от пользователя с одной из этих ролей, бот не отвечает

##############################
# 4. НАСТРОЙКИ СЕРВЕРОВ И КАНАЛОВ (DISCORD)
##############################

# mode:
#   "reply"  – бот только отвечает на сообщения
#   "active" – бот только пишет сам
#   "both"   – бот и отвечает, и пишет сам

GUILDS = {
    1340963835271712838: {  # Сервер 1
        1340963835271712841: {  # Канал 1
            "mode": "reply",
            "prompt_reply": "You talk like a real person—chill, sarcastic when needed, and never overly polished. Keep it short (1-2 sentences). Use slang, contractions, and casual phrasing. Sound like someone just texting back, not like you're writing an article. Avoid sounding too excited or overly formal.",
            "prompt_active": "You’re just casually dropping messages like a normal dude. No forced enthusiasm, no corporate-sounding text. If you talk about music, tech, or games, sound like you actually care but keep it real. Drop a take, ask a quick question, or just react. No more than a sentence or two.",
            "negative_prompt": ["you are a bot", "your are bot", "youre a bot", "bot", "ai", "robot", "искусственный интеллект", "робот", "бот", "машина", "virtual assistant"] # Своего рода банворды, бот их полностью игнорирует в любом режиме.
        },
    },
    987654321098765432: {  # Сервер 2
        112233445566778899: {  # Канал 1
            "mode": "active",
            "prompt_reply": "",
            "prompt_active": "Share interesting thoughts and discussions.",
            "negative_prompt": []
        }
    }
}

##############################
# 5. НАСТРОЙКИ ГЕНЕРАЦИИ И МОДЕЛЕЙ
##############################

# --- Локальная модель ---
LOCAL_MODEL_PATH = "/root/DiscordBOT/models/deepseek-1.5b" # Полный путь к вашей модели.(если вы решили ее использовать)
MODEL_DEVICE = "cpu" #или "cuda" если gpu
MODEL_SETTINGS = {
    "do_sample": True,            # Включает сэмплирование (случайный выбор токенов)
    "temperature": 0.5,           # Температура выбора слов (чем выше, тем разнообразнее текст)
    "top_p": 0.9,                 # Nucleus sampling: учитываются только топ-90% вероятных токенов
    "top_k": 50,                  # Учитывает только топ-50 самых вероятных токенов
    "repetition_penalty": 1.2,    # Штраф за повторения (чем выше, тем меньше повторов)
    "num_beams": 3,               # Количество лучей (beam search), большее значение = качественнее, но медленнее
    "early_stopping": True,       # Останавливает генерацию, когда модель считает, что текст закончен
    "max_length": 200             # Максимальная длина сгенерированного текста (в токенах)
}


# --- OpenAI API ---
OPENAI_API_KEY = "" # Ваш купленный API 
OPENAI_ENGINE = "gpt-4o-mini"  # Эта модель самая сбалансированная, но можете использовать другую 
OPENAI_MODEL_SETTINGS = {
    "max_tokens": 200,
    "temperature": 0.5,
    "top_p": 0.9,
    "n": 1,
    "stop": None
}


##############################
# 6. ОПЕЧАТКИ
##############################

TYPO_SETTINGS = {
    "typo_chance": 0.2,  # 20% шанс на опечатку
    "correction_delay": 5  # Исправление через 5 секунд
}
