# Multi-Agent AI Systems: LangChain 1.0, LangGraph 1.0 та CrewAI

Цей репозиторій містить практичні приклади мультиагентних систем, побудованих на найсучасніших фреймворках для AI-агентів.

## Структура репозиторію

```
module5/
└── agents_v1/          # LangChain 1.0 + LangGraph 1.0
    ├── 01_basic_agent.py
    ├── 02_agent_with_middleware.py
    ├── 03_rag_agent_langgraph.py
    ├── 04_multiagent_langgraph.py
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

## agents_v1: LangChain 1.0 & LangGraph 1.0

Приклади, що демонструють найновіші можливості LangChain 1.0 та LangGraph 1.0:

### Ключові фічі
- **create_agent API** - новий спрощений API для створення агентів
- **StateGraph** - потужна оркестрація через графи станів
- **MemorySaver checkpointing** - персистентність між викликами
- **Middleware Architecture** - pre/post-processing hooks
- **Agentic RAG** - динамічний пошук з оцінкою релевантності
- **Supervisor Pattern** - ієрархічна координація агентів

### Агенти
1. **Basic Agent** - базовий агент з create_agent API
2. **Middleware Agent** - агент з logging, security, token limit middleware
3. **RAG Agent** - Agentic RAG з LangGraph StateGraph
4. **Multi-Agent System** - Supervisor Pattern з 4 спеціалізованими агентами

### Технології
- LangChain >= 1.0.0
- LangGraph >= 1.0.0
- OpenAI GPT-4o-mini
- FAISS vector store
- LangSmith tracing
- **РЕАЛЬНІ API:** OpenWeatherMap, Tavily Search, yfinance

[Детальна документація в agents_v1/README.md](agents_v1/README.md)

---



## Швидкий старт

### Налаштування середовища

1. Клонуйте репозиторій:
```bash
git clone https://github.com/agentspro/module5.git
cd module5
```

2. Оберіть фреймворк (agents_v1):
```bash
cd agents_v1  # або cd agents_v2
```

3. Встановіть залежності:
```bash
pip install -r requirements.txt
```

4. Налаштуйте змінні середовища:
```bash
cp .env.example .env
# Відредагуйте .env та додайте ваші API ключі
```

### Конфігурація API ключів

Створіть файл `.env` у відповідній папці:

```bash
# OpenAI API (Required)
OPENAI_API_KEY=sk-your-openai-api-key-here

# LangSmith (Optional, для трейсингу)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-langsmith-api-key-here
LANGCHAIN_PROJECT=your-project-name

# Real API Integrations (Required для agent tools)
# OpenWeatherMap - безкоштовна реєстрація
OPENWEATHERMAP_API_KEY=your-key-here  # https://openweathermap.org/api

# Tavily Search - безкоштовна реєстрація
TAVILY_API_KEY=tvly-your-key-here  # https://tavily.com
```

### Запуск прикладів

**agents_v1 (LangChain/LangGraph):**
```bash
cd agents_v1
python 01_basic_agent.py
python 02_agent_with_middleware.py
python 03_rag_agent_langgraph.py
python 04_multiagent_langgraph.py
```

---

## Порівняння фреймворків

### LangChain/LangGraph (agents_v1)

**Переваги:**
- Повний контроль над потоком виконання
- Гнучкість в архітектурі
- Потужна система middleware
- Детальний контроль над state management
- Відмінна інтеграція з LangSmith

**Використовуйте коли:**
- Потрібен детальний контроль над логікою
- Складні графи станів з умовними переходами
- Критична важливість observability
- Кастомні pattern'и оркестрації

---

## Вимоги

- Python >= 3.10, < 3.14
- OpenAI API ключ
- (Опціонально) LangSmith API ключ для трейсингу

---

## Ресурси

### LangChain/LangGraph
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)
---

## Ліцензія

MIT License

---

## Автор

sanyaden <alex.denysyuk@gmail.com>

