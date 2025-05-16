# Document QA API 🚀

FastAPI-сервис для обработки документов и ответов на вопросы по их содержимому. Поддерживает загрузку DOCX-файлов и интеллектуальный поиск информации.

## 📋 Предварительные требования

- Docker и Docker Compose ([установка](https://docs.docker.com/get-docker/))
- Git ([установка](https://git-scm.com/downloads))
- Запросить API Keys для модели qwen/qwen3-235b-a22b:free на сайте OpenRouter ([установка](https://openrouter.ai/settings/keys)) и вписать в .\config\settings.py API Keys

## 🛠 Установка и запуск

### Вариант 1: Запуск через Docker (рекомендуется)

```bash
# 1. Клонировать репозиторий
git clone [https://github.com/Ziraelsik/RAG_API]
RAG_API

# 2. Собрать и запустить контейнеры
docker-compose up -d --build

# 3. Сервис будет доступен по адресу:
#    http://localhost:8000



