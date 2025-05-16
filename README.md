# Document QA API 🚀

FastAPI-сервис для обработки документов и ответов на вопросы по их содержимому. Поддерживает загрузку DOCX-файлов и интеллектуальный поиск информации.

## 📋 Предварительные требования

- Docker и Docker Compose ([установка](https://docs.docker.com/get-docker/))
- Git ([установка](https://git-scm.com/downloads))
- Для разработки: Python 3.9+

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
