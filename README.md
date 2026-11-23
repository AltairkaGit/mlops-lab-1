# Лабораторные работы по MLOps

## ЛР №1. Сервис классификации токсичности

Python 3.12, FastAPI, PyTorch, модель [cointegrated/rubert-tiny-toxicity](https://huggingface.co/cointegrated/rubert-tiny-toxicity). FastAPI загружает модель один раз при запуске. При первом старте Hugging Face скачивает веса в локальный кеш. Для класса `toxic` сервис использует формулу из карточки модели `1 − P(non-toxic) × (1 − P(dangerous))` и порог `0.5`. Поле `probability` возвращает оценку выбранного класса.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
python -m pytest tests/ -v
uvicorn app.main:app
```

Swagger UI доступен по адресу <http://127.0.0.1:8000/docs>.

| Метод и путь | Тело запроса | Результат |
| --- | --- | --- |
| `GET /health` | | `{"status":"ok"}` |
| `POST /predict` | `{"text":"Доброе утро!"}` | Класс и его вероятность |
| `POST /predict_batch` | `{"texts":["Доброе утро!","Ну ты и идиот!"]}` | Предсказания в исходном порядке |
| `POST /predict_batch` | `{"texts":[]}` | `{"predictions":[]}` |
| `GET /model_info` | | Название, ревизия модели, версия Transformers |

Пять тестов написаны до реализации. Первый запуск завершился ошибкой импорта, пока модуля приложения ещё не было. Тесты используют подставную модель с фиксированными оценками и проверяют HTTP-контракт без загрузки весов. Для проверки pre-commit добавлялось сравнение `value == None`. flake8 остановил коммит с `E711`; проверочный файл удалён.
