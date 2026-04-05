# AutoSubtitles (single-file app)

Минимальный веб-сервис для автосубтитров в одном `app.py`:

- загрузка видео,
- транскрибация через Groq Whisper,
- редактирование и предпросмотр субтитров,
- экспорт с вшитыми субтитрами через ffmpeg.

## 1) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Также нужен `ffmpeg` и `ffprobe` в PATH.

## 2) Настройка окружения

```bash
cp .env.example .env
```

Заполните `GROQ_API_KEY` в `.env`.

## 3) Запуск одной командой

```bash
python app.py
```

После запуска откройте `http://localhost:8000`.


## Структура

- `app.py` — весь backend (API + обработка ffmpeg/Groq + хранение файлов).
- `frontend/templates/index.html` — HTML.
- `frontend/static/style.css` — стили.
- `frontend/static/app.js` — frontend-логика.

## API

- `POST /api/upload` — принимает видео (form-data `file`), сохраняет временно, валидирует размер/длительность.
- `POST /api/transcribe` — `{ file_id, language? }`, извлекает аудио через ffmpeg, отправляет в Groq Whisper, возвращает `words[]` и `segments[]`.
- `POST /api/render` — `{ file_id, cues[], style{} }`, вшивает субтитры в mp4.
- `GET /api/download/<id>` — отдаёт готовый mp4.

## Ограничения без регистрации

Через `.env`:

- `MAX_FILE_SIZE_MB` — лимит размера файла,
- `MAX_VIDEO_DURATION_SEC` — лимит длительности видео,
- `FILE_TTL_SEC` — время жизни временных файлов,
- `CLEANUP_INTERVAL_SEC` — интервал автоочистки.
