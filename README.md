# montage

Фоновый обработчик Remember. Отдельный процесс на Python, наружу не смотрит: в цикле берёт задачи
из таблицы `jobs` в Postgres, выполняет и почти всё время спит. Единственный способ дать ему работу —
вставить строку в `jobs` (это делает API).

Сейчас он умеет семь вещей: делать превью загруженного медиа (`media_preview`), читать из оригинала
размер кадра, камеру и место съёмки (`media_frame_info`), считать по нему предложение
автоматического отсева (`media_screening`), разбирать, что на кадре и кто на нём
(`media_understanding`), расшифровывать речь в видео (`media_speech`), собирать ролик из материала
воспоминания (`montage`) и раз в час подчищать оборванные загрузки.

```sh
python -m montage run     # цикл разбора очереди; остальные команды — docs/running.md
```

## Где что лежит

```
montage/          код воркера, один пакет
  worker.py         цикл очереди: взять задачу, позвать обработчик, повторить упавшее и брошенное
  sweeps.py         ежечасная уборка: чего очередь не досчиталась
  storage.py        бакет, одна дверь: за ней S3 (MinIO) или Google по ключу служебной учётки
  ffmpeg.py         единственное место, где запускается ffmpeg
  journal.py        журнал монтажа — тонкий писатель в таблицу
  photo.py          HEIC и EXIF, отдельным процессом
  ai/               облачная модель: дверь, провайдеры за ней, разбор ответа
  media/            задачи об одном загруженном файле
  video/            задача об одном заказанном ролике и весь пайплайн сборки
tests/            тесты и их фикстуры
docs/             как это поднять, что обещает каждая задача очереди и как всё это проверяется
```

## Что читать

- [docs/running.md](docs/running.md) — что нужно, как поднять и как запустить.
- [docs/queue.md](docs/queue.md) — как устроена очередь, что бывает с брошенной задачей и при
  остановке, что делает уборка.
- [docs/testing.md](docs/testing.md) — как гонять тесты, в том числе в контейнере.
- [docs/ai.md](docs/ai.md) — облачная модель и журнал запросов к ней.
- [docs/director.md](docs/director.md) — режиссёр-нейросеть.

Контракты задач очереди — по файлу на задачу:

- [docs/jobs/preview.md](docs/jobs/preview.md) — `media_preview`
- [docs/jobs/frame-info.md](docs/jobs/frame-info.md) — `media_frame_info`
- [docs/jobs/screening.md](docs/jobs/screening.md) — `media_screening`
- [docs/jobs/understanding.md](docs/jobs/understanding.md) — `media_understanding`
- [docs/jobs/speech.md](docs/jobs/speech.md) — `media_speech`
- [docs/jobs/montage.md](docs/jobs/montage.md) — `montage`
- [docs/jobs/montage-journal.md](docs/jobs/montage-journal.md) — журнал сборки ролика
