# Библиотека музыки

Десять треков экрана «Под что монтировать» (`design/screens/MontageMusic.html`): пять, что показывает
макет, — в его порядке, следом ещё пять того же рода. Имя файла — это id трека, и по нему
`library.track_file` находит файл. Заказ ролика называет музыку
списком `params.music_tracks` в порядке проигрывания: элемент — id отсюда или ключ `music/…` трека,
который организатор загрузил сам. Пожелание к треку, если его написали, лежит в
`params.music_wishes` под тем же id — `{"calm": "торт — под припев"}`; поле необязательное.
Переименовать файл — значит переименовать трек, а его id знают ещё api, который проверяет заказ, и
строки уже заказанных роликов (весь вход — `docs/jobs/montage.md`, «Музыка заказа»).

| id | на экране | что за трек | длина | темп | файл |
|---|---|---|---|---|---|
| `bright` | Яркая | Happy Alley | 1:20 | 112 | `bright.mp3` |
| `warm` | Тёплая | Easy Lemon | 2:06 | 82 | `warm.mp3` |
| `calm` | Спокойная | Thinking of You | 1:50 | 72 | `calm.mp3` |
| `night` | Ночь | Night Cave | 1:50 | 104 | `night.mp3` |
| `road` | Дорога | Neo Western | 2:26 | 84 | `road.mp3` |
| `merry` | Весёлая | Merry Go | 2:00 | 132 | `merry.mp3` |
| `tender` | Нежная | Heartwarming | 1:12 | 82 | `tender.mp3` |
| `dreamy` | Мечтательная | Dreamy Flashback | 2:07 | 110 | `dreamy.mp3` |
| `mystery` | Таинственная | Frost Waltz | 2:15 | 99 | `mystery.mp3` |
| `energetic` | Энергичная | Cipher | 3:51 | 150 | `energetic.mp3` |

Трек короче ролика играет с начала ещё раз, длиннее — обрывается на конце картинки; поэтому длина
трека ни на что не влияет, а темп влияет: под него ложатся склейки (`rhythm.py`).

## Откуда и по какому праву

Все десять написал **Kevin MacLeod**, все десять взяты с его сайта **incompetech.com** — скачиваются
прямой ссылкой, без регистрации и без оплаты. Лицензия у всех одна:
**Creative Commons Attribution 4.0** (<https://creativecommons.org/licenses/by/4.0/>).

CC BY 4.0 разрешает любое использование, включая коммерческое, и переделку — **при условии
указания автора**. Условие относится и к роликам, которые соберёт наш воркер: там, где ролик
показывают и откуда его отдают, автор музыки должен быть назван. Строки для этого — ниже, их и
нужно поставить в приложении (экран ролика, «о треке» или титры — как решит продукт). Лицензия без
указания авторства у incompetech платная; пока она не куплена, строки обязательны.

| файл | строка для указания авторства |
|---|---|
| `bright.mp3` | «Happy Alley» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `warm.mp3` | «Easy Lemon» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `calm.mp3` | «Thinking of You» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `night.mp3` | «Night Cave» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `road.mp3` | «Neo Western» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `merry.mp3` | «Merry Go» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `tender.mp3` | «Heartwarming» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `dreamy.mp3` | «Dreamy Flashback» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `mystery.mp3` | «Frost Waltz» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |
| `energetic.mp3` | «Cipher» by Kevin MacLeod (incompetech.com) — CC BY 4.0 |

Полностью, как просит сам автор: *"<название>" Kevin MacLeod (incompetech.com), Licensed under
Creative Commons: By Attribution 4.0 License, http://creativecommons.org/licenses/by/4.0/*.

## Что именно лежит в репозитории

Файлы — байт в байт те, что отдаёт сайт: ничего не перекодировано и не обрезано, чтобы файл здесь и
файл по ссылке можно было сверить. Первые пять скачаны 2026-09-18, остальные — 2026-10-28; вместе —
51 МБ; это единственное бинарное, что есть в репозитории воркера.

Ссылка на файл — это `https://incompetech.com/music/royalty-free/mp3-royaltyfree/<название>.mp3`,
страница трека — `https://incompetech.com/music/royalty-free/index.html?isrc=<ISRC>`. Каталог автора
машиночитаемый (<https://incompetech.com/music/royalty-free/pieces.json>) — оттуда взяты длина,
темп и ISRC в таблицах выше.

| файл | название на сайте | ISRC |
|---|---|---|
| `bright.mp3` | Happy Alley | USUAN1100482 |
| `warm.mp3` | Easy Lemon | USUAN1200076 |
| `calm.mp3` | Thinking of You | USUAN1100637 |
| `night.mp3` | Night Cave | USUAN1100446 |
| `road.mp3` | Neo Western | USUAN1100615 |
| `merry.mp3` | Merry Go | USUAN1100731 |
| `tender.mp3` | Heartwarming | USUAN1100207 |
| `dreamy.mp3` | Dreamy Flashback | USUAN1100532 |
| `mystery.mp3` | Frost Waltz | USUAN1100516 |
| `energetic.mp3` | Cipher | USUAN1100844 |

Что файл тот самый — `shasum -a 256 tracks/*.mp3`:

```
f598c8ee624f1db9a2cd38886524a40baff4ce730dcf999c0763791268fa8265  tracks/bright.mp3
11229fe56a1cdb9e47696164f4386e5c09fb4a67e961a065e19ac6af830ba52e  tracks/calm.mp3
e63abdb9bcadadb48945760a06ccd56f7fe17207a940729cb9bc806bac636473  tracks/dreamy.mp3
d3d0614535a61d748e2fd8d5117fc4d781022a7624794802a2462d8af3fdd522  tracks/energetic.mp3
c592ca0a9e2312280ea54070831a216c58ab7242890c0742eb95be5365b6e307  tracks/merry.mp3
560f932189551c8a009a671da25d8b51cf2067df85fcac35731ea6ac0c7d6cd3  tracks/mystery.mp3
b8a80bbf0fc66db1758acdec416a219c10d7546ec4e2ac33c5f78c35f7a77534  tracks/night.mp3
27046fcf6c91a3f59dc8acaca40c7d3ef200cb1054f699aa27ebc9f4a80d9c28  tracks/road.mp3
a621a943ab940567f4136060e0f1f0637038b8a03c10ff8926f956e1b2ef34e7  tracks/tender.mp3
2486b8ecfe4575f9ab31613a2c03c2ee45e6e5280301d05b4dd7237c9e1c2531  tracks/warm.mp3
```

## Заменить трек

Положить рядом файл с тем же именем, поправить обе таблицы выше (и строку авторства, и откуда
файл) — больше ничего: код библиотеки знает только id, а темп ищется в самом файле на каждом
ролике. Добавить ещё трек — строка в `library.TRACKS`, файл сюда, и такой же список id у api,
иначе он не даст его заказать.
