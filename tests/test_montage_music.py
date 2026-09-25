"""Музыка ролика: что заказали, как это звучит и что бывает с плохим треком.

См. `docs/jobs/montage.md`, «Треки». Всё живое: настоящий Postgres, бакет и ffmpeg.
"""

import json
import uuid

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from montage import ffmpeg, storage, worker
from montage.video import assembly, library, music, rhythm, selection

from checks import about, loudness
from material import click_track, hush, jpeg, long_clip, mute_clip, probe, tone, unopenable, video
from rows import add_item, api_params, fetch, layout_of, order, read_montage, ready_material


def pulled_to_the_beat(layout, beats, own_lengths):
    """What the beats did to a finished montage, read off its layout — the lengths of the clips as
    they were encoded, so of the file itself. Every join is heard half a crossfade into the piece
    after it, and that moment has to be one of the track's beats: within a frame and a half, for the
    encoding, which rounds every piece to one — a join that missed its beat would miss it by
    hundreds of milliseconds. And the pieces have to be held visibly off the lengths they had
    before the music, or nothing was pulled anywhere."""
    for _, start, _ in layout[1:]:
        join = start + assembly.CROSSFADE_SECONDS / 2
        assert min(abs(join - beat) for beat in beats) < 0.1, f"the join at {join:.2f} is off beat"
    assert any(abs(seconds - own) > 0.2 for (_, _, seconds), own in zip(layout, own_lengths)), (
        "nothing was pulled anywhere — the pieces stayed as they were"
    )


def test_a_montage_is_cut_to_the_beat_of_the_track_the_organizer_uploaded(
    conn, s3, memory, tmp_path
):
    """The organizer's own track instead of the library's: the order names a key rather than an id
    (the api's `music_uploads`), and the joins fall into its rhythm just the same — while the video
    stays the nine seconds its pieces add up to, because the last piece takes up what the joins
    moved.
    """
    track_key = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(click_track(tmp_path / "click.wav")), storage.bucket(), track_key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_key": track_key})
    # The clip first (oldest): its file runs three seconds and the pieces after it are stills.
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)
    s3.upload_file(str(video(tmp_path / "clip.mp4")), storage.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "the beats changed how long the video is"
    beats = rhythm.beats(tmp_path / "click.wav")
    assert beats, "a metronome is a tempo, or there is nothing here to check"
    pulled_to_the_beat(layout_of(conn, montage_id), beats, (3.0, 2.5, 2.5, 2.5))


@pytest.mark.parametrize(
    "bad_track", ["bytes that are not media", "a photograph", "a clip shot with the microphone off"]
)
def test_an_uploaded_track_with_no_music_in_it_fails_the_montage_at_once_and_says_why(
    conn, s3, memory, tmp_path, monkeypatch, bad_track
):
    """An uploaded track is somebody else's bytes, and ffmpeg reads them: the api takes any file
    under a track's name. Not music at all, a picture, a clip with nothing to hear (or a playlist,
    which would send ffmpeg walking this worker's disks) — refused on the spot and with the reason,
    before a single piece is encoded, and the job ends clean: the second attempt would meet the
    same file, and three failed assemblies with «не удалось собрать ролик» tell nobody anything.
    The shelf track next to it in the list does not save the order."""
    track_key = f"music/{uuid.uuid4()}"
    made, name = {
        "bytes that are not media": (unopenable, "track"),
        "a photograph": (jpeg, "track.jpg"),
        "a clip shot with the microphone off": (mute_clip, "track.mp4"),
    }[bad_track]
    s3.upload_file(str(made(tmp_path / name)), storage.bucket(), track_key)
    montage_id = order(conn, memory, {"music_tracks": ["warm", track_key]})
    ready_material(conn, s3, memory, tmp_path)
    job_id = worker.enqueue(conn, "montage", montage_id)
    encoded = []
    monkeypatch.setattr(assembly, "assemble", lambda *args, **kwargs: encoded.append(args))

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    assert read_montage(conn, montage_id) == ("failed", None, None, "загруженный трек не читается")
    assert encoded == [], "трек отвергнут после того, как начали собирать ролик"


@pytest.mark.parametrize("form", ["в списке рядом с треком полки", "один, в заказе до списков"])
def test_an_uploaded_track_gone_from_the_bucket_fails_the_montage_at_once_and_says_why(
    conn, s3, memory, tmp_path, monkeypatch, form
):
    """Своего трека нет в бакете — файл пропал, и ко второй попытке он не появится. Отказ с первой
    же, с причиной и без сборки, как у трека, который не читается: три повтора и «не удалось
    собрать ролик» сказали бы организатору только то, что ролика нет."""
    track_key = f"music/{uuid.uuid4()}"
    params = (
        {"music_tracks": ["warm", track_key]}
        if form.startswith("в списке")
        else {"music_track_key": track_key}
    )
    montage_id = order(conn, memory, params)
    ready_material(conn, s3, memory, tmp_path)
    job_id = worker.enqueue(conn, "montage", montage_id)
    encoded = []
    monkeypatch.setattr(assembly, "assemble", lambda *args, **kwargs: encoded.append(args))

    assert worker.run_once(conn) is True

    assert conn.execute(
        "SELECT status, attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone() == ("done", 0)
    assert read_montage(conn, montage_id) == ("failed", None, None, "загруженный трек не найден")
    assert encoded == [], "трек отвергнут после того, как начали собирать ролик"


@pytest.mark.parametrize(
    "failure",
    [
        ClientError(
            {
                "Error": {"Code": "503", "Message": "Service Unavailable"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
            "HeadObject",
        ),
        EndpointConnectionError(endpoint_url="http://storage"),
        ReadTimeoutError(endpoint_url="http://storage"),
    ],
    ids=["хранилище недоступно", "сеть", "таймаут"],
)
def test_an_uploaded_track_the_bucket_did_not_hand_over_is_retried_like_any_other_failure(
    conn, s3, memory, tmp_path, monkeypatch, failure
):
    """Трек в бакете есть, а хранилище его не отдало: такой сбой проходит сам, так что это не отказ,
    а обычный повтор очереди — ролик ждёт следующей попытки, а не падает с причиной про трек."""
    track_key = f"music/{uuid.uuid4()}"
    s3.upload_file(str(tone(tmp_path / "own.wav")), storage.bucket(), track_key)
    montage_id = order(conn, memory, {"music_tracks": ["warm", track_key]})
    ready_material(conn, s3, memory, tmp_path)
    job_id = worker.enqueue(conn, "montage", montage_id)
    real = storage.client

    def flaky():
        client = real()
        download = client.download_file

        def download_file(bucket, key, *args, **kwargs):
            if key == track_key:
                raise failure
            return download(bucket, key, *args, **kwargs)

        client.download_file = download_file
        return client

    monkeypatch.setattr(storage, "client", flaky)
    encoded = []
    monkeypatch.setattr(assembly, "assemble", lambda *args, **kwargs: encoded.append(args))

    assert worker.run_once(conn) is True

    assert conn.execute(
        "SELECT status, attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone() == ("queued", 1)
    status, _, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("processing", None), "сбой хранилища принят за отказ"
    assert encoded == []


def test_the_tracks_of_an_order_become_one_soundtrack_in_the_order_they_were_given(tmp_path):
    """Список любой длины — одна дорожка: треки идут друг за другом в заданном порядке, стык
    проходит коротким фейдом, и дорожка выходит короче суммы ровно на него. Порядок проверяется
    звуком: первым отдан тон, вторым тишина, и в дорожке громкое стоит впереди тихого — собранная
    задом наперёд, она звучала бы наоборот."""
    tracks = [tone(tmp_path / "tone.wav"), hush(tmp_path / "hush.wav")]

    soundtrack = assembly.one_track(tracks, tmp_path)

    assert float(probe(soundtrack)["format"]["duration"]) == pytest.approx(
        6.0 - assembly.TRACK_FADE_SECONDS, abs=0.2
    )
    assert loudness(soundtrack, at=0.2) > loudness(soundtrack, at=3.6) + 30, (
        "треки в дорожке идут не в том порядке, в каком их заказали"
    )
    twice = assembly.one_track([tracks[0], tracks[0]], tmp_path)
    assert float(probe(twice)["format"]["duration"]) == pytest.approx(
        6.0 - assembly.TRACK_FADE_SECONDS, abs=0.2
    ), "один и тот же трек в списке дважды — два трека, а не один"
    # И два вырожденных случая: собирать нечего, и собирать не из чего.
    assert assembly.one_track(tracks[:1], tmp_path) is tracks[0], "один трек незачем пересобирать"
    assert assembly.one_track([], tmp_path) is None, "пустой список — это «без музыки»"


def test_a_montage_under_a_list_of_tracks_is_cut_to_the_beat_of_the_whole_soundtrack(
    conn, s3, memory, tmp_path
):
    """Свой трек и трек полки вперемешку: воркер склеивает их в одну дорожку и ищет ритм по ней —
    склейки идут в такт всему списку, а не первому файлу из него.

    Проверяется, как и под одним загруженным треком, по раскладке готового файла: каждая склейка —
    на бите. Биты берутся у дорожки, собранной здесь же и из тех же двух файлов: это ровно та, под
    которую собрали ролик.
    """
    track_key = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(click_track(tmp_path / "click.wav")), storage.bucket(), track_key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": [track_key, "warm"]})
    # Клип первым (он старше всех). Набирать заказанные тридцать секунд тут нечем — клип и так
    # взят целиком, а фотографии не растягиваются (`selection.stretch`).
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)
    s3.upload_file(str(video(tmp_path / "clip.mp4")), storage.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assembled = {
        stream["codec_type"]
        for stream in probe(fetch(s3, video_key, tmp_path / "montage.mp4"))["streams"]
    }
    assert assembled == {"video", "audio"}, "картинка и музыка обе на месте"

    soundtrack = assembly.one_track([tmp_path / "click.wav", library.track_file("warm")], tmp_path)
    beats = rhythm.beats(soundtrack)
    assert beats, "в дорожке из метронома и музыки есть темп, иначе проверять нечего"
    pulled_to_the_beat(layout_of(conn, montage_id), beats, (3.0, 2.5, 2.5, 2.5))
    assert about(duration, 9.0), "подгонка под биты изменила длину ролика"


@pytest.mark.parametrize(
    "seconds", [3.0, 40.0], ids=["shorter than the video", "longer than the video"]
)
def test_a_track_shorter_or_longer_than_the_video_plays_exactly_as_long_as_the_picture(
    conn, s3, memory, tmp_path, seconds
):
    """A short track goes round again and a long one is cut off with the picture — the music is
    never told how long to play, the montage is cut to the material. A tone has no tempo, so the
    pieces stay as chosen: nine seconds either way, and the sound lasts to the last of them."""
    own = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(tone(tmp_path / "own.wav", seconds=seconds)), storage.bucket(), own)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": [own]})
    ready_material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert duration == pytest.approx(9.0, abs=0.3), "ролик подстроился под длину трека"
    assembled_file = fetch(s3, video_key, tmp_path / "montage.mp4")
    streams = {stream["codec_type"]: stream for stream in probe(assembled_file)["streams"]}
    assert float(streams["audio"]["duration"]) == pytest.approx(duration, abs=0.3), (
        "звук кончился раньше картинки или пережил её"
    )
    assert loudness(assembled_file, at=duration - 3.2, seconds=1.0) > -45, (
        "к концу ролика музыка смолкла"
    )


@pytest.mark.parametrize("shape", ["by a shelf id", "by an uploaded key"])
def test_a_failed_montage_ordered_before_track_lists_is_assembled_again_the_way_it_was_ordered(
    conn, s3, memory, tmp_path, monkeypatch, shape
):
    """Повтор упавшего ролика: строка лежит в базе так, как её записали до списков, — с
    `music_track_id` или `music_track_key` и без `music_tracks` — и статусом `failed`. Собирается
    она
    как собиралась: под тот трек, что в ней назван, и после этого не failed и без старой ошибки."""
    own = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(tone(tmp_path / "own.wav", seconds=7.0)), storage.bucket(), own)
    params = {"target_seconds": 30}
    params.update(
        {"music_track_id": "calm"} if shape == "by a shelf id" else {"music_track_key": own}
    )
    montage_id = order(conn, memory, params)
    conn.execute(
        "UPDATE montages SET status = 'failed', error_text = 'не удалось собрать ролик' "
        "WHERE id = %s",
        (montage_id,),
    )
    ready_material(conn, s3, memory, tmp_path)
    played = []
    assembling = assembly.assemble

    def watched(pieces, orientation, track, *rest, **kwargs):
        played.append((track == library.track_file("calm"), ffmpeg.duration(track)))
        return assembling(pieces, orientation, track, *rest, **kwargs)

    monkeypatch.setattr(assembly, "assemble", watched)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key, error_text) == ("ready", f"montages/{montage_id}.mp4", None)
    ((shelf, length),) = played
    if shape == "by a shelf id":
        assert shelf, "под трек полки, что назван в заказе"
    else:
        assert not shelf and length == pytest.approx(7.0, abs=0.2), (
            "под свой трек, что назван в заказе"
        )


def test_an_order_with_no_music_at_all_keeps_the_live_sound_of_the_videos(
    conn, s3, memory, tmp_path
):
    """Пустой список — «без музыки», и тогда ролик звучит сам собой: у клипов остаётся их
    собственный звук, фотографии молчат (макет MontageMusicNone, «Только живой звук из видео»).
    Клип тут новее всех, поэтому стоит в конце ролика: конец должен звучать, начало — нет.

    Ритма в таком ролике нет вовсе — подгонять склейки не подо что, и куски держатся ровно столько,
    сколько им дали отсев и набор заказанной длины: три фотографии по 2,5 секунды и трёхсекундный
    клип, минус полсекунды на каждую из трёх склеек. Заказанные тридцать секунд набрать нечем —
    клип и так взят целиком, а фотографии не растягиваются, — так что длина тут ровная и видно,
    что куски никто не двигал. Под музыку те же четыре файла дают другую длину (тест выше).
    """
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready", age_hours=5 - index)
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
    _, key = add_item(conn, memory, "video", status="ready")  # новее всех → последним в ролике
    s3.upload_file(str(video(tmp_path / "clip.mp4")), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert duration == pytest.approx(9.0, abs=0.3), "без музыки куски никто не двигал"

    assembled = fetch(s3, video_key, tmp_path / "montage.mp4")
    sound = [stream for stream in probe(assembled)["streams"] if stream["codec_type"] == "audio"]
    assert [stream["codec_name"] for stream in sound] == ["aac"], "дорожка одна, и она есть"
    assert loudness(assembled, at=7.2) > loudness(assembled, at=0.5) + 30, (
        "в конце ролика должен звучать клип, а в начале — молчать фотография"
    )


def test_the_live_sound_of_a_montage_without_music_fades_at_both_ends_instead_of_clicking(
    conn, s3, memory, tmp_path
):
    """Звук, оборванный на полуслове, щёлкает: живой звук держится в краях ролика тем же
    полусекундным фейдом, что и склейки. Мерится по готовому файлу, как и на ревью: первые и
    последние сотни миллисекунд против середины куска. Оба края ролика — клипы со звуком (старший
    и новейший), между ними фотография и клип, снятый без звука, — оба молчат, но склеиваются с
    остальными; а заказано больше, чем есть, — так что клипы растянуты и середина первого из них
    лежит далеко от обоих краёв и от склеек."""
    montage_id = order(conn, memory, {"target_seconds": 20, "music_tracks": []})
    for name, kind, age, make in (
        ("first.mp4", "video", 5, long_clip),
        ("still.jpg", "photo", 4, jpeg),
        ("mute.mp4", "video", 3, mute_clip),
        ("last.mp4", "video", 0, long_clip),
    ):
        _, key = add_item(conn, memory, kind, status="ready", age_hours=age)
        s3.upload_file(str(make(tmp_path / name)), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assembled = fetch(s3, video_key, tmp_path / "montage.mp4")
    length = float(probe(assembled)["format"]["duration"])
    middle = loudness(assembled, at=2.5, seconds=1.0)
    assert middle > -40, "в середине клипа должен быть слышен его живой звук"
    assert loudness(assembled, at=0, seconds=0.1) < middle - 10, "звук начинается сразу, с щелчка"
    assert loudness(assembled, at=length - 0.1, seconds=0.1) < middle - 10, (
        "живой звук обрывается с концом ролика на полной громкости, а не гаснет"
    )


def test_without_music_every_piece_gets_a_track_even_when_it_has_nothing_to_say(tmp_path):
    """Звук ролика без музыки склеивается тем же перехлёстом, что и картинка, а склеивать
    что-то с ничем нельзя: дорожка есть у каждого куска — у клипа своя, у фотографии и у клипа,
    снятого без звука, тишина той же формы и той же длины."""
    silent_ones = (
        ("фотография", selection.Piece(kind="photo", seconds=2.0, path=jpeg(tmp_path / "p.jpg"))),
        (
            "немой клип",
            selection.Piece(kind="video", seconds=2.0, path=mute_clip(tmp_path / "mute.mp4")),
        ),
        (
            "клип со звуком",
            selection.Piece(kind="video", seconds=2.0, path=video(tmp_path / "loud.mp4")),
        ),
    )
    for name, piece in silent_ones:
        clip = tmp_path / f"clip-{len(name)}.mp4"
        assembly._normalise(piece, "landscape", clip, live_sound=True)

        normalised = probe(clip)
        assert {stream["codec_type"] for stream in normalised["streams"]} == {"video", "audio"}, (
            f"у куска «{name}» нет звуковой дорожки — сборке нечего будет склеивать"
        )
        assert float(normalised["format"]["duration"]) == pytest.approx(2.0, abs=0.15), (
            f"кусок «{name}» вышел не той длины, и склейки разъедутся на эту разницу"
        )


def test_what_the_order_says_about_music_is_read_the_old_way_too():
    """Форма заказа: список треков, пустой список — «без музыки», и заказ старой формы, в котором
    списка нет вовсе. Последнее и есть повтор упавшего ролика: он лежит в базе так, как его
    записали до списков, и собраться должен как собирался — свой трек, трек полки или «тёплая» по
    умолчанию."""
    assert music.ordered({"music_tracks": ["warm", "music/abc"]}) == (["warm", "music/abc"], None)
    assert music.ordered({"music_tracks": []}) == ([], None)

    assert music.ordered({}) == ([library.DEFAULT_TRACK], None)
    assert music.ordered({"music_track_id": "calm"}) == (["calm"], None)
    assert music.ordered({"music_track_key": "music/abc", "music_track_id": "calm"}) == (
        ["music/abc"],
        None,
    ), "ключ своего трека сильнее id полки, как и было"

    named, refusal = music.ordered({"music_tracks": ["warm", "polka"]})
    assert named is None and "polka" in refusal
    # Чужой ключ бакета — не трек: только полка и `music/…` качаются под музыку.
    for stranger in ("originals/abc", "tracks/warm.mp3", "../music/abc"):
        named, refusal = music.ordered({"music_tracks": [stranger]})
        assert named is None and stranger in refusal
    assert music.ordered({"music_tracks": ["warm", "music/abc", "warm", "music/abc"]}) == (
        ["warm", "music/abc", "warm", "music/abc"],
        None,
    ), "список отдаётся как заказан: дубли не сливаются и порядок не меняется"
    named, refusal = music.ordered({"music_track_id": "polka"})
    assert named is None and "polka" in refusal


# Что организатор может написать у трека: буквы разных письменностей, эмодзи, перевод строки,
# кавычки и обратный слэш (JSON их экранирует), край api в 500 символов и втрое больше.
AWKWARD_WISHES = [
    'Тихо 🎂 и 日本語, é — «под тост»\nвторая строка\t"в кавычках" \\ слэш',
    "я" * 500,
    "🎂" * 1500,
]


def test_the_director_hears_each_wish_at_the_track_it_was_written_to_and_nowhere_else(tmp_path):
    """Дорожка словами для режиссёра (`music.heard`) из заказа в том виде, в каком его пишет api:
    пожелание стоит у своего трека и у каждого его повтора, у остальных поля `wish` нет вовсе, а
    пожелание к треку, которого в списке нет, не приложено ни к чему и ничего не роняет. Текст
    доходит буква в букву — пожелание не чистится и не режется: обрезает его api."""
    own = f"music/{uuid.uuid4()}"
    files = {own: tone(tmp_path / "own.wav", seconds=4.0)}

    def heard(params):
        named, refusal = music.ordered(params)
        assert refusal is None
        paths = [files.get(one) or library.track_file(one) for one in named]
        return music.heard(named, paths, params.get("music_wishes") or {})

    assert heard(api_params()) == [], "без музыки в запрос уходит пустой список"
    assert heard(api_params(["calm"])) == [
        {"track": "calm", "seconds": round(ffmpeg.duration(library.track_file("calm")), 1)}
    ]

    said = heard(
        api_params(
            [own, "calm", "road", "night"], {own: "  начало  ", "night": "к финалу", "road": "   "}
        )
    )
    assert [(one["track"], one.get("wish")) for one in said] == [
        ("свой трек", "начало"),
        ("calm", None),
        ("road", None),
        ("night", "к финалу"),
    ]
    assert all("wish" not in one for one in said[1:3]), "у трека без пожелания поля нет"
    assert said[0]["seconds"] == 4.0, "свой трек назван словами, а длина — его собственная"
    assert own not in json.dumps(said), "ключ бакета в запрос не уходит"

    # Повтор в списке — то же пожелание у каждого появления (api повторов не пропускает, но воркер
    # читает и такой заказ).
    twice = heard({"music_tracks": [own, "warm", own], "music_wishes": {own: "и в конце"}})
    assert [one.get("wish") for one in twice] == ["и в конце", None, "и в конце"]

    # Пожелание не к тому, что играет, и заказ вовсе без музыки: воркер не падает и не прикладывает.
    stray = {"music_tracks": ["calm"], "music_wishes": {"night": "не сюда", own: "и не сюда"}}
    assert [one.get("wish") for one in heard(stray)] == [None]
    assert heard({"music_tracks": [], "music_wishes": {"calm": "некому"}}) == []

    for wish in AWKWARD_WISHES:
        (only,) = heard(api_params(["warm"], {"warm": wish}))
        assert only["wish"] == wish.strip() and json.loads(json.dumps(only)) == only


def test_a_track_nobody_has_fails_the_montage_instead_of_retrying(conn, memory):
    montage_id = order(conn, memory, {"music_track_id": "polka"})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "трек" in error_text


@pytest.mark.parametrize("track_id", library.TRACKS)
def test_every_track_of_the_library_is_music_that_plays(track_id):
    """Five files on disk, and a montage ordered with one of them falls over only when somebody
    happens to order *that* one — so all five are opened here, not just the default. A track
    renamed, moved or never committed fails this test instead of somebody's video."""
    played = probe(library.track_file(track_id))

    assert "audio" in [stream["codec_type"] for stream in played["streams"]]
    assert float(played["format"]["duration"]) > 30, "music, not a second and a half of signal"


def library_in_the_bucket(s3):
    """What the bucket holds under each track's key: its size and type, or None for one that is not
    there at all."""
    heads = {}
    for track_id in library.TRACKS:
        try:
            head = s3.head_object(Bucket=storage.bucket(), Key=library.track_key(track_id))
        except s3.exceptions.ClientError:
            heads[track_id] = None
        else:
            heads[track_id] = (head["ContentLength"], head["ContentType"], head["LastModified"])
    return heads


def test_the_library_is_published_to_the_bucket_for_the_app_to_play(s3):
    """The five files are this worker's own disk, and «Послушать» is the api's and the phone's — so
    the library has to be in the bucket too, under the keys the api builds from the same ids."""
    for track_id in library.TRACKS:
        s3.delete_object(Bucket=storage.bucket(), Key=library.track_key(track_id))

    assert sorted(library.publish_tracks(s3)) == sorted(library.TRACKS)

    landed = library_in_the_bucket(s3)
    assert [track_id for track_id, head in landed.items() if head is None] == []
    assert {track_id: head[:2] for track_id, head in landed.items()} == {
        track_id: (library.track_file(track_id).stat().st_size, "audio/mpeg")
        for track_id in library.TRACKS
    }, "all five, whole, and served as music rather than as bytes of nothing in particular"

    # Run on every deploy: five mp3s must not be written over every time.
    assert library.publish_tracks(s3) == [], "what is already there is not uploaded again"
    assert library_in_the_bucket(s3) == landed, "and the objects themselves were not touched"
