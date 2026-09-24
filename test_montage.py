"""Whole montages, ordered and assembled end to end — see README.md.

Real everything, like `test_media.py`: a real Postgres, a real bucket, real ffmpeg, and fixtures
ffmpeg draws on the spot. What the choosing does on its own, with no file behind it, is
`test_selection.py`; these are here for what only a real video can answer — that the material
actually comes out of the bucket, through both modules and into one playable file.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import json
import re
import subprocess
import uuid

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

import ai
import assembly
import media
import montage
import rhythm
import selection
import worker

# The recorded answer and the transport that serves it are `test_ai.py`'s, like everywhere else the
# cloud model is involved: there is one way to answer without a network in this suite.
from test_ai import Recorded

# The fixtures and the ffmpeg-drawn material are `test_media.py`'s — a second copy of a Postgres
# fixture is a second thing to keep true. pytest resolves fixtures by name in this module, so they
# have to be imported by name and not through the module.
from test_media import (  # noqa: F401
    FIXTURES, add_item, conn, ffmpeg, fetch, memory, near, photo, probe, s3, video,
)


# --- the whole thing --------------------------------------------------------


def order(conn, memory, params):
    """A `montages` row as the api leaves it when a video is ordered."""
    if conn.execute("SELECT to_regclass('montages')").fetchone()[0] is None:
        pytest.skip("montages is missing — run the api's migrations first (see README.md)")
    memory_id, user_id = memory
    return str(conn.execute(
        """INSERT INTO montages (memory_id, ordered_by, params)
           VALUES (%s, %s, %s::jsonb) RETURNING id""",
        (memory_id, user_id, json.dumps(params)),
    ).fetchone()[0])


def read_montage(conn, montage_id):
    return conn.execute(
        "SELECT status, video_key, duration_seconds, error_text FROM montages WHERE id = %s",
        (montage_id,),
    ).fetchone()


def poster_of(conn, montage_id):
    """The montage's cover key, `None` until a video has actually been assembled. Read on its own
    rather than added to `read_montage`: only the two tests below have anything to say about it."""
    return conn.execute(
        "SELECT poster_key FROM montages WHERE id = %s", (montage_id,)
    ).fetchone()[0]


# A montage is cut to the beat of its track, and the last piece takes up what the joins moved
# (`rhythm.fit`): the video comes out the sum of its pieces to within what a piece that cannot be
# held longer leaves over and a frame or two of encoding. A second of slack — the promise's own —
# covers that and still tells a montage of four pieces from one of three or five: there are three
# seconds between those. What the windows below are for is how many pieces went in, not the music
# — the rhythm itself is `test_rhythm.py`'s, and a cut to an uploaded track is below.
RHYTHM_SLACK = 1.0


def about(duration, expected):
    """The montage came out about as long as it should, with room for the pull towards the beat."""
    return abs(duration - expected) <= RHYTHM_SLACK


BLACK = 16  # the value `selection._exposed` already counts as black


def dimmest_edge(path):
    """The dimmest edge in the whole file: the brightest pixel along each of a frame's four edges,
    and the smallest of those numbers over every second of the video.

    This is how a black bar is caught on the finished file instead of in the filter string that was
    supposed to prevent it. A letterboxed montage has a whole edge row of nothing but black, so the
    brightest pixel along it is 0; a canvas the picture fills has something in every edge. The
    frames come from `selection`'s own 96×54 grey sampling — coarse, but a bar is a third of the
    canvas, and the material here is a test pattern with its corners lit.
    """
    width = selection.SAMPLE_WIDTH
    dimmest = 255
    for frame in selection._grey_frames(path, per_second=selection.SAMPLE_FPS):
        rows = [frame[at:at + width] for at in range(0, len(frame), width)]
        edges = (rows[0], rows[-1], [row[0] for row in rows], [row[-1] for row in rows])
        dimmest = min(dimmest, *(max(edge) for edge in edges))
    return dimmest


def material(conn, s3, memory, tmp_path):
    """Three photographs and a clip in the bucket, `ready`, as the previews leave them. Their ids
    come back in the order the montage reads them, for the tests that name one in the order."""
    ids = []
    for index in range(3):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(item_id)
    item_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), key)
    return [*ids, item_id]


def test_a_memory_of_photos_and_a_clip_becomes_one_playable_video(conn, s3, memory, tmp_path):
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm"})
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, video_key, error_text) == ("ready", f"montages/{montage_id}.mp4", None)
    # Three stills of two and a half seconds and a four-second clip cut to the three it has — ten
    # and a half seconds of material, less half a second for each of the three crossfades. Pieces
    # butted together instead of overlapped would come to 10.5, which is outside the window even
    # with the slack. Thirty seconds were ordered and there is no more material to reach them with:
    # the clip is already whole and a photograph is never stretched (`selection.stretch`).
    assert about(duration, 9.0)

    assembled_file = fetch(s3, video_key, tmp_path / "montage.mp4")
    assembled = probe(assembled_file)
    assert about(float(assembled["format"]["duration"]), 9.0)
    tracks = {stream["codec_type"]: stream for stream in assembled["streams"]}
    assert set(tracks) == {"video", "audio"}, "picture and music both landed"
    assert tracks["video"]["codec_name"] == "h264"
    assert tracks["audio"]["codec_name"] == "aac"
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (1920, 1080), (
        "an order that says nothing about orientation is the horizontal canvas, as it always was"
    )
    # And the vertical clip in that material (1280×1440) fills the horizontal canvas instead of
    # standing between two black bars: it is cropped to the canvas, not padded into it.
    assert dimmest_edge(assembled_file) > BLACK, "чёрные поля по краям горизонтального ролика"

    # And the cover beside it, on the same row — the list of videos shows a still, not a black
    # rectangle, before anyone presses play.
    poster_key = poster_of(conn, montage_id)
    assert poster_key == f"montages/{montage_id}.jpg"
    poster_file = fetch(s3, poster_key, tmp_path / "poster.jpg")
    poster = probe(poster_file)["streams"][0]
    assert (poster["codec_name"], poster["width"]) == ("mjpeg", media.THUMBNAIL_WIDTH)
    # Not just a jpeg of the right size — a frame worth showing. The same judgement the worker
    # applies to source material, turned on its own cover: an all-black or blown-out still fails
    # it, which is exactly what a poster cut from frame zero of a clip that opens dark would be.
    assert selection._exposed(selection._grey_frames(poster_file)[0])


def long_clip(path, seconds=12):
    """A clip with room in it. The happy path's own clip runs three seconds, which the default cut
    already takes whole — there is nothing left in it to hold longer, and a video ordered longer
    than its material is exactly what this material cannot show."""
    ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=24:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )
    return path


@pytest.mark.parametrize("music", ["a track of the shelf, the way it was ordered before lists",
                                   "an uploaded track and a shelf track in one list"])
def test_a_montage_reaches_the_length_that_was_ordered_by_holding_its_clips_longer(
    conn, s3, memory, tmp_path, music
):
    """The ordered length as a promise, on a real file: two twelve-second clips and two stills come
    to eleven and a half seconds at the lengths the отбор cuts them at, and twenty seconds were
    ordered. The clips are held longer — equally, and well inside their own files — until the video
    is as long as it was asked to be. Before this, the same order came back at those eleven and a
    half seconds without a word (TASK-193).

    Under a list of the organizer's own track and one of the shelf's, with a note to the director
    that no director is there to read, the file is the same: picture, and music that plays to the
    very end. The uploaded track is three seconds of silence, so a shelf track that was not joined
    on behind it would leave the rest of the video quiet."""
    if music.startswith("a track"):
        params = {"target_seconds": 20, "music_track_id": "warm"}
    else:
        own = f"music/{uuid.uuid4()}.wav"
        s3.upload_file(str(hush(tmp_path / "own.wav")), media.bucket(), own)
        params = {"target_seconds": 20, "music_tracks": [own, "warm"],
                  "music_note": "первый трек — на вступление"}
    montage_id = order(conn, memory, params)
    for index in range(2):
        _, key = add_item(conn, memory, "video", status="ready")
        s3.upload_file(str(long_clip(tmp_path / f"clip-{index}.mp4")), media.bucket(), key)
    for index in range(2):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # 4 + 4 + 2.5 + 2.5 less three joins is 11.5, so 8.5 seconds are missing and the two clips take
    # 4.25 each — 8.25 seconds of a twelve-second file apiece.
    assert about(duration, 20.0), "ролик вышел не той длины, что заказали"
    assembled_file = fetch(s3, video_key, tmp_path / "m.mp4")
    assembled = probe(assembled_file)
    assert about(float(assembled["format"]["duration"]), 20.0)
    tracks = {stream["codec_type"]: stream for stream in assembled["streams"]}
    assert (tracks["video"]["codec_name"], tracks["audio"]["codec_name"]) == ("h264", "aac")
    assert float(tracks["audio"]["duration"]) == pytest.approx(
        float(tracks["video"]["duration"]), abs=0.3
    ), "дорожка кончилась раньше картинки или пережила её"
    # The clips are on the screen at the start and the stills at the end, and the clips' own sound
    # is left out of a montage with music: a loud end can only be the music.
    assert loudness(assembled_file, at=duration - 3.2, seconds=1.0) > -45, "в конце ролика нет музыки"


@pytest.mark.parametrize("music", [["night"], []], ids=["under «Ночь»", "with no music"])
def test_a_montage_under_music_comes_out_the_length_it_was_ordered(
    conn, s3, memory, tmp_path, music
):
    """The ordered length under music, on a real file at a length a test can afford: a long clip
    and twelve stills ordered for thirty seconds. Under «Ночь», a hundred beats a minute, joins each
    pulled from the one before would all fall the same way and leave the video three seconds short;
    with no music there is nothing to pull. Either way the file ffprobe reads runs the thirty
    seconds to within one."""
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": music})
    _, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(long_clip(tmp_path / "clip.mp4")), media.bucket(), key)
    for index in range(12):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    seconds = float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"])
    assert abs(seconds - 30.0) <= 1.0, f"тридцать секунд заказали, а ролик — {seconds:.2f} с"


def layout_of(conn, montage_id):
    """Which frames the montage is made of and where each of them stands in it, in the order they
    are shown — `montage_frames`, the table the frame sheet's «в каких роликах» is answered from."""
    if conn.execute("SELECT to_regclass('montage_frames')").fetchone()[0] is None:
        pytest.skip("montage_frames is missing — run the api's migrations first (see README.md)")
    return conn.execute(
        """SELECT media_id::text, start_seconds, duration_seconds
           FROM montage_frames WHERE montage_id = %s ORDER BY start_seconds""",
        (montage_id,),
    ).fetchall()


def test_the_finished_montage_says_where_in_it_each_frame_it_took_stands(
    conn, s3, memory, tmp_path
):
    """Every piece of the assembled video has its second and its length written down, and the one
    the organizer struck out has nothing written down at all — «Сведения о кадре» shows a frame the
    videos it is in and where, and a frame that is in none of them shows none (doc-02).

    Checked against the video that was actually made rather than against numbers named here: the
    pieces are pulled to the beat of the track, so what any of them is held for is the rhythm's to
    decide — but they must add up, join by join, to the file that came out of it.
    """
    ids = material(conn, s3, memory, tmp_path)
    montage_id = order(conn, memory, {"target_seconds": 30, "exclude_media_ids": [ids[0]]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    layout = layout_of(conn, montage_id)
    assert [media_id for media_id, _, _ in layout] == ids[1:], (
        "не те кадры или не в том порядке: у вошедших запись есть, у исключённого — нет"
    )
    assert layout[0][1] == 0.0, "the video opens on its first piece"
    # Each piece begins where the one before it ends, less the half-second the two of them share.
    for (_, start, seconds), (_, next_start, _) in zip(layout, layout[1:]):
        assert next_start == pytest.approx(start + seconds - assembly.CROSSFADE_SECONDS)
    # And the last of them ends with the video itself: the layout is of this file, not of a plan
    # that was made before it was encoded.
    assert layout[-1][1] + layout[-1][2] == pytest.approx(duration, abs=0.2)


def test_a_montage_assembled_again_replaces_its_layout_instead_of_doubling_it(conn, memory):
    """Re-ordering the same video writes its layout a second time, and the first one has to go with
    it: a frame that was dropped from the remake would otherwise still say it stands in it, and one
    that stayed would be listed twice.

    Straight at the writing, with pieces made up here: what a remake is, as far as this table is
    concerned, is one more run of it under the same montage id — and assembling two whole videos to
    see that costs minutes of ffmpeg for one `delete`.
    """
    montage_id = order(conn, memory, {"target_seconds": 30})
    dropped, _ = add_item(conn, memory, "photo", status="ready")
    kept, _ = add_item(conn, memory, "photo", status="ready")

    montage._write_layout(conn, montage_id, [selection.Piece(media_id=dropped, seconds=3.0)])
    montage._write_layout(conn, montage_id, [selection.Piece(media_id=kept, seconds=4.0)])

    assert layout_of(conn, montage_id) == [(kept, 0.0, 4.0)]


def test_a_layout_that_cannot_be_written_does_not_cost_the_montage(conn, memory):
    """The video is the thing; its layout is a line in a sheet about one frame. Minutes of encoding
    are not thrown away because an insert did not land, so a write that fails is a warning in the
    log — and the connection is still good afterwards, or the montage would die on the next
    statement instead of on this one."""
    gone = str(uuid.uuid4())  # no `montages` row, so both foreign keys refuse the rows

    montage._write_layout(conn, gone, [selection.Piece(media_id=str(uuid.uuid4()), seconds=3.0)])

    assert layout_of(conn, gone) == []


def click_track(path, gap=1.6):
    """Thirty seconds of metronome: one click every 1.6 seconds. The tempo is known because it was
    put there, and it is deliberately slow — under a track like this the pieces are pulled visibly
    off their own lengths."""
    ffmpeg("-f", "lavfi",
           "-i", f"aevalsrc='0.8*sin(2*PI*1200*t)*exp(-40*mod(t,{gap}))':d=30", str(path))
    return path


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
    s3.upload_file(str(click_track(tmp_path / "click.wav")), media.bucket(), track_key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_key": track_key})
    # The clip first (oldest): its file runs three seconds and the pieces after it are stills.
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "the beats changed how long the video is"
    beats = rhythm.beats(tmp_path / "click.wav")
    assert beats, "a metronome is a tempo, or there is nothing here to check"
    pulled_to_the_beat(layout_of(conn, montage_id), beats, (3.0, 2.5, 2.5, 2.5))


@pytest.mark.parametrize("bad_track", ["bytes that are not media", "a photograph",
                                       "a clip shot with the microphone off"])
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
    made, name = {"bytes that are not media": (unopenable, "track"),
                  "a photograph": (photo, "track.jpg"),
                  "a clip shot with the microphone off": (mute_clip, "track.mp4")}[bad_track]
    s3.upload_file(str(made(tmp_path / name)), media.bucket(), track_key)
    montage_id = order(conn, memory, {"music_tracks": ["warm", track_key]})
    material(conn, s3, memory, tmp_path)
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
    params = ({"music_tracks": ["warm", track_key]} if form.startswith("в списке")
              else {"music_track_key": track_key})
    montage_id = order(conn, memory, params)
    material(conn, s3, memory, tmp_path)
    job_id = worker.enqueue(conn, "montage", montage_id)
    encoded = []
    monkeypatch.setattr(assembly, "assemble", lambda *args, **kwargs: encoded.append(args))

    assert worker.run_once(conn) is True

    assert conn.execute(
        "SELECT status, attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone() == ("done", 0)
    assert read_montage(conn, montage_id) == ("failed", None, None, "загруженный трек не найден")
    assert encoded == [], "трек отвергнут после того, как начали собирать ролик"


@pytest.mark.parametrize("failure", [
    ClientError({"Error": {"Code": "503", "Message": "Service Unavailable"},
                 "ResponseMetadata": {"HTTPStatusCode": 503}}, "HeadObject"),
    EndpointConnectionError(endpoint_url="http://storage"),
    ReadTimeoutError(endpoint_url="http://storage"),
], ids=["хранилище недоступно", "сеть", "таймаут"])
def test_an_uploaded_track_the_bucket_did_not_hand_over_is_retried_like_any_other_failure(
    conn, s3, memory, tmp_path, monkeypatch, failure
):
    """Трек в бакете есть, а хранилище его не отдало: такой сбой проходит сам, так что это не отказ,
    а обычный повтор очереди — ролик ждёт следующей попытки, а не падает с причиной про трек."""
    track_key = f"music/{uuid.uuid4()}"
    s3.upload_file(str(tone(tmp_path / "own.wav")), media.bucket(), track_key)
    montage_id = order(conn, memory, {"music_tracks": ["warm", track_key]})
    material(conn, s3, memory, tmp_path)
    job_id = worker.enqueue(conn, "montage", montage_id)
    storage = media.storage

    def flaky():
        client = storage()
        download = client.download_file

        def download_file(bucket, key, *args, **kwargs):
            if key == track_key:
                raise failure
            return download(bucket, key, *args, **kwargs)

        client.download_file = download_file
        return client

    monkeypatch.setattr(media, "storage", flaky)
    encoded = []
    monkeypatch.setattr(assembly, "assemble", lambda *args, **kwargs: encoded.append(args))

    assert worker.run_once(conn) is True

    assert conn.execute(
        "SELECT status, attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone() == ("queued", 1)
    status, _, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("processing", None), "сбой хранилища принят за отказ"
    assert encoded == []


# --- список треков, и пустой список -----------------------------------------


def tone(path, seconds=3.0, frequency=880):
    """Секунды слышимого звука — трек, который в собранной дорожке ни с чем не спутаешь."""
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}", str(path))
    return path


def hush(path, seconds=3.0):
    """Столько же тишины — второй трек, слышный ровно наоборот: по нему и видно порядок."""
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds), str(path))
    return path


def loudness(path, at, seconds=1.2):
    """Насколько громко звучит кусочек файла: `mean_volume` в децибелах, как его считает ffmpeg.
    Тишина — около −91 дБ (или −inf), живой звук и музыка — на десятки децибел громче."""
    done = subprocess.run(
        ("ffmpeg", "-nostdin", "-y", "-ss", str(at), "-t", str(seconds), "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"),
        capture_output=True, text=True, check=True,
    )
    (mean,) = re.findall(r"mean_volume: (\S+) dB", done.stderr)
    return float(mean)


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
    s3.upload_file(str(click_track(tmp_path / "click.wav")), media.bucket(), track_key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": [track_key, "warm"]})
    # Клип первым (он старше всех). Набирать заказанные тридцать секунд тут нечем — клип и так
    # взят целиком, а фотографии не растягиваются (`selection.stretch`).
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assembled = {stream["codec_type"] for stream in probe(
        fetch(s3, video_key, tmp_path / "montage.mp4")
    )["streams"]}
    assert assembled == {"video", "audio"}, "картинка и музыка обе на месте"

    soundtrack = assembly.one_track([tmp_path / "click.wav", assembly.track_file("warm")], tmp_path)
    beats = rhythm.beats(soundtrack)
    assert beats, "в дорожке из метронома и музыки есть темп, иначе проверять нечего"
    pulled_to_the_beat(layout_of(conn, montage_id), beats, (3.0, 2.5, 2.5, 2.5))
    assert about(duration, 9.0), "подгонка под биты изменила длину ролика"


@pytest.mark.parametrize("seconds", [3.0, 40.0], ids=["shorter than the video", "longer than the video"])
def test_a_track_shorter_or_longer_than_the_video_plays_exactly_as_long_as_the_picture(
    conn, s3, memory, tmp_path, seconds
):
    """A short track goes round again and a long one is cut off with the picture — the music is
    never told how long to play, the montage is cut to the material. A tone has no tempo, so the
    pieces stay as chosen: nine seconds either way, and the sound lasts to the last of them."""
    own = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(tone(tmp_path / "own.wav", seconds=seconds)), media.bucket(), own)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": [own]})
    material(conn, s3, memory, tmp_path)
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
    assert loudness(assembled_file, at=duration - 3.2, seconds=1.0) > -45, "к концу ролика музыка смолкла"


@pytest.mark.parametrize("shape", ["by a shelf id", "by an uploaded key"])
def test_a_failed_montage_ordered_before_track_lists_is_assembled_again_the_way_it_was_ordered(
    conn, s3, memory, tmp_path, monkeypatch, shape
):
    """Повтор упавшего ролика: строка лежит в базе так, как её записали до списков, — с
    `music_track_id` или `music_track_key` и без `music_tracks` — и статусом `failed`. Собирается она
    как собиралась: под тот трек, что в ней назван, и после этого не failed и без старой ошибки."""
    own = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(tone(tmp_path / "own.wav", seconds=7.0)), media.bucket(), own)
    params = {"target_seconds": 30}
    params.update({"music_track_id": "calm"} if shape == "by a shelf id" else {"music_track_key": own})
    montage_id = order(conn, memory, params)
    conn.execute("UPDATE montages SET status = 'failed', error_text = 'не удалось собрать ролик' "
                 "WHERE id = %s", (montage_id,))
    material(conn, s3, memory, tmp_path)
    played = []
    assembling = assembly.assemble

    def watched(pieces, orientation, track, *rest, **kwargs):
        played.append((track == assembly.track_file("calm"), media._duration(track)))
        return assembling(pieces, orientation, track, *rest, **kwargs)

    monkeypatch.setattr(assembly, "assemble", watched)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key, error_text) == ("ready", f"montages/{montage_id}.mp4", None)
    (shelf, length), = played
    if shape == "by a shelf id":
        assert shelf, "под трек полки, что назван в заказе"
    else:
        assert not shelf and length == pytest.approx(7.0, abs=0.2), "под свой трек, что назван в заказе"


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
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    _, key = add_item(conn, memory, "video", status="ready")  # новее всех → последним в ролике
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), key)
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
    for name, kind, age, make in (("first.mp4", "video", 5, long_clip), ("still.jpg", "photo", 4, photo),
                                  ("mute.mp4", "video", 3, mute_clip), ("last.mp4", "video", 0, long_clip)):
        _, key = add_item(conn, memory, kind, status="ready", age_hours=age)
        s3.upload_file(str(make(tmp_path / name)), media.bucket(), key)
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


def mute_clip(path, seconds=3.0):
    """Клип, снятый с выключенным микрофоном: картинка есть, звуковой дорожки нет вовсе."""
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size=640x480:rate=24:duration={seconds}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path


def test_without_music_every_piece_gets_a_track_even_when_it_has_nothing_to_say(tmp_path):
    """Звук ролика без музыки склеивается тем же перехлёстом, что и картинка, а склеивать
    что-то с ничем нельзя: дорожка есть у каждого куска — у клипа своя, у фотографии и у клипа,
    снятого без звука, тишина той же формы и той же длины."""
    silent_ones = (
        ("фотография", selection.Piece(kind="photo", seconds=2.0, path=photo(tmp_path / "p.jpg"))),
        ("немой клип", selection.Piece(kind="video", seconds=2.0,
                                       path=mute_clip(tmp_path / "mute.mp4"))),
        ("клип со звуком", selection.Piece(kind="video", seconds=2.0,
                                           path=video(tmp_path / "loud.mp4"))),
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
    assert montage._ordered_music({"music_tracks": ["warm", "music/abc"]}) == (
        ["warm", "music/abc"], None
    )
    assert montage._ordered_music({"music_tracks": []}) == ([], None)

    assert montage._ordered_music({}) == ([assembly.DEFAULT_TRACK], None)
    assert montage._ordered_music({"music_track_id": "calm"}) == (["calm"], None)
    assert montage._ordered_music({"music_track_key": "music/abc", "music_track_id": "calm"}) == (
        ["music/abc"], None
    ), "ключ своего трека сильнее id полки, как и было"

    named, refusal = montage._ordered_music({"music_tracks": ["warm", "polka"]})
    assert named is None and "polka" in refusal
    # Чужой ключ бакета — не трек: только полка и `music/…` качаются под музыку.
    for stranger in ("originals/abc", "tracks/warm.mp3", "../music/abc"):
        named, refusal = montage._ordered_music({"music_tracks": [stranger]})
        assert named is None and stranger in refusal
    assert montage._ordered_music({"music_tracks": ["warm", "music/abc", "warm", "music/abc"]}) == (
        ["warm", "music/abc", "warm", "music/abc"], None
    ), "список отдаётся как заказан: дубли не сливаются и порядок не меняется"
    named, refusal = montage._ordered_music({"music_track_id": "polka"})
    assert named is None and "polka" in refusal


def test_a_montage_ordered_portrait_is_assembled_the_other_way_up(conn, s3, memory, tmp_path):
    """The organizer asked for a video to watch on a phone. The same material, the same length, the
    same everything else — cut into 1080×1920 instead of 1920×1080. The numbers are written out
    here rather than read off `assembly.CANVASES`, which is the thing under test."""
    montage_id = order(conn, memory, {"target_seconds": 30, "orientation": "portrait"})
    # Stills 1200×800 and a clip 1280×1440 — material shot both ways round, and the portrait canvas
    # crops every piece of it to itself rather than padding any into it.
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "turning the canvas does not change what goes in it"

    assembled_file = fetch(s3, video_key, tmp_path / "portrait.mp4")
    tracks = {stream["codec_type"]: stream for stream in probe(assembled_file)["streams"]}
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (1080, 1920)
    # The real size is only half of it: a vertical file made of horizontal stills is exactly where
    # black bars are burnt in, and 1080×1920 is 1080×1920 with or without them.
    assert dimmest_edge(assembled_file) > BLACK, "чёрные поля по краям вертикального ролика"

    # The cover too: it is cut from the assembled file at the thumbnail width and keeps that file's
    # own shape, so a vertical montage gets a vertical cover with nobody deciding it should. The one
    # output whose orientation is a consequence rather than a choice, so it is the one that could
    # quietly go back to 16:9 while everything above stayed green.
    poster = probe(fetch(s3, poster_of(conn, montage_id), tmp_path / "poster.jpg"))["streams"][0]
    assert poster["width"] == media.THUMBNAIL_WIDTH
    assert poster["height"] > poster["width"], "the cover stands the same way up as the video"


def test_an_orientation_nobody_has_fails_the_montage_instead_of_retrying(conn, memory):
    """The api writes one of two words here and refuses the rest, so a third one is the contract
    having moved on one side only — and it would be the same word on all three attempts."""
    montage_id = order(conn, memory, {"orientation": "diagonal"})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "ориентация" in error_text


def dark_opening_clip(path):
    """A clip that opens black for its first 1.5s, then three and a half seconds of a sharp,
    properly exposed pattern — the shape of a real phone clip that takes a moment to find its
    exposure. `_best_moment` finds its best frame only after the black stretch, but the piece is cut
    *centred* on that moment and pulled back to fit the file, so the cut can still start inside the
    black part. The happy-path fixture (`material`) always puts a photo first, so its montage never
    opens on a video and this shape of bug never gets a chance to show up."""
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:size=1280x720:rate=24:duration=1.5",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=3.5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )
    return path


def test_a_montage_that_opens_on_a_video_with_a_dark_start_gets_a_bright_cover(
    conn, s3, memory, tmp_path
):
    """The regression the poster fix (ревью m-16) exists for: a phone clip that opens black. The
    piece cut from it starts inside that black stretch (centred on the best moment, then pulled back
    to the file), so a poster taken off frame zero of the assembled file — the pre-fix behaviour —
    would be black. Made to open the montage (oldest item, sorted first) so this is actually the
    first piece the cover is cut from, unlike the happy-path fixture where a photo always leads."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)  # oldest → opens the montage
    s3.upload_file(str(dark_opening_clip(tmp_path / "dark.mp4")), media.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)

    poster_key = poster_of(conn, montage_id)
    poster_file = fetch(s3, poster_key, tmp_path / "poster.jpg")
    assert selection._exposed(selection._grey_frames(poster_file)[0]), (
        "the cover must be the opening clip's best moment, not the dark frame it was cut from"
    )


def planning_over(monkeypatch, media_ids, fixture="gemini_plan.json"):
    """The recorded plan (`fixtures/gemini_plan*.json`) as the only cloud model this worker can
    find.

    The ids in it are this run's own: a montage is planned over material whose ids Postgres makes up
    at insert time, so the three the answer was written about are rewritten into the three actually
    ordered. Everything else — what a plan is, what a step of it carries — is the recorded file's,
    and nothing here goes anywhere near the network.
    """
    recorded = Recorded(fixture)
    for index, media_id in enumerate(media_ids, start=1):
        recorded.answer = recorded.answer.replace(f"m-{index}".encode(), media_id.encode())
    monkeypatch.setattr(ai, "from_env", lambda *_, **__: ai.Gemini("test-key", transport=recorded))
    return recorded


def test_a_montage_ordered_with_a_story_is_cut_to_the_directors_plan(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The director end to end: a real order with a story in it, real material out of the bucket,
    and a video that is the plan's rather than the отбор's — a different length, out of a different
    three files. What the plan does with pieces on its own is `test_director.py`; this is here for
    what only a real video can answer, that the order actually reaches it and comes back assembled.

    Every other montage in this file is cut with no cloud model at all, which is the other half of
    the same proof: none of them changed.
    """
    ids = material(conn, s3, memory, tmp_path)
    for media_id, description in zip(ids, ("накрывают на стол", "вносят торт", "все за столом")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    net = planning_over(monkeypatch, ids[:3])
    own = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(tone(tmp_path / "own.wav", seconds=5.0)), media.bucket(), own)
    montage_id = order(conn, memory, {
        "target_seconds": 30, "story_text": "день рождения, всё вокруг торта",
        "music_tracks": [own, "warm", own], "music_note": "  торт — под припев второго трека  ",
    })
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # The plan's own three, four and three seconds, less half a second for each of two joins, and
    # room for rhythm's pull towards the beat (`about`, `RHYTHM_SLACK`) — not the отбор's ten to
    # eleven, which is what these same four files come to on their own. The clip, which the plan
    # leaves out and the отбор takes, is the visible half of the difference.
    assert about(duration, 9.0), "the video is as long as the plan made it"
    assert about(float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"]), 9.0)

    question = json.loads(net.sent.data)["input"][0]["text"]
    assert "всё вокруг торта" in question, "the story the organizer typed reached the director"
    assert "вносят торт" in question, "and so did what the pipeline already knew about the material"
    (request,), = conn.execute(
        "SELECT request FROM ai_requests WHERE operation = 'plan_montage' AND ref = %s", (montage_id,)
    ).fetchall()
    said = json.loads(request["prompt"])
    warm = round(media._duration(assembly.track_file("warm")), 1)
    assert said["music"] == [{"track": "свой трек", "seconds": 5.0}, {"track": "warm", "seconds": warm},
                             {"track": "свой трек", "seconds": 5.0}], (
        "the soundtrack goes to the director in the order it plays, an own track by no key of a bucket"
    )
    assert said["music_note"] == "торт — под припев второго трека", "the wish is the organizer's words"


def test_the_directors_film_is_held_to_the_ordered_length_the_same_way(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The other half of the promise: the length is filled after the film is planned, not inside
    the planning, so a plan that comes to nine seconds against a sixteen-second order is held to it
    exactly as the отбор's own cut is. The plan opens on the long clip, and that is what the
    missing seconds go into — the model is not asked again and nothing is added to the film."""
    clip_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(long_clip(tmp_path / "clip.mp4")), media.bucket(), key)
    ids = [clip_id]
    for index in range(2):
        photo_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(photo_id)
    for media_id, description in zip(ids, ("вносят торт", "все за столом", "задувают свечи")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    planning_over(monkeypatch, ids)
    montage_id = order(conn, memory, {"target_seconds": 16, "story_text": "день рождения"})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # The plan's own three, four and three seconds come to nine; the seven that are missing all go
    # into the clip, which the plan put first — ten seconds of its twelve.
    assert about(duration, 16.0), "план не дотянули до заказанной длины"


def unopenable(path):
    """Bytes in the bucket that ffmpeg will not open — a truncated upload, a file the phone
    mislabelled. Not a playlist: that is the security case and `test_media.py` has it."""
    path.write_bytes(b"this is not a photo and never was")
    return path


def crushed(path):
    """A photo with the same edges `photo()` has — sharp, easily past `MIN_SHARPNESS` — pushed
    almost to black. ffmpeg opens it without complaint, unlike `unopenable`; only the exposure
    check has anything to say about it, which is the point: a flat black frame would be rejected on
    blur alone and prove nothing about the histogram half of `_best_moment`."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=640x480:duration=1", "-vf", "eq=brightness=-0.85",
        "-frames:v", "1", str(path),
    )
    return path


def lopsided(path):
    """Two seconds of picture under three and a half of sound, so the container calls it 3.5 —
    `format=duration` is the longest stream, and `-t` cuts against the picture."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )
    return path


def test_a_file_that_will_not_open_is_left_out_rather_than_failing_the_video(
    conn, s3, memory, tmp_path
):
    """One corrupt upload is that file's problem, not the memory's. It used to raise straight out
    of the handler: three retries, then a montage failed for good — and «повторить» met the same
    file, so any one member could deny the organizer a video permanently."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(unopenable(tmp_path / "broken")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "the four that opened went in, the broken one was left out"


def test_a_badly_exposed_photo_is_left_out_rather_than_failing_the_video(
    conn, s3, memory, tmp_path
):
    """A different code path from a file ffmpeg refuses to open: this one decodes fine and the
    отсев turns it down on exposure, the one frame in it being crushed almost to black. The unit
    tests on `choose` never exercise this — they score every piece by hand."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "the crushed photo had no usable frame and was left out uncounted"


def test_a_photo_the_organizer_marked_include_is_in_the_video_the_algorithm_notwithstanding(
    conn, s3, memory, tmp_path
):
    """The same crushed photo as above, and the same verdict from the algorithm — except this time
    the order names it. «Включить» is the organizer overruling the automatic отсев, so the piece
    the test above proves is dropped is here proved to be in the video."""
    material(conn, s3, memory, tmp_path)
    crushed_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    montage_id = order(
        conn, memory, {"target_seconds": 30, "include_media_ids": [crushed_id]}
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Four stills of 2.5 and the three-second clip instead of three stills, less half a second
    # for each of four joins.
    assert about(duration, 11.0), "the crushed photo went in because the order asked for it"


def test_the_floor_counts_the_pieces_the_organizer_asked_for_too(conn, s3, memory, tmp_path):
    """A memory the отсев turns down almost entirely, and an organizer who overrules it. The floor
    is «сколько кусков реально войдёт», so the two crushed photos named in «включить» count towards
    it — refusing the order here would be the automatic отсев beating the person, which is the one
    thing «включить» exists to prevent (doc-03, «Кадры, отмеченные включить, есть в ролике»)."""
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "good.jpg")), media.bucket(), key)
    wanted = []
    for index in range(2):
        crushed_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(crushed(tmp_path / f"crushed-{index}.jpg")), media.bucket(), key)
        wanted.append(crushed_id)
    montage_id = order(conn, memory, {"target_seconds": 30, "include_media_ids": wanted})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 6.5), "all three went in: three stills, less half a second for each join"


def test_no_montage_runs_past_the_ceiling_whatever_the_organizer_asked_for_and_the_beats_did(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The ten-minute edge on a real file, at a ceiling lowered to eight seconds: what is asked is
    whether the edge holds, not whether ten minutes can be encoded. Five stills, every one marked
    «включить», come to 10.5 seconds, and a fast metronome pulls their joins about on top of that.
    The video is cut from its tail to the ceiling — the stills that stay are the first ones, in
    their order — and the file ffprobe reads runs no longer than it."""
    monkeypatch.setattr(selection, "MAX_SECONDS", 8.0)
    track_key = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(click_track(tmp_path / "click.wav", gap=0.43)), media.bucket(), track_key)
    ids = []
    for index in range(5):
        item_id, key = add_item(conn, memory, "photo", status="ready", age_hours=5 - index)
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(item_id)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_key": track_key,
                                      "include_media_ids": ids})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    seconds = float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"])
    assert duration <= 8.0 and seconds <= 8.0, f"the file runs {seconds}s past the ceiling"
    assert seconds == pytest.approx(8.0, abs=0.1), "cut short of the ceiling rather than to it"
    layout = [media_id for media_id, _, _ in layout_of(conn, montage_id)]
    assert layout == ids[:len(layout)] and len(layout) < len(ids), "not cut from the tail"


def test_a_photo_the_organizer_marked_exclude_stays_out_of_the_video(conn, s3, memory, tmp_path):
    """The other half of the same word from the organizer, on material the algorithm is perfectly
    happy with: a photo named in «исключить» is not in the video, and the rest of the material
    fills the time it would have taken."""
    photo_id = material(conn, s3, memory, tmp_path)[0]
    montage_id = order(conn, memory, {"target_seconds": 30, "exclude_media_ids": [photo_id]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Two stills and the clip — eight seconds of material, less half a second for each of two
    # joins.
    assert about(duration, 7.0), "one of the four pieces was left out on the order's own word"


def test_the_order_judges_a_photograph_on_the_same_copy_the_proposal_did(
    conn, s3, memory, tmp_path
):
    """Which file the отсев looks at when a video is ordered — told from outside the only way it
    can be: the large copy beside this photograph is black and the original behind it is a picture.

    The frame is thrown out, which means the order judged the copy — the same file
    `screening.media_screening` wrote its proposal on and the same one the piece would have been cut
    from. Two places looking at two different files would show the organizer one thing in the
    gallery and put another in the video.
    """
    photo_id = material(conn, s3, memory, tmp_path)[0]
    black = tmp_path / "black.jpg"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=2048x1366", "-frames:v", "1", str(black))
    s3.upload_file(str(black), media.bucket(), f"previews/{photo_id}.jpg")
    conn.execute("UPDATE media_items SET preview_key = %s WHERE id = %s",
                 (f"previews/{photo_id}.jpg", photo_id))
    montage_id = order(conn, memory, {"target_seconds": 30})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Two stills and the clip: the same eight seconds less two joins as a piece struck out by hand.
    assert about(duration, 7.0), "снимок с чёрной копией остался в ролике — судили оригинал"


def test_the_floor_does_not_count_the_pieces_the_organizer_struck_out(conn, s3, memory, tmp_path):
    """Four files the отсев is happy with, two of them named in «исключить». The floor is what can
    actually end up on screen, and `selection.choose` throws the struck-out two away whatever the
    floor thought — so a minute-long order backed by two pieces is refused here rather than
    assembled into a seven-second «ролик»."""
    struck = material(conn, s3, memory, tmp_path)[:2]
    montage_id = order(conn, memory, {"target_seconds": 60, "exclude_media_ids": struck})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key) == ("failed", None), "nothing was assembled out of the two left"
    assert "не годится" in error_text


def test_the_floor_treats_a_piece_named_on_both_lists_as_struck_out(conn, s3, memory, tmp_path):
    """The order contradicting itself is settled the same way here as in `selection.choose`:
    «исключить» is the one that cannot be undone once a video is made, so it is the one the floor
    counts too. Three good photos, one of them on both lists, are really only two — and a floor that
    let «включить» save it here would pass an order `selection.choose` then cuts down to two anyway."""
    ids = []
    for index in range(3):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(item_id)
    montage_id = order(conn, memory, {
        "target_seconds": 60, "include_media_ids": [ids[0]], "exclude_media_ids": [ids[0]],
    })
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key) == ("failed", None), "the contradiction did not save it from the floor"
    assert "не годится" in error_text


def test_too_little_material_that_actually_opens_fails_without_retrying(conn, s3, memory, tmp_path):
    """The floor is about material that can go in the video, not rows in the table: three rows of
    which two are unreadable would otherwise have been assembled into one three-second still."""
    montage_id = order(conn, memory, {})
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "good.jpg")), media.bucket(), key)
    for index in range(montage.MIN_ITEMS - 1):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(unopenable(tmp_path / f"broken-{index}")), media.bucket(), key)
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "не годится" in error_text


def test_a_clip_is_laid_into_the_chain_at_the_length_it_really_is(conn, s3, memory, tmp_path):
    """The crossfade offsets step along by each clip's length. Booking a clip in at the length it
    was *asked* for, when its picture is shorter than its container says, walks the rest of the
    chain off the end — and ffmpeg truncates the montage without a word of complaint."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    # Oldest, so the chronological order puts it first and the drift hits every join after it.
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)
    s3.upload_file(str(lopsided(tmp_path / "lopsided.mp4")), media.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, _ = read_montage(conn, montage_id)
    assert status == "ready"
    # Two seconds of picture and three stills of 2.5, less half a second for each of three
    # joins: 8. Booked at the container's 3.5 instead, ffmpeg truncates the chain and the whole
    # montage comes out at three seconds — nowhere near this window, slack for the rhythm or not.
    assert about(duration, 8.0), "the clip went in at its real two seconds"
    assert about(float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"]), 8.0)


def test_a_photograph_is_scaled_once_for_the_piece_and_not_once_a_frame(tmp_path, monkeypatch):
    """The root of TASK-183: looping the *input* (`-loop 1`) hands the file back to the decoder for
    every frame of the clip, so a 46-megapixel snapshot was decoded and scaled eighty-five times
    over for three seconds of video — seventeen minutes for one piece on the stand, with the whole
    montage behind it. The frame is held after the crop instead, by the filter.

    Checked on the command because that is where the decoding is decided, and the cost of the
    difference is hardware: the same three seconds of video comes out of both, one of them fifty
    times slower. The clip itself is made for real below — an infinite loop that `-t` fails to stop
    would hang here rather than in somebody's montage.
    """
    commands = []
    monkeypatch.setattr(media, "_ffmpeg", lambda *args: commands.append(args))
    held = selection.Piece(kind="photo", seconds=3.0, path=tmp_path / "photo.jpg")
    assembly._normalise(held, "landscape", tmp_path / "clip.mp4")
    (command,) = commands

    assert "-loop" not in command, "исходник зациклен — фото декодируется заново на каждом кадре"
    filters = command[command.index("-vf") + 1]
    assert filters.count("scale=") == 1, "фото масштабируется больше одного раза"
    assert filters.index("loop=") > filters.index("crop="), "удерживается не обрезанный кадр"

    monkeypatch.undo()
    photo(tmp_path / "photo.jpg")
    assembly._normalise(held, "landscape", tmp_path / "clip.mp4")
    clip = probe(tmp_path / "clip.mp4")
    assert float(clip["format"]["duration"]) == pytest.approx(3.0, abs=0.05), "не та длина куска"
    picture = clip["streams"][0]
    assert (picture["width"], picture["height"]) == assembly.CANVASES["landscape"]


def test_a_montage_of_hundreds_of_pieces_is_assembled_without_ffmpeg_running_out_of_threads(
    tmp_path, monkeypatch
):
    """Ten minutes of montage is about three hundred pieces, and all of them are open in one ffmpeg
    at once: a decoder with a thread pool of its own on every input took the process past its thread
    limit at about two hundred (ревью m-26). The pieces here are short and on a canvas the size of a
    stamp — what is being asked is only whether that many inputs can be open at the same time, and
    the picture is not the point. Five hundred, not three: the limit is threads, cores times inputs,
    and on this ten-core machine three hundred inputs of the same clips still fit under it with the
    fix taken out."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    still = photo(tmp_path / "still.jpg")
    pieces = [selection.Piece(kind="photo", seconds=0.6, path=still) for _ in range(500)]
    soundtrack = tone(tmp_path / "track.wav", seconds=40)
    out = tmp_path / "long.mp4"

    clips = assembly.assemble(pieces, "landscape", soundtrack, tmp_path, out)

    assert len(clips) == len(pieces)
    assembled = probe(out)
    assert float(assembled["format"]["duration"]) == pytest.approx(assembly.length_of(clips), abs=0.5)
    assert {stream["codec_type"] for stream in assembled["streams"]} == {"video", "audio"}


def test_a_plan_that_ends_on_the_ceiling_does_not_come_out_frames_past_it(tmp_path, monkeypatch):
    """Every clip is encoded to the nearest frame of what it was asked for, and the beats ask for
    lengths that are no whole number of frames: 2.52 seconds is 75.6 of them and comes out as 76.
    Five such stills planned to end on a ceiling of 10.6 seconds come out two frames past it, and
    those frames are cut off the file rather than left in it. The canvas is a stamp, as above."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    still = photo(tmp_path / "still.jpg")
    pieces = [selection.Piece(kind="photo", seconds=2.52, path=still) for _ in range(5)]
    out = tmp_path / "capped.mp4"

    clips = assembly.assemble(pieces, "landscape", tone(tmp_path / "track.wav", seconds=20),
                              tmp_path, out, max_seconds=10.6)

    assert assembly.length_of(clips) > 10.6, "the frames did not add up past the ceiling"
    assembled = probe(out)
    assert float(assembled["format"]["duration"]) <= 10.6
    assert {stream["codec_type"] for stream in assembled["streams"]} == {"video", "audio"}


def test_a_photograph_is_cut_from_the_large_copy_it_was_judged_on_unless_it_is_short(tmp_path):
    """The montage is cut from the 2048px copy the preview already made — the same file the отсев
    looked at — and not from the original off somebody's camera: the same picture on the canvas out
    of a tenth of the bytes. The copy's cap is on its long side, so whether it covers the canvas is
    measured rather than assumed: the very same file is enough for a landscape montage and short
    for a portrait one, and a photo whose copy never got made was judged on the original anyway.
    """
    original = tmp_path / "original"
    copy = photo(tmp_path / "large.jpg", size="2048x1366")
    judged_on_the_copy = selection.Piece(media_id="one", kind="photo", path=copy)
    judged_on_the_original = selection.Piece(media_id="two", kind="photo", path=original)
    originals = {"one": original, "two": original}

    (landscape,) = montage._back_to_originals([judged_on_the_copy], originals, "landscape", tmp_path)
    (portrait,) = montage._back_to_originals([judged_on_the_copy], originals, "portrait", tmp_path)
    (alone,) = montage._back_to_originals([judged_on_the_original], originals, "landscape", tmp_path)

    assert landscape.path == copy, "ролик режется не из копии"
    assert portrait.path == original, "1366 в высоту мало для холста 1080×1920 — нужен оригинал"
    assert alone.path == original


def pixel_at(video, at, x, y):
    """The colour at (x, y) of a finished video's frame at `at` seconds, at full size. Cut out two
    pixels square and averaged down to one: a yuv420 frame cannot be cropped to an odd size."""
    return tuple(subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"crop=2:2:{x}:{y},scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-"],
        capture_output=True, check=True,
    ).stdout)


@pytest.mark.parametrize("copy", ["no copy", "a copy made before previews stood HEIC up"])
def test_a_portrait_heic_stands_up_in_the_montage(conn, s3, memory, tmp_path, copy):
    """What an iPhone holds upright is a sensor frame with an `irot` note, which ffmpeg does not
    apply: cut from that file, or from a large copy made while previews did not either, the photo
    lies on its side in the video. `portrait-photo.heic` is 96×64 on the sensor with red in its
    top-left corner and blue in its bottom-right; seen upright it is 64×96, red top-right and blue
    bottom-left. On the 1920×1080 canvas it is scaled to the width and cropped to its middle band,
    which leaves a thin red strip at the top right and a thin blue one at the bottom left — and
    turned on its side, a big red block at the top left instead.
    """
    ids = []
    for index in range(2):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(item_id)
    heic_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(FIXTURES / "portrait-photo.heic"), media.bucket(), key)
    if copy != "no copy":
        # What the old preview made of it: ffmpeg's own reading, on its side, and big enough for the
        # canvas to be cut from without going back to the original.
        sideways = tmp_path / "sideways.jpg"
        ffmpeg("-i", str(FIXTURES / "portrait-photo.heic"), "-vf", "scale=2048:1365", str(sideways))
        s3.upload_file(str(sideways), media.bucket(), f"previews/{heic_id}.jpg")
        conn.execute("UPDATE media_items SET preview_key = %s WHERE id = %s",
                     (f"previews/{heic_id}.jpg", heic_id))
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": [],
                                      "include_media_ids": [*ids, heic_id]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    video_file = fetch(s3, video_key, tmp_path / "montage.mp4")
    (start, seconds), = [(start, seconds) for media_id, start, seconds in layout_of(conn, montage_id)
                         if media_id == heic_id]
    middle = start + seconds / 2  # clear of the crossfades on both sides
    assert near(pixel_at(video_file, middle, 1560, 20), (255, 0, 0)), "красной полосы справа вверху нет"
    assert near(pixel_at(video_file, middle, 360, 1050), (0, 0, 255)), "синей полосы слева внизу нет"
    assert not near(pixel_at(video_file, middle, 300, 100), (255, 0, 0)), "кадр лежит на боку"


def test_too_little_material_fails_the_montage_instead_of_retrying(conn, memory):
    """Not a transient failure: the library would be just as small on the third attempt, and the
    organizer needs the reason under the video rather than three useless retries."""
    montage_id = order(conn, memory, {})
    for _ in range(montage.MIN_ITEMS - 1):
        add_item(conn, memory, "photo", status="ready")
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "материал" in error_text
    # No video was assembled, so there is no frame to stand for one: the api serves `poster_url`
    # null and the organizer sees «повторить» rather than a cover for a video that does not exist.
    assert poster_of(conn, montage_id) is None


def test_a_track_nobody_has_fails_the_montage_instead_of_retrying(conn, memory):
    montage_id = order(conn, memory, {"music_track_id": "polka"})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "трек" in error_text


def test_a_montage_that_keeps_failing_fails_the_row(conn, memory):
    # Rows whose originals never made it into the bucket: the download fails every time.
    montage_id = order(conn, memory, {})
    for _ in range(montage.MIN_ITEMS):
        add_item(conn, memory, "photo", status="ready")
    job_id = worker.enqueue(conn, "montage", montage_id)

    for _ in range(worker.MAX_ATTEMPTS):
        assert worker.run_once(conn) is True
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,))  # skip the backoff

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "failed"
    # And the montage with it, so the organizer gets «повторить» and the reason it fell over.
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    # A fixed, human sentence — never the exception's own text. `error_text` goes out to every
    # member of the memory, and the exception here is a botocore message naming the bucket host.
    assert error_text == "не удалось собрать ролик"


@pytest.mark.parametrize("track_id", assembly.TRACKS)
def test_every_track_of_the_library_is_music_that_plays(track_id):
    """Five files on disk, and a montage ordered with one of them falls over only when somebody
    happens to order *that* one — so all five are opened here, not just the default. A track
    renamed, moved or never committed fails this test instead of somebody's video."""
    played = probe(assembly.track_file(track_id))

    assert "audio" in [stream["codec_type"] for stream in played["streams"]]
    assert float(played["format"]["duration"]) > 30, "music, not a second and a half of signal"


def library_in_the_bucket(s3):
    """What the bucket holds under each track's key: its size and type, or None for one that is not
    there at all."""
    heads = {}
    for track_id in assembly.TRACKS:
        try:
            head = s3.head_object(Bucket=media.bucket(), Key=assembly.track_key(track_id))
        except s3.exceptions.ClientError:
            heads[track_id] = None
        else:
            heads[track_id] = (head["ContentLength"], head["ContentType"], head["LastModified"])
    return heads


def test_the_library_is_published_to_the_bucket_for_the_app_to_play(s3):
    """The five files are this worker's own disk, and «Послушать» is the api's and the phone's — so
    the library has to be in the bucket too, under the keys the api builds from the same ids."""
    for track_id in assembly.TRACKS:
        s3.delete_object(Bucket=media.bucket(), Key=assembly.track_key(track_id))

    assert sorted(assembly.publish_tracks(s3)) == sorted(assembly.TRACKS)

    landed = library_in_the_bucket(s3)
    assert [track_id for track_id, head in landed.items() if head is None] == []
    assert {track_id: head[:2] for track_id, head in landed.items()} == {
        track_id: (assembly.track_file(track_id).stat().st_size, "audio/mpeg")
        for track_id in assembly.TRACKS
    }, "all five, whole, and served as music rather than as bytes of nothing in particular"

    # Run on every deploy: five mp3s must not be written over every time.
    assert assembly.publish_tracks(s3) == [], "what is already there is not uploaded again"
    assert library_in_the_bucket(s3) == landed, "and the objects themselves were not touched"


def test_an_api_that_does_not_answer_does_not_fail_a_finished_montage(monkeypatch):
    """The video exists and the row already says so — a failure here would rebuild the whole thing
    on the next attempt to deliver one push."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # discard: nothing is listening
    monkeypatch.setenv("INTERNAL_WORKER_TOKEN", "secret")

    montage._tell_the_api("a3b1f2e4-0000-4000-8000-000000000000")  # no exception is the assertion
