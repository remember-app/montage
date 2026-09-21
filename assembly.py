"""Сборка: the chosen pieces laid onto one canvas and cut into one file.

Everything in here is ffmpeg, and nothing in here asks why these pieces and not others. It is
handed a list of pieces that already know their length, where in their own file each is cut from
and where in the video it goes; it normalises every one of them onto the montage's canvas and
crossfades them end to end with the track over the top. Sharpness, exposure, bursts, the
organizer's two lists, a plan from a neural network — all of that is `selection`'s, and how long a
piece is held so the join lands on a beat is `rhythm`'s, so the way a video is directed can change
without a line of this module moving.

A piece, as far as this module is concerned, is anything that carries `path`, `kind`, `seconds` and
`start`. That is the whole contract — it is deliberately not an import.

Every call here goes through `media.py`'s helpers: one worker, one way of running the tool.
"""

import dataclasses
import pathlib

import media

# The common canvas everything in one montage is normalised onto — the organizer picks which at the
# order (`params.orientation`). Landscape by default: the finished video leaves the app through
# «поделиться» and 16:9 is what plays everywhere; portrait is the same 1080p turned on its side, for
# a montage meant to be watched on the phone it was shot on. Either way material shot the other way
# round fills the canvas and loses its edges to it, rather than keeping all of itself inside black
# bars (человек, 2026-09-21) — a video is watched, and bars burnt into the file stay there
# everywhere it is ever played.
CANVASES = {"landscape": (1920, 1080), "portrait": (1080, 1920)}
DEFAULT_ORIENTATION = "landscape"  # what an order that says nothing about orientation means
FPS = 30

CROSSFADE_SECONDS = 0.5
MUSIC_FADE_SECONDS = 1.5
TRACK_FADE_SECONDS = 1.0  # стык двух соседних треков заказа в одной дорожке

# Тишина под кусок, которому нечем звучать, — фотография и клип, снятый без звука. Нужна только
# ролику без музыки: там дорожка ролика склеивается из дорожек кусков, а склеить что-то с ничем
# нельзя. 44,1 кГц стерео — то же, во что приводится и живой звук клипов: так стыкуются все куски.
SAMPLE_RATE = 44100
SILENCE = f"anullsrc=r={SAMPLE_RATE}:cl=stereo"

# The library: the five tracks of the «Под что монтировать» screen
# (`design/screens/MontageMusic.html`), in its order and under its names — «Яркая», «Тёплая»,
# «Спокойная», «Ночь», «Дорога». Ids only, because the worker shows no names: which name is which
# track, who wrote it, where the file came from and under what licence is `tracks/README.md`, kept
# beside the files themselves.
TRACKS = ("bright", "warm", "calm", "night", "road")
DEFAULT_TRACK = "warm"  # what an order that says nothing about music means

TRACKS_DIR = pathlib.Path(__file__).parent / "tracks"


def track_file(track_id):
    """The library's file for a track. The organizer's own track never comes through here — that
    one arrives out of R2 and reaches `assemble` the same way, just from somewhere else."""
    return TRACKS_DIR / f"{track_id}.mp3"


def track_key(track_id):
    """Where the same track lives in the bucket, for «Послушать» on the «Под что монтировать»
    screen to play before any video is ordered. The montage itself never reads these objects — it
    cuts from the file on disk beside it — but that disk is this worker's alone, and the api and the
    phone have to be able to reach the music too. Same id in the key as in `TRACKS`, so the api
    builds it without knowing anything more about the library."""
    return f"tracks/{track_id}.mp3"


def publish_tracks(s3=None):
    """The library into the bucket. Returns the ids that were actually uploaded.

    What is already there is left alone rather than written over: this runs on deploy, a track's
    file never changes under an id, and re-uploading five mp3s on every run would only spend the
    bucket's write quota. The flip side is that a track *replaced* under an id that is already
    published needs its object deleted first — README.md, «Треки».
    """
    s3 = s3 or media.storage()
    published = []
    for track_id in TRACKS:
        try:
            s3.head_object(Bucket=media.bucket(), Key=track_key(track_id))
            continue
        except s3.exceptions.ClientError:
            # Nothing under this key — or a bucket that refused to answer, and then the upload
            # below is refused the same way and says so. Storage that cannot be reached at all is
            # not a `ClientError` and never gets here: it comes out of `head_object`, and the one
            # caller (`worker.py tracks`) turns it into a line rather than a traceback.
            pass
        media._upload(s3, track_file(track_id), track_key(track_id), "audio/mpeg")
        published.append(track_id)
    return published


def one_track(tracks, workdir):
    """Треки заказа одной дорожкой: один за другим в заданном порядке, стык — короткий фейд.

    `tracks` — готовые файлы, и чей какой (полки или свой, загруженный организатором) здесь уже
    неразличимо, как и везде ниже по сборке. Пустой список — это заказ «без музыки»: ответ None, и
    тогда звук ролика собирается из самих клипов. Один трек отдаётся файлом как есть: перекодировать
    его, чтобы склеить сам с собой, не с чем.

    Дальше эта дорожка живёт как любой трек — в ней ищется ритм и она подгоняется под длину ролика
    (короче ролика — играет с начала ещё раз, длиннее — обрывается вместе с картинкой).
    """
    if len(tracks) < 2:
        return tracks[0] if tracks else None
    # Перехлёст не длиннее половины самого короткого трека: acrossfade смешивает две дорожки
    # внахлёст, и перехлёст длиннее того, что смешивается, — это уже не стык. Длина, которой файл
    # не назвал, считается достаточной: узнать её больше неоткуда, а фейд в секунду переживёт любая
    # музыка.
    shortest = min((media._duration(track) or TRACK_FADE_SECONDS * 2) for track in tracks)
    fade = max(min(TRACK_FADE_SECONDS, shortest / 2), 0.1)

    inputs, chain = [], []
    for index, track in enumerate(tracks):
        # `-protocol_whitelist` — входная опция, и ffmpeg относит её к тому входу, перед которым
        # она стоит: у каждого трека свой, как у каждого клипа в `_assemble`.
        inputs += ["-protocol_whitelist", "file", "-i", str(track)]
        # Треки разные — mp3 полки и что угодно, что загрузил организатор, — а acrossfade склеивает
        # только одинаковое. Отсюда общий формат на входе каждого.
        chain.append(
            f"[{index}:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo[t{index}]"
        )
    label = "t0"
    for index in range(1, len(tracks)):
        chain.append(f"[{label}][t{index}]acrossfade=d={fade:.3f}[j{index}]")
        label = f"j{index}"

    out = workdir / "track.m4a"
    media._ffmpeg(*inputs, "-filter_complex", ";".join(chain), "-map", f"[{label}]",
                  "-c:a", "aac", "-b:a", "192k", str(out))
    return out


def length_of(pieces):
    """How long these pieces run once crossfaded — each join eats half a second of the total."""
    return sum(piece.seconds for piece in pieces) - CROSSFADE_SECONDS * (len(pieces) - 1)


def assemble(pieces, orientation, track, workdir, out, on_clip=None):
    """The chosen pieces as one video at `out`, and the normalised clips it was made of.

    `track` is a file — the montage's whole soundtrack, however many tracks of whose it was built
    (`one_track`). Which of them came out of the library and which the organizer uploaded was
    settled before this module was called, and nothing here can tell them apart. `None` is the
    order that asked for no music at all, and then the video keeps the clips' own sound (doc-03,
    «Без музыки ролик идёт с живым звуком самих видео»).

    The clips come back because they are the only place the lengths are true: `_normalise` is asked
    for a length and the file that comes out of it is measured rather than believed, and whoever
    cuts a cover off the montage needs those measured numbers to find its opening piece in it.

    `on_clip` is called with how many pieces are encoded so far, every time one more is — the one
    thing this module tells anybody while it works, and the montage takes minutes: «склеено 12 из
    61» on the live screen is that number and nothing else. Nothing is asked of it and nothing is
    read back, so a caller with nothing to show simply does not pass one.
    """
    clips = []
    for index, piece in enumerate(pieces):
        clip = workdir / f"clip-{index}.mp4"
        _normalise(piece, orientation, clip, live_sound=track is None)
        if on_clip:
            on_clip(index + 1)
        # Measured, not the length that was asked for. `-t` cuts against the *video* stream while
        # `_duration` reads the container, which is the longest stream in it — a clip whose audio
        # outruns its picture normalises shorter than declared. `_assemble` steps the crossfade
        # offsets along by these numbers, so a stale one walks the whole chain off the end and
        # ffmpeg truncates the montage without a word of complaint.
        clips.append(dataclasses.replace(
            piece, path=clip, seconds=media._duration(clip) or piece.seconds
        ))
    _assemble(clips, track, out)
    return clips


def _normalise(piece, orientation, out, live_sound=False):
    """One piece onto this montage's canvas: 1080p the way it was ordered, 30 fps, cropped, silent.

    The canvas is built here rather than held as a constant because it belongs to the order and not
    to the module — but every clip of one montage is normalised with the same `orientation`, which
    `_assemble`'s crossfades require: xfade joins two streams of one size or nothing.

    A piece shaped unlike the canvas is scaled until it covers the whole of it (`increase`) and the
    overflow is cut off the edges, centred — the way a vertical video is made of horizontal material
    everywhere it is watched. The alternative is bars burnt into the file, and those are in it for
    good. Centred because nothing here knows where the picture is: the crop is blind to what it
    cuts, and moving it towards a face is `understanding`'s knowledge, not this module's.

    `force_divisible_by=2` keeps the scaled size even: the crop makes the output exact either way,
    but an odd intermediate is a size yuv420p has no chroma plane for.

    Silent because the finished video's only sound is the track (architecture.md: «звук исходных
    клипов по умолчанию убираем»), which also makes every clip's audio identical by having none.
    `live_sound` is the one order that says otherwise — no music, so the video sounds like the
    videos in it: a clip keeps its own track and everything with nothing to say (a photograph, a
    clip shot silent) gets silence of the same shape, because `_assemble` joins the clips' sound
    with the same crossfade as their picture and there is nothing to crossfade a missing stream
    with.

    A photograph is decoded, scaled and cropped **once** and then held: `loop` re-emits the one
    frame that came out of the crop for as long as the piece runs. Looping the *input* instead
    (`-loop 1`) hands the decoder the file again for every frame of the clip, and on a snapshot off
    a real camera — 8368×5584, 23 МБ — that is eighty-five decodes of forty-six megapixels: one
    such piece took seventeen minutes on the stand and the whole montage stood (TASK-183). The loop
    is infinite and `-t` is what ends it, so that `-t` is an *output* option here and may not be
    dropped: without it ffmpeg would encode the same frame forever.
    """
    width, height = CANVASES[orientation]
    fitted = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,"
        f"crop={width}:{height},setsar=1"
    )
    if piece.kind == "photo":
        # `loop` sits after the crop, so what it holds is one canvas-sized frame and not the
        # original, and `fps` after it is what gives that frame the montage's rate.
        canvas = f"{fitted},loop=loop=-1:size=1,fps={FPS},format=yuv420p"
        opened, cut = ("-i", str(piece.path)), ("-t", f"{piece.seconds}")
    else:
        canvas = f"{fitted},fps={FPS},format=yuv420p"
        opened = ("-ss", f"{piece.start}", "-t", f"{piece.seconds}", "-i", str(piece.path))
        cut = ()
    added, sound = (), ("-an",)
    if live_sound:
        its_own = piece.kind == "video" and _has_sound(piece.path)
        # `-shortest` для клипа, которому дописали тишину: она бесконечна, и без него кодировался
        # бы бесконечный кусок. Фотографию всё равно кончает её собственный `-t`.
        added = () if its_own else ("-f", "lavfi", "-i", SILENCE)
        sound = ("-map", "0:v", "-map", "0:a" if its_own else "1:a", "-c:a", "aac", "-b:a", "128k",
                 "-ar", str(SAMPLE_RATE), "-ac", "2", "-shortest")
    media._ffmpeg(
        *opened, *added, "-vf", canvas, *cut, *sound,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out),
    )


def _has_sound(path):
    """Есть ли в файле звуковая дорожка вообще. Клип, снятый с выключенным микрофоном, — обычное
    дело, и на попытку взять у него звук ffmpeg отвечает ошибкой, а не тишиной."""
    return bool(media._run(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
    ).strip())


def _assemble(clips, track, out):
    """The normalised clips into one file: crossfaded end to end, and the sound with them — the
    track faded over the top, or the clips' own sound when the order asked for no music."""
    total = length_of(clips)

    # Each xfade joins everything built so far to the next clip, starting half a second before the
    # chain ends — so the chain grows by the clip's length minus the overlap, which is `length_of`.
    chain, label, ends_at = [], "0:v", clips[0].seconds
    for index, clip in enumerate(clips[1:], start=1):
        offset = ends_at - CROSSFADE_SECONDS
        chain.append(
            f"[{label}][{index}:v]xfade=transition=fade:"
            f"duration={CROSSFADE_SECONDS}:offset={offset:.3f}[v{index}]"
        )
        label, ends_at = f"v{index}", offset + clip.seconds
    # A single clip is never crossfaded, so there is no filter output to map — the input itself is
    # the video, and it is named the way ffmpeg names inputs rather than filter outputs.
    video = f"[{label}]" if len(clips) > 1 else "0:v"

    music = ()
    if track is None:
        # Живой звук: дорожки кусков склеиваются тем же перехлёстом, что и картинки. acrossfade
        # накладывает конец одной на начало следующей ровно так, как xfade накладывает кадры, —
        # поэтому звук и изображение остаются вместе, а не расходятся на каждой склейке.
        joined = "0:a"
        for index in range(1, len(clips)):
            chain.append(f"[{joined}][{index}:a]acrossfade=d={CROSSFADE_SECONDS}[a{index}]")
            joined = f"a{index}"
        audio = f"[{joined}]" if len(clips) > 1 else "0:a"
    else:
        fade = min(MUSIC_FADE_SECONDS, total / 3)
        # A track shorter than the montage is played again from the top (`-stream_loop -1`), a
        # longer one is cut off with the picture, because `-shortest` is down below. Either way the
        # music never has to be told how long to play: the montage is cut to the material, not to
        # the track. `-protocol_whitelist` is the track's own, like every clip's below.
        music = ("-protocol_whitelist", "file", "-stream_loop", "-1", "-i", str(track))
        chain.append(
            f"[{len(clips)}:a]afade=t=in:st=0:d={fade:.3f},"  # the track is the input after the
            f"afade=t=out:st={total - fade:.3f}:d={fade:.3f}[a]"  # last clip
        )
        audio = "[a]"

    # `-protocol_whitelist` is an *input* option: ffmpeg scopes it to the group it precedes, so the
    # one `media._ffmpeg` puts at the front guards input 0 and nothing else. This is the worker's
    # only multi-input call, so it is the only place that has to repeat it — every clip and the
    # track get their own, and the whitelist means the same thing on all of them.
    inputs = [
        argument
        for clip in clips
        for argument in ("-protocol_whitelist", "file", "-i", str(clip.path))
    ]
    media._ffmpeg(
        *inputs, *music,
        # Один кусок без музыки — единственный случай, когда склеивать нечего ни там, ни там, и
        # цепочки фильтров нет вовсе: пустую ffmpeg не берёт.
        *(("-filter_complex", ";".join(chain)) if chain else ()),
        "-map", video, "-map", audio,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(out),
    )
