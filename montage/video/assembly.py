"""Сборка: the chosen pieces laid onto one canvas and cut into one file.

Everything in here is ffmpeg, and nothing in here asks why these pieces and not others. It is
handed a list of pieces that already know their length, where in their own file each is cut from,
where in the video it goes and what it is heard with; it normalises every one of them onto the
montage's canvas and crossfades them end to end, their own sound and the track mixed under them
piece by piece. Sharpness, exposure, bursts, the organizer's two lists, a plan from a neural network
— all of that is `selection`'s, and how long a piece is held so the join lands on a beat is
`rhythm`'s, so the way a video is directed can change without a line of this module moving.

A piece, as far as this module is concerned, is anything that carries `path`, `kind`, `seconds`,
`start` and `live`. That is the whole contract — it is deliberately not an import.

Every call here goes through `ffmpeg.py`: one worker, one way of running the tool.
"""

import dataclasses
import itertools
import math

from montage import ffmpeg

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

# How the pieces reach the montage's one encoder (`_assemble`): bare frames of the canvas. They
# carry nothing about their colour, so what they are — limited range, BT.709, as `_normalise` makes
# every clip — is said on both ends of the pipe, and the encoder writes it into the file.
RAW = ("-f", "rawvideo", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709")

# Тишина под кусок, которому нечем звучать, — фотография и клип, снятый без звука. Живой звук
# ролика склеивается из дорожек кусков, а склеить что-то с ничем нельзя. 44,1 кГц стерео — то же,
# во что приводится и живой звук клипов: так стыкуются все куски.
SAMPLE_RATE = 44100
SILENCE = f"anullsrc=r={SAMPLE_RATE}:cl=stereo"

# `volume` weighs a whole frame of samples by one number, and a decoder's frame is a thousand-odd of
# them: a half-second turn from the music to a clip's own sound would go in twenty steps, and a step
# is heard. Frames this short make it three hundred and forty, too small to hear.
RAMP_SAMPLES = 64


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
    shortest = min((ffmpeg.duration(track) or TRACK_FADE_SECONDS * 2) for track in tracks)
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
    ffmpeg.run(
        *inputs,
        "-filter_complex",
        ";".join(chain),
        "-map",
        f"[{label}]",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out),
    )
    return out


def length_of(pieces):
    """How long these pieces run once crossfaded — each join eats half a second of the total."""
    return sum(piece.seconds for piece in pieces) - CROSSFADE_SECONDS * (len(pieces) - 1)


def assemble(pieces, orientation, track, workdir, out, on_clip=None, max_seconds=None):
    """The chosen pieces as one video at `out`, and the normalised clips it was made of.

    `track` is a file — the montage's whole soundtrack, however many tracks of whose it was built
    (`one_track`). Which of them came out of the library and which the organizer uploaded was
    settled before this module was called, and nothing here can tell them apart. `None` is the
    order that asked for no music at all, and then the video keeps the clips' own sound, every one
    of them whole (doc-03, «Без музыки ролик идёт с живым звуком самих видео»). Under a track every
    piece is heard as it says (`live`): the music, its own sound or the two mixed. A piece with no
    sound of its own — a photograph, a clip shot silent — is under the music whatever it says:
    what it would be heard with instead is silence.

    The clips come back because they are the only place the lengths are true: `_normalise` is asked
    for a length and the file that comes out of it is measured rather than believed, and whoever
    cuts a cover off the montage needs those measured numbers to find its opening piece in it.

    `on_clip` is called with how many pieces are encoded so far, every time one more is — the one
    thing this module tells anybody while it works, and the montage takes minutes: «склеено 12 из
    61» on the live screen is that number and nothing else. Nothing is asked of it and nothing is
    read back, so a caller with nothing to show simply does not pass one.

    `max_seconds` is the longest the file may run. The pieces are already planned inside it
    (`selection.cap`), but every clip is encoded to the nearest frame of what it was asked for, and
    over a few hundred pieces those roundings add up to a fraction of a second either way — so a
    plan that ends on the ceiling can come out a few frames past it, and those frames are cut off.
    """
    clips = []
    for index, piece in enumerate(pieces):
        clip = workdir / f"clip-{index}.mp4"
        its_own = _normalise(piece, orientation, clip)
        if on_clip:
            on_clip(index + 1)
        # Measured, not the length that was asked for — and the picture, not the container. Asked
        # for 2.52 seconds, a clip is encoded to the nearest frame of it, and one that the file
        # runs out under is shorter still; the container is the longest of its streams, and the
        # sound can run a little past the picture (`_normalise`). `_assemble` steps the crossfade
        # offsets along by these numbers, a part asks its clip for this many frames, and a clip
        # with one frame fewer than it was booked at leaves every part after it a frame behind its
        # sound.
        clips.append(
            dataclasses.replace(
                piece,
                path=clip,
                seconds=ffmpeg.duration(clip, "v:0") or piece.seconds,
                live=piece.live if its_own else 0.0,
            )
        )
    _assemble(clips, track, CANVASES[orientation], workdir, out, max_seconds)
    return clips


def _normalise(piece, orientation, out):
    """One piece onto this montage's canvas: 1080p the way it was ordered, 30 fps, cropped, limited
    range with BT.709 colour, and with sound — its own, or silence of the same shape. Answers
    whether the sound is its own.

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

    The colour is settled by the same scale, because a photograph decodes full range with the JPEG
    matrix and a phone's clip limited with BT.709. Crossfaded or encoded side by side as they come,
    one of the two plays back too flat or too contrasty and a little off in hue — in the old single
    chain it was whichever did not open the montage. Limited BT.709 is what a video player assumes.

    Every clip comes out with sound, whatever was ordered: `_assemble` joins the clips' sound into
    one tape with the same crossfade as their picture, and there is nothing to crossfade a missing
    stream with. A clip keeps its own track and everything with nothing to say (a photograph, a
    clip shot silent) gets silence. How much of that tape is heard over the music is settled over
    the whole of it, piece by piece, and not here.

    Its own track is the **first** one of the file and only that one — `0:a:0`, never `0:a`,
    which is every audio stream the file happens to carry. An iPhone 16 Pro writes two into a MOV:
    what the microphones heard, and a second, spatial one whose codec ffmpeg has no decoder for.
    Handed both, ffmpeg refuses the whole call («no decoder found for: none») and there is no clip
    at all, so one such video off one guest's phone took down the entire montage (TASK-460,
    поймано на живом заказе 2026-09-27). `ffmpeg.has_sound` cannot stand in for this: it asks
    whether there is an `a:0` at all, and here there is. Every `-map` in this module names one
    stream for the same reason — a class of streams is whatever the file turns out to hold.

    Both streams are cut where they come in, not where they go out: the clip's by the `-t` of its
    own input, the silence by one of its own. Cut on the way out (`-shortest`, an output `-t`), the
    picture loses its last frame whenever `fps` has to make it up at the end of a 24-frame clip,
    because the sound has ended first. So the picture comes out with every frame it was asked for,
    and the sound may run a few hundredths of a second past it or stop short of it — which is why
    the clip is measured by its picture (`assemble`) and every part's sound is cut and padded to
    the part's frames (`_part`).

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
        f"scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2:"
        f"out_range=tv:out_color_matrix=bt709,crop={width}:{height},setsar=1"
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
    its_own = piece.kind == "video" and ffmpeg.has_sound(piece.path)
    added = () if its_own else ("-t", f"{piece.seconds}", "-f", "lavfi", "-i", SILENCE)
    ffmpeg.run(
        *opened,
        *added,
        "-vf",
        canvas,
        *cut,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0" if its_own else "1:a:0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        str(out),
    )
    return its_own


def _assemble(clips, track, canvas, workdir, out, max_seconds=None):
    """The normalised clips into one file: crossfaded end to end, and the sound with them — the
    clips' own sound joined the same way into one tape and, when there is a track, the two tapes
    mixed piece by piece as each piece says (`_live_share`).

    One encoder runs from the first frame to the last, as it always did, but no ffmpeg opens more
    than two clips. Each input an ffmpeg is handed stays open and decoding to the end at 80–100 МиБ
    apiece, so one call over the whole montage grew with its length — a minute of video took 3 ГиБ,
    and ten minutes would fit no pod (running.md, «Образ»). Here every clip is its own short ffmpeg
    that dissolves the one before into it and writes the frames, raw, into the encoder's pipe; the
    encoder cannot tell where one of them ended and the next began. The sound is laid over the
    finished picture in one last pass that copies it.
    """
    # Where every clip begins in the finished video: each join starts half a second before the
    # chain so far ends, which is `length_of`. A clip's own stretch runs from there to where the
    # next one begins, on whole frames — every clip is placed against the video's first frame and
    # not against the clip before, so no rounding carries from one join to the next.
    starts = [0.0]
    for clip in clips[:-1]:
        starts.append(starts[-1] + clip.seconds - CROSSFADE_SECONDS)
    end = round((starts[-1] + clips[-1].seconds) * FPS)
    if max_seconds is not None:
        # The picture is cut at the ceiling and the sound fades out on it rather than past it. A
        # hair over, because in floating point 4.1 seconds is 122.999… frames.
        end = min(end, math.floor(max_seconds * FPS + 1e-6))
    cuts = [min(round(start * FPS), end) for start in starts] + [end]
    # A clip shorter than a dissolve has no stretch of its own — the next one starts dissolving in
    # before it is fully there — so it is left no frames rather than handed a negative count, which
    # ffmpeg takes for «no limit».
    cuts = list(itertools.accumulate(cuts, max))

    picture, sounds = workdir / "picture.mp4", []
    width, height = canvas
    with ffmpeg.encoding(
        (*RAW, "-video_size", f"{width}x{height}", "-framerate", str(FPS)),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(picture),
    ) as pipe:
        for index, clip in enumerate(clips):
            begins, ends = cuts[index], cuts[index + 1]
            if begins == ends:
                continue  # past the ceiling, or no stretch of its own
            sound = workdir / f"sound-{index}.wav"
            before = clips[index - 1] if index else None
            # The clip before goes on from the frame its own part stopped at, so neither its
            # picture nor its sound skips where one part hands over to the next.
            skip = (begins - cuts[index - 1]) / FPS if index else 0.0
            _part(before, skip, clip, ends - begins, pipe, sound)
            sounds.append(sound)

    total = end / FPS
    # Живой звук уже склеен по кускам тем же перехлёстом, что и картинка, — остаётся сложить их
    # встык. Список читает concat и больше ничего; имена в нём — рядом с ним самим.
    # `-protocol_whitelist` is an *input* option and the one `ffmpeg.run` puts at the front guards
    # the picture alone, so every sound gets its own.
    listing = workdir / "sounds.txt"
    listing.write_text("".join(f"file '{sound.name}'\n" for sound in sounds))
    heard = ["-protocol_whitelist", "file", "-f", "concat", "-i", str(listing)]
    if track is None:
        # Края ролика — тем же полусекундным фейдом: звук, оборванный на полуслове, щёлкает.
        mixed, fade = "[1:a]", CROSSFADE_SECONDS
    else:
        # A track shorter than the montage is played again from the top (`-stream_loop -1`), a
        # longer one is cut off with the picture, because `-shortest` is down below. Either way the
        # music never has to be told how long to play: the montage is cut to the material, not to
        # the track.
        heard += ["-protocol_whitelist", "file", "-stream_loop", "-1", "-i", str(track)]
        # The two tapes against each other: what the clips' own sound is given, the music is
        # denied, so a piece mixed half and half is as loud as either alone. Summed, not averaged
        # (`normalize=0`): `amix` would halve both again.
        share = _live_share(clips, cuts)
        mixed = (
            f"[1:a]asetnsamples={RAMP_SAMPLES},volume=eval=frame:volume='{share}'[own];"
            f"[2:a]asetnsamples={RAMP_SAMPLES},volume=eval=frame:volume='1-({share})'[music];"
            "[own][music]amix=inputs=2:duration=first:normalize=0,"
        )
        fade = MUSIC_FADE_SECONDS
    fade = min(fade, total / 3)
    ffmpeg.run(
        "-i",
        str(picture),
        *heard,
        "-filter_complex",
        f"{mixed}afade=t=in:st=0:d={fade:.3f},afade=t=out:st={total - fade:.3f}:d={fade:.3f}[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(out),
    )


def _live_share(clips, cuts):
    """How much of the clips' own sound is heard at `t` of the finished video, as an ffmpeg
    expression: each clip's `live` while it is on screen, and from one to the next over the dissolve
    that joins their pictures — the same half second from the same frame, `cuts[index]`, where
    `_part` opens it. So the sound turns over with the picture: a hard switch between the music and
    somebody talking clicks, and one that ran ahead of the picture or behind it would be heard as
    the wrong piece's sound.

    Only a join that changes something is written into it: the plain montage, every piece under the
    music, is one constant nought. The ramps are summed rather than switched between, so two that
    overlap — around a clip shorter than a dissolve — still add up to where the next clip stands.
    """
    # ponytail: a term per change, all of them evaluated every 64 samples. Ten minutes that change
    # sound at each of 600 joins cost ~20 s of CPU and ~50 KB of argument; a search by `t` over the
    # changes would make it log(n), if that ever shows up.
    terms = [f"{clips[0].live:.3f}"]
    for index in range(1, len(clips)):
        step = clips[index].live - clips[index - 1].live
        if step:
            terms.append(f"{step:.3f}*clip((t-{cuts[index] / FPS:.6f})/{CROSSFADE_SECONDS},0,1)")
    return _summed(terms)


def _summed(terms):
    """`terms` added up as an ffmpeg expression, in halves: its parser nests one level deeper for
    every `+` in a row and refuses past a hundred, and a long montage changes its sound more often
    than that. Halved, a thousand terms are ten levels deep."""
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return f"({_summed(terms[:middle])})+({_summed(terms[middle:])})"


def _part(before, skip, clip, frames, pipe, sound):
    """`frames` frames of the finished video from where `before` begins to dissolve into `clip`,
    written raw into the encoder's `pipe` — and the same stretch of the clips' own sound into the
    `sound` file.

    `before` is the clip whose last half second the stretch opens on, from `skip` seconds into it,
    and None for the opening clip, which nothing dissolves into. The dissolve starts on the part's
    first frame, so both inputs are read from the start and neither waits on the other with frames
    piling up. The sound is PCM, so the parts join sample for sample, and it is padded out to the
    picture's length, so a clip whose sound runs short cannot pull everything after it out of step.
    """
    inputs, graph = ["-protocol_whitelist", "file", "-i", str(clip.path)], []
    video, joined = "0:v:0", "[0:a]"
    if before is not None:
        # No more of it than the dissolve: acrossfade plays whatever precedes a track's last half
        # second ahead of the fade, and that would put `clip`'s sound behind its picture.
        tail = ("-ss", f"{skip:.6f}", "-t", str(CROSSFADE_SECONDS), "-protocol_whitelist", "file")
        inputs = [*tail, "-i", str(before.path), *inputs]
        graph.append(f"[0:v][1:v]xfade=transition=fade:duration={CROSSFADE_SECONDS}:offset=0[v]")
        # acrossfade накладывает конец одной дорожки на начало следующей ровно так, как xfade
        # накладывает кадры, — поэтому звук и изображение остаются вместе на каждой склейке. Хвост
        # короче перехлёста он не склеивает, а глушит всю часть, — поэтому хвост добит тишиной.
        graph.append(f"[0:a]apad=whole_dur={CROSSFADE_SECONDS}[tail]")
        video, joined = "[v]", f"[tail][1:a]acrossfade=d={CROSSFADE_SECONDS},"
    graph.append(f"{joined}apad[a]")
    ffmpeg.run(
        *inputs,
        "-filter_complex",
        ";".join(graph),
        "-map",
        video,
        "-frames:v",
        str(frames),
        *RAW,
        "-",
        "-map",
        "[a]",
        "-c:a",
        "pcm_s16le",
        "-t",
        f"{frames / FPS:.6f}",
        str(sound),
        into=pipe,
    )
