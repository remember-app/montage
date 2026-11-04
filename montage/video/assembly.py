import dataclasses
import itertools
import math

from montage import ffmpeg

CANVASES = {"landscape": (1920, 1080), "portrait": (1080, 1920)}
DEFAULT_ORIENTATION = "landscape"
FPS = 30

CROSSFADE_SECONDS = 0.5
MUSIC_FADE_SECONDS = 1.5
TRACK_FADE_SECONDS = 1.0

RAW = ("-f", "rawvideo", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709")

SAMPLE_RATE = 44100
SILENCE = f"anullsrc=r={SAMPLE_RATE}:cl=stereo"

RAMP_SAMPLES = 64


def one_track(tracks, workdir):
    if len(tracks) < 2:
        return tracks[0] if tracks else None
    shortest = min((ffmpeg.duration(track) or TRACK_FADE_SECONDS * 2) for track in tracks)
    fade = max(min(TRACK_FADE_SECONDS, shortest / 2), 0.1)

    inputs, chain = [], []
    for index, track in enumerate(tracks):
        inputs += ["-protocol_whitelist", "file", "-i", str(track)]
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
    return sum(piece.seconds for piece in pieces) - CROSSFADE_SECONDS * (len(pieces) - 1)


def assemble(
    pieces, orientation, track, workdir, out, on_clip=None, on_join=None, max_seconds=None
):
    clips = []
    for index, piece in enumerate(pieces):
        clip = workdir / f"clip-{index}.mp4"
        its_own = _normalise(piece, orientation, clip)
        if on_clip:
            on_clip(index + 1)
        clips.append(
            dataclasses.replace(
                piece,
                path=clip,
                seconds=ffmpeg.duration(clip, "v:0") or piece.seconds,
                live=piece.live if its_own else 0.0,
            )
        )
    if on_join:
        on_join()
    _assemble(clips, track, CANVASES[orientation], workdir, out, max_seconds)
    return clips


def _normalise(piece, orientation, out):
    width, height = CANVASES[orientation]
    fitted = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2:"
        f"out_range=tv:out_color_matrix=bt709,crop={width}:{height},setsar=1"
    )
    if piece.kind == "photo":
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
    starts = [0.0]
    for clip in clips[:-1]:
        starts.append(starts[-1] + clip.seconds - CROSSFADE_SECONDS)
    end = round((starts[-1] + clips[-1].seconds) * FPS)
    if max_seconds is not None:
        end = min(end, math.floor(max_seconds * FPS + 1e-6))
    cuts = [min(round(start * FPS), end) for start in starts] + [end]
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
                continue
            sound = workdir / f"sound-{index}.wav"
            before = clips[index - 1] if index else None
            skip = (begins - cuts[index - 1]) / FPS if index else 0.0
            _part(before, skip, clip, ends - begins, pipe, sound)
            sounds.append(sound)

    total = end / FPS
    listing = workdir / "sounds.txt"
    listing.write_text("".join(f"file '{sound.name}'\n" for sound in sounds))
    heard = ["-protocol_whitelist", "file", "-f", "concat", "-i", str(listing)]
    if track is None:
        mixed, fade = "[1:a]", CROSSFADE_SECONDS
    else:
        heard += ["-protocol_whitelist", "file", "-stream_loop", "-1", "-i", str(track)]
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
    terms = [f"{clips[0].live:.3f}"]
    for index in range(1, len(clips)):
        step = clips[index].live - clips[index - 1].live
        if step:
            terms.append(f"{step:.3f}*clip((t-{cuts[index] / FPS:.6f})/{CROSSFADE_SECONDS},0,1)")
    return _summed(terms)


def _summed(terms):
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return f"({_summed(terms[:middle])})+({_summed(terms[middle:])})"


def _part(before, skip, clip, frames, pipe, sound):
    inputs, graph = ["-protocol_whitelist", "file", "-i", str(clip.path)], []
    video, joined = "0:v:0", "[0:a]"
    if before is not None:
        tail = ("-ss", f"{skip:.6f}", "-t", str(CROSSFADE_SECONDS), "-protocol_whitelist", "file")
        inputs = [*tail, "-i", str(before.path), *inputs]
        graph.append(f"[0:v][1:v]xfade=transition=fade:duration={CROSSFADE_SECONDS}:offset=0[v]")
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
