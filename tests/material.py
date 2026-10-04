"""Материал для тестов: файлы, которые они рисуют себе сами, и то немногое, что нарисовать нельзя.

Почти всё здесь делает ffmpeg на месте — так в репозитории не лежит ничего двоичного. Исключение —
фотографии с EXIF в `fixtures/`: ffmpeg его не пишет, а HEIC не пишет вовсе.
"""

import json
import pathlib
import struct
import subprocess

from PIL import ExifTags, Image, ImageDraw, TiffImagePlugin

FIXTURES = pathlib.Path(__file__).with_name("fixtures")

# Eight bytes of `exif-photo.heic` overwritten so that ffmpeg still cuts a thumbnail out of it and
# libheif segfaults inside `Image.open` — found by flipping bytes, libheif 1.23. Should a newer
# libheif stop falling over, what the tests below hold to stays true: the worker outlives the file.
CRASHING_HEIC = {153: 163, 179: 220, 228: 183, 450: 174, 484: 87, 606: 93, 616: 114, 619: 181}


def draw(*args):
    subprocess.run(("ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args), check=True)


def probe(path):
    """`{"streams": [...], "format": {...}}`, as ffprobe sees the file."""
    out = subprocess.run(
        ("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def jpeg(path, size="1200x800"):
    draw("-f", "lavfi", "-i", f"testsrc=size={size}:duration=1", "-frames:v", "1", str(path))
    return path


def video(path):
    """Three seconds of picture and sound, taller than 720 so the preview has to scale down."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1280x1440:rate=24:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    )
    return path


def sideways_photo(path, size=(1200, 800), orientation=6):
    """A photograph off a phone held sideways: the sensor's own frame, and the EXIF note saying
    which way up it goes. Drawn by Pillow — `ffmpeg` writes no EXIF at all."""
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    Image.new("RGB", size, "gray").save(path, exif=exif)
    return path


def upright_video(path):
    """A clip the phone was held upright for: stored on its side, with a display matrix beside it
    saying to stand it up. Written in two passes because that matrix survives only a copy —
    re-encoding turns the picture itself instead and drops it, autorotate doing its job."""
    stored = video(path.with_name("stored.mp4"))
    draw("-display_rotation", "90", "-i", str(stored), "-c", "copy", str(path))
    return path


def sound(path):
    """A second of tone and no picture at all: a file with nothing whatever to measure."""
    draw("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(path))
    return path


def phone_video(path, place="+55.5180+037.5300+158.000/"):
    """A clip with the tags a phone puts beside the picture: which camera took it, and where.
    `use_metadata_tags` so they are written under those names and not dropped as unknown."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=24:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-metadata",
        "com.apple.quicktime.model=iPhone 15 Pro",
        "-metadata",
        f"com.apple.quicktime.location.ISO6709={place}",
        "-movflags",
        "use_metadata_tags",
        str(path),
    )
    return path


def marked_photo(path, orientation):
    """A photograph with something to tell its corners apart, and the EXIF note that says how the
    phone was held: what the sensor wrote is 200×120, and the note turns it (or flips it) as the
    person saw it. Drawn by Pillow — `ffmpeg` writes no EXIF."""
    picture = Image.new("RGB", (200, 120), (128, 128, 128))
    paint = ImageDraw.Draw(picture)
    paint.rectangle([0, 0, 59, 39], fill=(255, 0, 0))
    paint.rectangle([160, 0, 199, 19], fill=(0, 255, 0))
    paint.rectangle([0, 90, 29, 119], fill=(0, 0, 255))
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    picture.save(path, exif=exif, quality=95)
    return path


def crashing_heic(path):
    """`exif-photo.heic` damaged so that opening it with libheif crashes the process."""
    data = bytearray((FIXTURES / "exif-photo.heic").read_bytes())
    for offset, value in CRASHING_HEIC.items():
        data[offset] = value
    path.write_bytes(data)
    return path


def photo_with_odd_numbers(path):
    """What some cameras write when they have nothing to say: an f-number over zero, a focal length
    of nothing, and a latitude of two hundred degrees."""

    def rational(number):
        return TiffImagePlugin.IFDRational(number, 1)

    exif = Image.Exif()
    optics = exif.get_ifd(ExifTags.IFD.Exif)
    optics[ExifTags.Base.FNumber] = TiffImagePlugin.IFDRational(1, 0)
    optics[ExifTags.Base.FocalLength] = rational(0)
    place = exif.get_ifd(ExifTags.IFD.GPSInfo)
    place[ExifTags.GPS.GPSLatitudeRef] = "N"
    place[ExifTags.GPS.GPSLatitude] = (rational(200), rational(0), rational(0))
    place[ExifTags.GPS.GPSLongitudeRef] = "E"
    place[ExifTags.GPS.GPSLongitude] = (rational(37), rational(0), rational(0))
    Image.new("RGB", (64, 48), "gray").save(path, exif=exif)
    return path


def photo_with_settings(path, *, iso=None, exposure_ev=None, shutter=None):
    """A photograph with the shooting settings the strip on the sheet is built of — the sensitivity,
    the exposure compensation in stops, and the shutter speed as the fraction a camera writes it.

    Drawn by Pillow, like every other EXIF here: `ffmpeg` writes no EXIF at all. What is left out
    is left out of the file too, which is what makes this one material for both halves of the rule —
    a photograph with the settings gets them, one without keeps the columns empty.
    """

    def fraction(value):
        number, denominator = value
        return TiffImagePlugin.IFDRational(number, denominator)

    exif = Image.Exif()
    exif[ExifTags.Base.Model] = "iPhone 15 Pro"
    optics = exif.get_ifd(ExifTags.IFD.Exif)
    optics[ExifTags.Base.FocalLength] = TiffImagePlugin.IFDRational(24, 1)
    optics[ExifTags.Base.FNumber] = TiffImagePlugin.IFDRational(18, 10)
    if iso is not None:
        optics[ExifTags.Base.ISOSpeedRatings] = iso
    if exposure_ev is not None:
        optics[ExifTags.Base.ExposureBiasValue] = fraction(exposure_ev)
    if shutter is not None:
        optics[ExifTags.Base.ExposureTime] = fraction(shutter)
    Image.new("RGB", (64, 48), "gray").save(path, exif=exif)
    return path


def gigantic_photo(path):
    """`exif-photo.jpg` whose header claims 30000×30000 — 900 megapixels in 900 bytes. Pillow
    refuses that as a decompression bomb, and ffprobe reports it as a size of nothing."""
    data = bytearray((FIXTURES / "exif-photo.jpg").read_bytes())
    sof = data.index(b"\xff\xc0")  # start of frame: marker, length, precision, height, width
    data[sof + 5 : sof + 9] = struct.pack(">HH", 30000, 30000)
    path.write_bytes(data)
    return path


def long_clip(path, seconds=12):
    """A clip with room in it. The happy path's own clip runs three seconds, which the default cut
    already takes whole — there is nothing left in it to hold longer, and a video ordered longer
    than its material is exactly what this material cannot show."""
    draw(
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=1280x720:rate=24:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    )
    return path


def click_track(path, gap=1.6):
    """Thirty seconds of metronome: one click every 1.6 seconds. The tempo is known because it was
    put there, and it is deliberately slow — under a track like this the pieces are pulled visibly
    off their own lengths."""
    draw(
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc='0.8*sin(2*PI*1200*t)*exp(-40*mod(t,{gap}))':d=30",
        str(path),
    )
    return path


def tone(path, seconds=3.0, frequency=880):
    """Секунды слышимого звука — трек, который в собранной дорожке ни с чем не спутаешь."""
    draw("-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}", str(path))
    return path


def hush(path, seconds=3.0):
    """Столько же тишины — второй трек, слышный ровно наоборот: по нему и видно порядок."""
    draw("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds), str(path))
    return path


def mute_clip(path, seconds=3.0):
    """Клип, снятый с выключенным микрофоном: картинка есть, звуковой дорожки нет вовсе."""
    draw(
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=640x480:rate=24:duration={seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    return path


def spatial_sound_clip(path, seconds=3.0):
    """Клип с iPhone 16 Pro: обычная звуковая дорожка и вторая, пространственная, чей кодек ffmpeg
    не знает. Такие MOV приходят с живых заказов, и `-map 0:a` берёт из них обе дорожки — на второй
    ffmpeg отказывается от всего вызова («no decoder found for: none»), и выходного файла не
    остаётся вовсе (TASK-460).

    Нарисовать вторую дорожку нечем: apac ffmpeg не кодирует и с чужим четырёхбуквенным кодом
    muxer'у дорожку не отдаёт («Tag apac incompatible with output codec id»). Поэтому она пишется
    как PCM — у него в MOV нет `esds`, из которого демуксер иначе вычитал бы кодек обратно поверх
    кода, — а её код в `stsd` переписывается здесь на apac. После этого ffmpeg видит ровно то же,
    что видит в файле с телефона: дорожка есть, кодека к ней нет.
    """
    drawn = path.with_name(f"drawn-{path.name}")
    draw(
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=640x480:rate=24:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1760:duration={seconds}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a:0",
        "aac",
        # Вторая дорожка звучать не будет ни при каком исходе, и чем она мельче, тем мельче файл.
        "-c:a:1",
        "pcm_s16be",
        "-ar:a:1",
        "8000",
        "-ac:a:1",
        "1",
        "-shortest",
        str(drawn),
    )
    data = bytearray(drawn.read_bytes())
    # Первое вхождение — сама запись о формате дорожки в `stsd`; второй дорожки, кроме PCM, в файле
    # нет, и больше `twos` взяться в нём негде.
    twos = data.index(b"twos")
    data[twos : twos + 4] = b"apac"
    path.write_bytes(data)
    return path


def dark_opening_clip(path):
    """A clip that opens black for its first 1.5s, then three and a half seconds of a sharp,
    properly exposed pattern — the shape of a real phone clip that takes a moment to find its
    exposure. `_best_moment` finds its best frame only after the black stretch, but the piece is cut
    *centred* on that moment and pulled back to fit the file, so the cut can still start inside the
    black part. The happy-path fixture (`material`) always puts a photo first, so its montage never
    opens on a video and this shape of bug never gets a chance to show up."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "color=c=black:size=1280x720:rate=24:duration=1.5",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1280x720:rate=24:duration=3.5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=5",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    )
    return path


def unopenable(path):
    """Bytes in the bucket that ffmpeg will not open — a truncated upload, a file the phone
    mislabelled. Not a playlist: that is the security case and `test_media.py` has it."""
    path.write_bytes(b"this is not a photo and never was")
    return path


def crushed(path):
    """A photo with the same edges `jpeg()` has — sharp, easily past `MIN_SHARPNESS` — pushed
    almost to black. ffmpeg opens it without complaint, unlike `unopenable`; only the exposure
    check has anything to say about it, which is the point: a flat black frame would be rejected on
    blur alone and prove nothing about the histogram half of `_best_moment`."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x480:duration=1",
        "-vf",
        "eq=brightness=-0.85",
        "-frames:v",
        "1",
        str(path),
    )
    return path


def lopsided(path):
    """Two seconds of picture under three and a half of sound, so the container calls it 3.5 —
    `format=duration` is the longest stream, and `-t` cuts against the picture."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x480:rate=30:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    )
    return path
