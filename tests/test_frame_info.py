"""Размер кадра, камера и место съёмки — см. `docs/jobs/frame-info.md`.

Всё живое, и материал с EXIF — из `fixtures/`: ffmpeg его не пишет, а HEIC не пишет вовсе.
"""

import logging

import pytest
from PIL import ExifTags, Image, TiffImagePlugin

from montage import photo, storage, worker
from montage.media import frame_info

from material import (
    FIXTURES,
    crashing_heic,
    gigantic_photo,
    jpeg,
    phone_video,
    photo_with_odd_numbers,
    photo_with_settings,
    sideways_photo,
    sound,
    upright_video,
    video,
)
from rows import add_item, only_jobs_for, read_frame_info, read_settings, read_size

MOSCOW_SOUTH = (pytest.approx(55.518, abs=0.0005), pytest.approx(37.53, abs=0.0005))


@pytest.mark.parametrize(
    "kind, draw, name, seen_as",
    [
        ("photo", jpeg, "original.jpg", (1200, 800)),
        # The sideways phone: 1200×800 on the sensor, and the EXIF note stands it up.
        ("photo", sideways_photo, "sideways.jpg", (800, 1200)),
        ("video", video, "original.mp4", (1280, 1440)),
        # And the clip shot upright: stored 1280×1440, played the other way round.
        ("video", upright_video, "upright.mp4", (1440, 1280)),
    ],
)
def test_a_frame_is_measured_the_way_it_is_seen(
    conn, s3, memory, tmp_path, kind, draw, name, seen_as
):
    """«Размер 4032 × 3024» on the sheet: the pixels a person sees, not the ones the sensor wrote.
    A file turned by its metadata is the ordinary case — every photograph taken with the phone on
    its side, and every clip shot holding it upright."""
    item_id, key = add_item(conn, memory, kind)
    s3.upload_file(str(draw(tmp_path / name)), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == seen_as


def test_a_file_with_no_picture_in_it_is_left_unmeasured(conn, s3, memory, tmp_path):
    """Nothing to measure is not a failure: the two columns stay empty, the job is done, and the
    sheet leaves the line out — `kind` being only the phone's word for what it sent."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(sound(tmp_path / "original.m4a")), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (None, None)
    (status,) = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


@pytest.mark.parametrize("fixture", ["exif-photo.jpg", "exif-photo.heic"])
def test_a_photograph_brings_its_camera_and_its_place(conn, s3, memory, fixture):
    """HEIC beside JPEG on purpose: HEIC is what an iPhone actually uploads."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / fixture), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    camera, focal_length, aperture, latitude, longitude = read_frame_info(conn, item_id)
    # «iPhone 15 Pro · 24 мм · f/1.8» on the layout — in five columns, because those units are
    # words and the app puts them in the reader's own language.
    assert (camera, focal_length, aperture) == ("iPhone 15 Pro", 24, 1.8)
    assert (latitude, longitude) == MOSCOW_SOUTH


@pytest.mark.parametrize(
    "iso, exposure_ev, shutter, seen_as",
    [
        (400, (0, 1), (1, 60), (400, 0, "1/60")),  # what a phone writes on an ordinary day
        (12800, (-13, 10), (1, 125), (12800, -1.3, "1/125")),  # a dim room, a hand on the dial
        (100, (0, 1), (1, 1), (100, 0, "1")),  # bright sun, a whole second, no compensation
    ],
    ids=["ordinary", "dim-room", "sun"],
)
def test_a_photograph_brings_its_shooting_settings(
    conn, s3, memory, tmp_path, iso, exposure_ev, shutter, seen_as
):
    """The rest of the strip on the layout — ISO · мм · ev · ƒ · выдержка — off the same file the
    camera line came from, written by the same one update that writes it."""
    item_id, key = add_item(conn, memory, "photo")
    settings = photo_with_settings(
        tmp_path / "original.jpg", iso=iso, exposure_ev=exposure_ev, shutter=shutter
    )
    s3.upload_file(str(settings), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    # The shutter as the fraction a phone reads back out loud: «1/60», never the 0.0166… seconds
    # it equals, out of which the denominator would have to be guessed at.
    assert read_settings(conn, item_id) == seen_as


def test_a_photograph_without_the_settings_leaves_them_empty(conn, s3, memory, tmp_path):
    """The other half of the same rule, and the one a zero is easy to lose: a camera that wrote
    none of it leaves three nulls — not three zeroes, which would draw «ISO 0 · ev 0 · 1/0 с» on the
    sheet. Most material is this one: a screenshot, a frame re-saved by an editor."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(
        str(photo_with_settings(tmp_path / "original.jpg")), storage.client().bucket, key
    )
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_settings(conn, item_id) == (None, None, None)


def test_an_exposure_compensation_of_zero_is_a_number_and_not_an_absence(
    conn, s3, memory, tmp_path
):
    """`exposure_ev` is signed and nullable without a default precisely so that «сняли без
    компенсации» and «воркер не смотрел» stay two different things. This file is the first."""
    item_id, key = add_item(conn, memory, "photo")
    settings = photo_with_settings(
        tmp_path / "original.jpg", iso=64, exposure_ev=(0, 1), shutter=(1, 500)
    )
    s3.upload_file(str(settings), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    iso, exposure_ev, shutter = read_settings(conn, item_id)
    assert exposure_ev == 0, "a zero compensation is a reading, not a blank"
    assert (iso, shutter) == (64, "1/500")


def test_a_clip_keeps_the_settings_empty(conn, s3, memory, tmp_path):
    """A container says nothing about the optics — and says nothing about the settings beside them
    either, which is why `focal_length_mm` and `aperture` are empty at a clip too."""
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(phone_video(tmp_path / "original.mp4")), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_settings(conn, item_id) == (None, None, None)


def test_a_shutter_written_as_it_is_reduced_and_a_whole_second_is_whole(tmp_path):
    """2/120 is the same 1/60 written wider, and 1/1 is one second rather than a fraction of it.
    The sheet promises «1/60 с», so the text is the answer, in the fewest words that say it."""
    from fractions import Fraction

    from montage import photo

    for written, as_text in [((2, 120), "1/60"), ((1, 1), "1"), ((1, 2), "1/2"), ((1, 60), "1/60")]:
        path = photo_with_settings(tmp_path / f"{written[0]}-{written[1]}.jpg", shutter=written)

        found = photo.read(path)

        assert found["shutter"] == as_text
        assert Fraction(found["shutter"]) == Fraction(*written)


def test_a_clip_brings_its_camera_and_its_place_off_the_container(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(phone_video(tmp_path / "original.mp4")), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    camera, focal_length, aperture, latitude, longitude = read_frame_info(conn, item_id)
    assert camera == "iPhone 15 Pro"
    assert (latitude, longitude) == MOSCOW_SOUTH
    assert (focal_length, aperture) == (None, None), "a container says nothing about the optics"


@pytest.mark.parametrize("draw, name", [(jpeg, "original.jpg"), (video, "original.mp4")])
def test_a_file_that_carries_nothing_leaves_the_camera_empty(
    conn, s3, memory, tmp_path, draw, name
):
    """Which is most material: a screenshot, a frame re-saved by an editor, anything that came
    through a messenger. Reading it is not a failure and the job says so — nor is a file that is
    no photograph at all, `kind` being only the phone's word for what it sent."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(draw(tmp_path / name)), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_frame_info(conn, item_id) == (None, None, None, None, None)
    (status,) = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


def test_a_finished_preview_asks_for_the_camera_to_be_read(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert only_jobs_for(conn, item_id, "media_frame_info") == 1


def test_the_log_says_what_was_read_and_not_where_it_was(conn, s3, memory, caplog):
    """The place is where somebody was. The sheet shows it to the members of one memory; the log
    is read by whoever runs the worker and kept by whatever collects it."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / "exif-photo.jpg"), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    with caplog.at_level(logging.INFO):
        assert worker.run_once(conn) is True

    said = caplog.text
    assert item_id in said and "latitude" in said and "longitude" in said
    assert "55.5" not in said and "37.5" not in said


@pytest.mark.parametrize(
    "place",
    [
        # degrees and minutes, which reads as decimal degrees far off the globe
        "+5531.08+03731.80/",
        "+" + "9" * 400 + "+037.5300/",  # a run of digits that is infinity as a float
        # what a camera with no satellite fix writes; the Gulf of Guinea, not a place
        "+00.0000+000.0000/",
    ],
    ids=["degrees-and-minutes", "infinity", "no-fix"],
)
def test_a_place_no_globe_can_hold_is_no_place(conn, s3, memory, tmp_path, place):
    """A clip's place tag comes off somebody's phone. What is not a place on the globe — or is the
    point (0, 0) that a camera without a fix writes in its stead — is left out like any other thing
    the file does not say, and the camera beside it is still read."""
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(phone_video(tmp_path / "original.mp4", place)), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    camera, _, _, latitude, longitude = read_frame_info(conn, item_id)
    assert (camera, latitude, longitude) == ("iPhone 15 Pro", None, None)


def test_numbers_a_camera_wrote_that_nothing_can_use_are_left_out(conn, s3, memory, tmp_path):
    """An f-number over zero is NaN, and NaN goes into a double column and out to the sheet as if
    it were a reading. The same file's latitude is two hundred degrees."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(
        str(photo_with_odd_numbers(tmp_path / "original.jpg")), storage.client().bucket, key
    )
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_frame_info(conn, item_id) == (None, None, None, None, None)
    assert read_size(conn, item_id) == (64, 48)
    (status,) = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


def test_south_and_west_are_minus(tmp_path):
    """The hemisphere is a tag of its own next to the degrees, and the one place a sign is lost —
    the fixtures are all north and east."""

    def degrees(whole, minutes):
        return tuple(TiffImagePlugin.IFDRational(number, 1) for number in (whole, minutes, 0))

    exif = Image.Exif()
    place = exif.get_ifd(ExifTags.IFD.GPSInfo)
    place[ExifTags.GPS.GPSLatitudeRef], place[ExifTags.GPS.GPSLatitude] = "S", degrees(33, 52)
    place[ExifTags.GPS.GPSLongitudeRef], place[ExifTags.GPS.GPSLongitude] = "W", degrees(70, 40)
    Image.new("RGB", (32, 32), "gray").save(tmp_path / "original.jpg", exif=exif)

    found = photo.read(tmp_path / "original.jpg")

    assert found["latitude"] == pytest.approx(-33.8667, abs=1e-4)
    assert found["longitude"] == pytest.approx(-70.6667, abs=1e-4)


def test_a_photo_that_claims_to_be_a_gigapixel_is_left_unmeasured(conn, s3, memory, tmp_path):
    """The header of a decompression bomb says what nobody will ever decode. Nothing is decoded
    for it — and the size that comes back is none, not a size of zero."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(gigantic_photo(tmp_path / "original.jpg")), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (None, None)
    assert read_frame_info(conn, item_id) == (None, None, None, None, None)
    (status,) = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


@pytest.mark.parametrize(
    "model, kept",
    [
        ("abc\x00def", "abcdef"),  # a NUL, which the `text` column refuses
        ("iPhone\n15\x07 Pro", "iPhone15 Pro"),  # a line break and a bell
        ("M" * 60000, "M" * frame_info.MODEL_LENGTH),  # a paragraph, where a camera's name goes
    ],
)
def test_a_camera_name_is_made_a_short_line_of_text(conn, s3, memory, tmp_path, model, kept):
    """The model is the file's own text and every member of the memory is shown it. A NUL in it
    would fail the job three times over and be tried again every day."""
    exif = Image.Exif()
    exif[ExifTags.Base.Model] = model
    Image.new("RGB", (64, 48), "gray").save(tmp_path / "original.jpg", exif=exif)
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(tmp_path / "original.jpg"), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    (status,) = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"
    assert read_frame_info(conn, item_id)[0] == kept


def test_a_png_is_measured_by_ffprobe_and_carries_no_camera(conn, s3, memory, tmp_path):
    """PNG is left to ffprobe: Pillow decodes the whole picture to answer `getexif()` on one, and a
    camera does not write PNG in any case."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(jpeg(tmp_path / "original.png")), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (1200, 800)
    assert read_frame_info(conn, item_id) == (None, None, None, None, None)


def test_a_photo_that_crashes_libheif_does_not_take_the_worker_with_it(conn, s3, memory, tmp_path):
    """Opening this HEIC segfaults the library. It is read in a process of its own, so the job ends
    — `done`, with nothing read — instead of the worker ending with it."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(crashing_heic(tmp_path / "original.heic")), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    (status,) = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


def test_a_portrait_heic_is_measured_the_way_it_is_seen(conn, s3, memory):
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / "portrait-photo.heic"), storage.client().bucket, key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (64, 96)
    assert read_frame_info(conn, item_id)[0] == "iPhone 15 Pro"
