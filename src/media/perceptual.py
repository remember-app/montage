import pathlib

# dHash: восемь строк по восемь сравнений соседних пикселей серой сетки 9×8.
SIDE = 8


def hash_of(path: pathlib.Path) -> str:
    """64-битный отпечаток картинки в 16 hex-знаках: у похожих кадров отличаются немногие биты."""
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("L").resize((SIDE + 1, SIDE), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for row in range(SIDE):
        start = row * (SIDE + 1)
        for col in range(SIDE):
            bits = (bits << 1) | int(pixels[start + col] > pixels[start + col + 1])
    return f"{bits:016x}"


def _distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


if __name__ == "__main__":
    import tempfile

    from PIL import Image, ImageDraw

    def sample(shift: int) -> pathlib.Path:
        image = Image.new("L", (64, 64))
        ImageDraw.Draw(image).ellipse((10 + shift, 10, 50 + shift, 50), fill=255)
        file = pathlib.Path(tempfile.mkdtemp()) / f"{shift}.png"
        image.save(file)
        return file

    near, far = sample(2), sample(40)
    assert _distance(hash_of(sample(0)), hash_of(near)) <= 12, "small shift stays close"
    assert _distance(hash_of(sample(0)), hash_of(far)) > 12, "different picture does not"
    print("ok")
