"""Понимание кадров: what one media item is described as, and how one memory keeps one label per
person — see README.md.

Live like the rest of the suite (a real Postgres, a real bucket, real ffmpeg) except for the model
itself, which answers out of `fixtures/` exactly as in `test_ai.py`: nothing here goes to the
network, and there is no key for it to go with anyway (decision-10).

Warning: these wipe the `jobs`, `ai_requests` and `media_understandings` tables of whatever
DATABASE_URL points at.
"""

import json

import pytest

from montage import ai, storage, sweeps, worker
from montage.ai import providers
from montage.media import understanding

from material import FIXTURES, jpeg, video
from recorded import saying
from rows import add_item


class Answering:
    """The model, replaced by what somebody wrote down: one recorded answer per call, in order.

    The last of them stands for every call past the list, so a test that cares about one answer
    names one. Every request is kept — what *would* have gone to the network is half of what these
    tests are about.
    """

    def __init__(self, *fixtures):
        self.answers = [(FIXTURES / name).read_bytes() for name in fixtures]
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.answers[min(len(self.requests), len(self.answers)) - 1]

    def _body(self, call):
        return json.loads(self.requests[call].data)

    def asked(self, call=0):
        """What was asked on that call, without the standing instruction Gemini takes in the same
        part — the instruction names «человек-N» itself, and the question is the half that says who
        this memory has already met."""
        return self._body(call)["input"][0]["text"].split("\n\n", 1)[-1]

    def shown(self, call=0):
        """And the parts that are not words: the frames it was shown."""
        return self._body(call)["input"][1:]


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        for table in ("jobs", "ai_requests", "media_understandings"):
            connection.execute(f"DELETE FROM {table}")
        yield connection


@pytest.fixture
def model(monkeypatch):
    """The only cloud model this worker can find, answering out of `fixtures/`.

    `ai.from_env` and not the environment: the gateway builds its provider through it, so one
    recorded model reaches the handler, the queue and the sweep alike, with no key anywhere.
    """

    def answering(*fixtures):
        net = Answering(*fixtures)
        monkeypatch.setattr(
            ai, "from_env", lambda *_, **__: providers.Gemini("test-key", transport=net)
        )
        return net

    return answering


@pytest.fixture
def no_model(monkeypatch):
    """Today's state: nobody to ask (decision-10)."""
    monkeypatch.setattr(ai, "from_env", lambda *_, **__: None)


def ready_media(conn, s3, memory, preview, kind="photo"):
    """A media item as `media_preview` leaves it: `ready`, with the copy understanding looks at."""
    item_id, _ = add_item(conn, memory, kind, status="ready")
    preview_key = f"previews/{item_id}.{'mp4' if kind == 'video' else 'jpg'}"
    conn.execute("UPDATE media_items SET preview_key = %s WHERE id = %s", (preview_key, item_id))
    s3.upload_file(str(preview), storage.client().bucket, preview_key)
    return item_id


def described(conn, item_id):
    return conn.execute(
        "SELECT description, cast_labels FROM media_understandings WHERE media_id = %s", (item_id,)
    ).fetchone()


def queued_understanding_for(conn, item_id):
    """Whatever else this database holds is not this test's business — the sweep is over the whole
    table, and a dev stand's own media would otherwise be queued into the run."""
    conn.execute("DELETE FROM jobs WHERE ref IS DISTINCT FROM %s", (item_id,))
    return conn.execute(
        """SELECT count(*) FROM jobs
           WHERE type = 'media_understanding' AND ref = %s AND status = 'queued'""",
        (item_id,),
    ).fetchone()[0]


def test_a_ready_photo_is_described_and_the_answer_is_kept(conn, s3, memory, tmp_path, model):
    net = model("gemini_describe.json")
    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))
    worker.enqueue(conn, "media_understanding", item_id)

    assert worker.run_once(conn) is True

    description, cast = described(conn, item_id)
    assert description.startswith("Женщина")
    assert cast == ["человек-1", "животное-1"]
    assert [part["type"] for part in net.shown()] == ["image"]
    # And the call it came from is findable afterwards, under the item's own id.
    assert conn.execute("SELECT operation, ref, status FROM ai_requests").fetchall() == [
        ("describe_media", item_id, "answered")
    ]


def test_a_described_frame_is_never_described_again(conn, s3, memory, tmp_path, model):
    """doc-03: «Второй ролик из тех же медиа не описывает заново уже описанные кадры». The sweep
    and every later montage come back here on frames described weeks ago, and the model must not
    be paid for any of it twice."""
    net = model("gemini_describe.json")
    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))

    for _ in range(2):
        worker.enqueue(conn, "media_understanding", item_id)
        assert worker.run_once(conn) is True

    assert len(net.requests) == 1, "the second run asked nobody anything"
    assert conn.execute("SELECT count(*) FROM ai_requests").fetchone()[0] == 1
    assert described(conn, item_id)[0].startswith("Женщина"), "and the kept answer is still there"


def test_the_same_person_keeps_one_label_across_the_memory(conn, s3, memory, tmp_path, model):
    """The whole mechanism in one test: the cast a memory has already named goes out with the next
    frame, so the model recognises «человек-1» instead of starting the numbering over — and the
    stranger beside her becomes «человек-2» rather than «человек-1» a second time."""
    net = model("gemini_describe.json", "gemini_describe_again.json")
    first = ready_media(conn, s3, memory, jpeg(tmp_path / "one.jpg"))
    second = ready_media(conn, s3, memory, jpeg(tmp_path / "two.jpg"))

    for item_id in (first, second):
        worker.enqueue(conn, "media_understanding", item_id)
        assert worker.run_once(conn) is True

    assert "человек-1" not in net.asked(0), "nobody was named when the first frame was described"
    assert "человек-1" in net.asked(1) and "животное-1" in net.asked(1)
    assert described(conn, first)[1] == ["человек-1", "животное-1"]
    assert described(conn, second)[1] == ["человек-1", "человек-2"]


@pytest.mark.parametrize(
    "cast, labels",
    [("человек-1", []), (["человек-1", None, 2, {"label": "животное-1"}], ["человек-1"])],
)
def test_only_strings_out_of_a_list_become_labels(
    conn, s3, memory, tmp_path, monkeypatch, cast, labels
):
    """Метки лягут на кадр навсегда. Список строкой разобрался бы по буквам, а `null`, число и
    объект в списке стали бы метками «None», «2» и словарём текстом. Описание при этом остаётся —
    за него заплачено."""
    said = json.dumps({"description": "Женщина вносит торт", "cast": cast}, ensure_ascii=False)
    monkeypatch.setattr(
        ai, "from_env", lambda *_, **__: providers.Gemini("test-key", transport=saying(said))
    )
    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))
    worker.enqueue(conn, "media_understanding", item_id)

    assert worker.run_once(conn) is True

    assert described(conn, item_id) == ("Женщина вносит торт", labels)


def test_an_answer_nested_past_any_parser_is_paid_for_once(conn, s3, memory, tmp_path, monkeypatch):
    """Ответ в сотню тысяч скобок глубиной ронял разбор уже после оплаты: задача уходила на повтор,
    и каждая из попыток платила снова. Глубже предела — `null`, описание остаётся, задача закрыта с
    первой попытки, и вызов за неё один."""
    said = (
        '{"description": "Женщина вносит торт", "cast": ["человек-1"], '
        f'"nested": {"[" * 100_000}{"]" * 100_000}}}'
    )
    monkeypatch.setattr(
        ai, "from_env", lambda *_, **__: providers.Gemini("test-key", transport=saying(said))
    )
    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))
    worker.enqueue(conn, "media_understanding", item_id)

    assert worker.run_once(conn) is True

    assert described(conn, item_id) == ("Женщина вносит торт", ["человек-1"])
    assert conn.execute(
        "SELECT status, attempts FROM jobs WHERE ref = %s", (item_id,)
    ).fetchall() == [("done", 0)]
    assert conn.execute("SELECT status FROM ai_requests").fetchall() == [("answered",)]


def test_a_clip_is_described_by_stills_cut_out_of_it(conn, s3, memory, tmp_path, model):
    """Frames and not the clip whole: OpenAI takes no video at all, and Gemini charges one by its
    length (doc-11) — so a two-minute clip costs the same three photographs as a five-second one."""
    net = model("gemini_describe.json")
    item_id = ready_media(conn, s3, memory, video(tmp_path / "preview.mp4"), kind="video")
    worker.enqueue(conn, "media_understanding", item_id)

    assert worker.run_once(conn) is True

    shown = net.shown()
    assert [part["type"] for part in shown] == ["image"] * understanding.FRAMES_PER_CLIP
    assert {part["mime_type"] for part in shown} == {"image/jpeg"}
    assert len({part["data"] for part in shown}) == understanding.FRAMES_PER_CLIP, (
        "three moments of the clip, not one frame three times"
    )
    assert described(conn, item_id)[0], "and the clip is described like anything else"


def test_without_a_cloud_model_nothing_is_described_and_nobody_is_asked(
    conn, s3, memory, tmp_path, no_model
):
    """The state the worker is in today. The job finishes cleanly — there is nothing wrong with
    having no model — and the frame stays undescribed, so it is described for real once there is
    somebody to ask."""
    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))
    worker.enqueue(conn, "media_understanding", item_id)

    assert worker.run_once(conn) is True

    assert described(conn, item_id) is None
    assert conn.execute("SELECT count(*) FROM ai_requests").fetchone()[0] == 0, (
        "the gateway was never entered, so it has nothing to journal"
    )
    assert sweeps.sweep_media_without_ai_answers(conn) == 0, (
        "nor is the whole library queued every hour to be told the same thing"
    )


def test_the_analysis_stage_asks_about_a_frame_only_once(conn, s3, memory, tmp_path, model):
    """Главное обещание этапа анализа: кадр, описанный раньше, в нейросеть второй раз не уходит.

    Ролик собирается из материала, которое уже месяц лежит описанным (тот же вопрос, что и у задачи
    из очереди, и тот же ответ). Один кадр описан, другой — нет: модель спрашивается ровно об одном,
    и в журнале один помечен «описан ранее», а второй описан заново и таким помечен не помечен.

    Без этой проверки тест проходил бы и с тем, что этап спрашивает обоих: «описан ранее» тогда
    просто рисовалось бы, а второй вызов модели не был бы виден.
    """
    net = model("gemini_describe.json", "gemini_describe_again.json")
    known = ready_media(conn, s3, memory, jpeg(tmp_path / "one.jpg"))
    fresh = ready_media(conn, s3, memory, jpeg(tmp_path / "two.jpg"))
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
           VALUES (%s, %s, 'Бабушка у окна', %s)""",
        (known, str(memory[0]), ["человек-1"]),
    )

    earlier = understanding.describe_missing(conn, [known, fresh])

    assert earlier == {known}, "взято готовым — только то, что уже было описано"
    assert len(net.requests) == 1, "спрошено ровно об одном кадре — о том, кого ещё не описывали"
    assert described(conn, fresh) is not None, "и второй кадр описан на этот раз"


def test_the_analysis_stage_describes_nothing_when_there_is_no_cloud_model(
    conn, s3, memory, tmp_path, no_model
):
    """Сегодняшнее состояние (decision-10): описать нечем. Этап проходит, ничего не спрашивает и
    ничего не выдумывает — в журнале не будет ни одной записи об описании, которого не было.
    """
    ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))

    earlier = understanding.describe_missing(conn, [add_item(conn, memory, "photo")[0]])

    assert earlier == set()
    assert conn.execute("SELECT count(*) FROM ai_requests").fetchone()[0] == 0


def test_a_fresh_item_is_queued_for_understanding_as_soon_as_its_preview_is_made(
    conn, s3, memory, tmp_path
):
    """A frame is describable the moment the copies exist, and the director wants it described long
    before anybody orders a video — so the preview job queues the understanding one."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert queued_understanding_for(conn, item_id) == 1


def test_the_sweep_picks_up_media_that_was_ready_before_there_was_anybody_to_ask(
    conn, s3, memory, tmp_path, model
):
    """Everything that became ready while there was no key: nothing was ever queued for it, and
    nothing about the row changes on its own. Once described, it is not asked for again."""
    model("gemini_describe.json")
    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))

    sweeps.sweep_media_without_ai_answers(conn)
    assert queued_understanding_for(conn, item_id) == 1
    assert worker.run_once(conn) is True
    assert described(conn, item_id) is not None

    sweeps.sweep_media_without_ai_answers(conn)
    assert queued_understanding_for(conn, item_id) == 0, "an hour later it is not asked for again"


def test_the_sweep_picks_up_frames_the_model_refused_to_describe(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The live case this catch-up was written for: a free-tier account answering «429» to every
    call. Three attempts go in a minute, the job is `failed` for good, and nothing else will ever
    come for that frame — ordering a video describes nothing, it only reads what is in the tables.

    So the sweep comes back for it. Not the same hour, though: a model that refuses refuses every
    frame, and asking again an hour later is three paid calls an hour per frame for as long as the
    worker runs. A day later, once — by which time the account's limit has rolled over, or the
    human has bought a bigger one.
    """
    over_the_limit = [True]

    def answering_429(request):
        if over_the_limit[0]:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        return (FIXTURES / "gemini_describe.json").read_bytes()

    item_id = ready_media(conn, s3, memory, jpeg(tmp_path / "preview.jpg"))
    monkeypatch.setattr(
        ai, "from_env", lambda *_, **__: providers.Gemini("test-key", transport=answering_429)
    )

    sweeps.sweep_media_without_ai_answers(conn)
    assert queued_understanding_for(conn, item_id) == 1
    for _ in range(worker.MAX_ATTEMPTS):
        conn.execute("UPDATE jobs SET run_after = now() WHERE ref = %s", (item_id,))
        assert worker.run_once(conn) is True
    assert described(conn, item_id) is None

    sweeps.sweep_media_without_ai_answers(conn)
    assert queued_understanding_for(conn, item_id) == 0, "an hour later it is not asked all over"

    conn.execute(
        "UPDATE jobs SET updated_at = now() - make_interval(hours => %s) WHERE ref = %s",
        (sweeps.CATCH_UP_AFTER_HOURS + 1, item_id),
    )
    sweeps.sweep_media_without_ai_answers(conn)
    assert queued_understanding_for(conn, item_id) == 1, "the day after, it is given one more go"
    over_the_limit[0] = False  # the paid tier is on: the refused frame is described for real
    assert worker.run_once(conn) is True
    assert described(conn, item_id)[0].startswith("Женщина")
