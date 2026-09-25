"""Режиссёр on recorded answers — see README.md.

Nothing here goes to the network and there is nothing to go with (decision-10): the model answers
out of `fixtures/` through the same transport `test_ai.py` uses, so every test also sees what
*would* have gone out — which is half of what a director is. The database is real, like everywhere
else in this suite: what the director reads (descriptions, transcripts) and what it leaves behind
(the journal row of the plan) are rows, and a fake table would prove nothing about either.

No bucket and no ffmpeg though, and no api migrations: a judged piece is a dataclass, so the pieces
here are written out by hand exactly as in `test_selection.py`. What only a real video can answer —
that a montage actually comes out cut to the plan — is `test_montage.py`.

Warning: these wipe the `ai_requests`, `media_understandings` and `media_speech` tables of whatever
DATABASE_URL points at.
"""

import dataclasses
import datetime
import json

import pytest

from montage import ai, worker
from montage.ai import providers
from montage.video import director, selection

from recorded import Recorded


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        for table in ("ai_requests", "media_understandings", "media_speech"):
            connection.execute(f"DELETE FROM {table}")
        yield connection


@pytest.fixture(autouse=True)
def no_provider_in_the_environment(monkeypatch):
    """The machine running the tests may have a key on it one day; these must not use it."""
    for name in ("AI_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# --- the material -----------------------------------------------------------

MEMORY = "memory-1"
STORY = "день рождения, всё вокруг торта"


def shot(media_id, minute=0, kind="photo", **fields):
    """One piece as `selection.judge` leaves it: readable, with a best moment on it."""
    when = datetime.datetime(2026, 9, 5, 18, minute)
    return selection.Piece(
        media_id=media_id,
        kind=kind,
        sharpness=10.0,
        when=when,
        captured_at=when,
        seconds=selection.CLIP_SECONDS if kind == "video" else selection.PHOTO_SECONDS,
        **fields,
    )


MATERIAL = [shot("m-1", 0), shot("m-2", 5), shot("m-3", 10)]


def describe(conn, media_id, description, speech=None):
    """What `understanding.py` and `speech.py` leave behind on one piece — what the director
    reads."""
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
           VALUES (%s, %s, %s, %s)""",
        (media_id, MEMORY, description, ["человек-1"]),
    )
    if speech is not None:
        conn.execute(
            "INSERT INTO media_speech (media_id, transcript) VALUES (%s, %s)", (media_id, speech)
        )


@pytest.fixture
def described(conn):
    for media_id, description in (
        ("m-1", "накрывают на стол"),
        ("m-2", "вносят торт"),
        ("m-3", "собака под столом"),
    ):
        describe(conn, media_id, description)
    return conn


def directed(
    conn,
    fixture="gemini_plan.json",
    pieces=MATERIAL,
    target_seconds=60.0,
    provider=providers.Gemini,
    **order,
):
    """The film the recorded director plans, what it said while planning it, and the request it was
    planned out of."""
    net = Recorded(fixture)
    brain = ai.AI(conn, provider("test-key", transport=net))
    film, thoughts = director.direct(
        conn, pieces, target_seconds, brain=brain, ref="montage-3", **order
    )
    return film, thoughts, net


def asked(net):
    """The question that went out — the one part of it that changes from montage to montage."""
    return json.loads(net.sent.data)["input"][0]["text"]


# --- the plan ---------------------------------------------------------------


def test_the_plan_is_the_order_the_lengths_and_the_reading_behind_it(described):
    """Everything the director is for, in one: the pieces come back in the order it put them in,
    each at the length it asked for, and it was given what the pipeline already knows about them
    rather than a list of file names."""
    film, _, net = directed(described, story=STORY, target_seconds=30)

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    assert [piece.seconds for piece in film] == [3.0, 4.0, 3.0]
    question = asked(net)
    assert "вносят торт" in question, "what the model itself said about the frame came back to it"
    assert STORY in question and "30" in question
    # And the accent on every piece is in the journal, which is where the plan is kept whole.
    ((response,),) = described.execute(
        "SELECT response FROM ai_requests WHERE operation = 'plan_montage'"
    ).fetchall()
    assert [step["accent"] for step in response["plan"]] == [
        "стол накрывают, гости ещё собираются",
        "вносят торт — главный момент вечера",
        "общий кадр за столом на прощание",
    ]


def test_two_stories_over_the_same_material_are_two_different_films(described):
    """The reason the director exists at all: the same four files, two orders, and the video is not
    the same video. Nothing but the story differs between these two calls."""
    birthday, _, first = directed(described, "gemini_plan.json", story=STORY)
    about_the_dog, _, second = directed(described, "gemini_plan_again.json", story="про собаку")

    assert [piece.media_id for piece in birthday] == ["m-1", "m-2", "m-3"]
    assert [piece.media_id for piece in about_the_dog] == ["m-3", "m-1"]
    assert STORY in asked(first) and "про собаку" in asked(second)


@pytest.mark.parametrize(
    "music",
    [
        [
            {"track": "warm", "seconds": 121.0},
            {"track": "свой трек", "seconds": 48.2, "wish": "торт поставь под припев"},
        ],
        [],
        [
            {
                "track": "calm",
                "seconds": 90.5,
                "wish": 'Тихо 🎂 日本語\nвторая строка\t"кавычки" \\',
            },
            {"track": "свой трек", "seconds": 5.0, "wish": "я" * 5000},
            {"track": "road", "seconds": 60.0},
        ],
    ],
    ids=["a wish at an own track", "no music", "awkward and very long wishes"],
)
def test_the_music_and_what_the_organizer_asked_of_each_track_go_to_the_director(described, music):
    """Под что соберут ролик — часть того, из чего режиссёр строит план: треки по порядку, длина
    каждого и пожелание организатора его же словами — у того трека, к которому оно написано, а не
    одной строкой на весь ролик. Дорожку от этого никто не перекраивает — она собрана до вопроса; в
    запросе она для того, чтобы под неё лёг порядок и длина кусков.

    И всё это обязано оказаться в журнале запросов: запрос к модели сохраняется целиком (doc-03,
    «все запросы обязательно сохраняются»), а прочитать, почему ролик вышел таким, можно только там.
    """
    _, _, net = directed(described, story=STORY, music=music)

    ((request,),) = described.execute(
        "SELECT request FROM ai_requests WHERE operation = 'plan_montage'"
    ).fetchall()
    assert asked(net).endswith(request["prompt"]), "в журнале не тот вопрос, что ушёл модели"
    said = json.loads(request["prompt"])
    assert said["music"] == music, "дорожка ушла не по порядку, без длин или пожелание не у трека"
    assert "music_note" not in said, "общего пожелания на весь ролик больше нет"
    assert "wish" in request["instruction"] and "wish" in asked(net), (
        "поле wish уходит модели, а объяснения, что это пожелание к треку, нет"
    )


def test_what_is_said_in_a_clip_goes_to_the_director_too(conn):
    """A transcript is written once and read from then on (doc-03) — and this is the reader."""
    describe(conn, "m-1", "накрывают на стол", speech="Ну что, все собрались?")
    describe(conn, "m-2", "вносят торт")
    describe(conn, "m-3", "собака под столом")

    _, _, net = directed(conn, story=STORY)

    assert "Ну что, все собрались?" in asked(net)


def test_a_clip_is_never_held_longer_than_it_actually_runs(described):
    """The plan asks for four seconds of a two-second clip. What it gets is the two there are —
    booked in at four, the crossfade chain would walk off the end of the montage."""
    two_seconds = shot("m-2", 5, kind="video", duration=2.0, best_second=1.0)
    film, *_ = directed(described, pieces=[MATERIAL[0], two_seconds, MATERIAL[2]])

    assert [piece.seconds for piece in film] == [3.0, 2.0, 3.0]


def test_a_plan_longer_than_the_video_that_was_ordered_is_cut_back_to_it(described):
    """The length is in the prompt, but the prompt is not a promise: ten seconds of plan against
    eight seconds ordered leaves the piece that does not fit out of the film."""
    film, thoughts, _ = directed(described, target_seconds=8)

    # Three and four seconds with half a second of crossfade between them is 6.5; the third piece
    # would make it 9.
    assert [piece.media_id for piece in film] == ["m-1", "m-2"]
    # И рассказано это тем же словом: кадр, который планировали, но он не влез, — «мимо, места нет».
    assert [(one["media_id"], one["decision"], one.get("reason")) for one in thoughts] == [
        ("m-1", "take", None),
        ("m-2", "take", None),
        ("m-3", "skip", selection.NO_ROOM),
    ]


# --- что режиссёр рассказал о своём плане ------------------------------------


def about(thoughts, media_id):
    """Что режиссёр сказал об одном кадре — одной репликой, или ни одной."""
    (one,) = [one for one in thoughts if one.get("media_id") == media_id]
    return one


def test_the_director_says_what_it_sees_which_part_it_is_and_where_to_look(described):
    """Рассказ режиссёра целиком, порядком плана: мысль о плане в целом первой и без кадра, за ней
    по реплике на каждый показанный кадр — мысль его словами, часть истории номером и названием,
    рамка «куда смотрит» долями, — а кадр, который он в план не взял, идёт «мимо» с его же
    словами о причине.
    """
    film, thoughts, _ = directed(described, "gemini_plan_thoughts.json", story=STORY)

    assert [piece.media_id for piece in film] == ["m-1", "m-2"]
    assert thoughts[0] == {
        "thought": (
            "План на две части: сперва сборы за столом, потом торт — на нём и держим финал. "
            "Собака под столом в эту историю не встала."
        )
    }, "обзорная мысль — первой записью и без кадра: она про фильм, а не про кусок"
    assert [one.get("media_id") for one in thoughts] == [None, "m-1", "m-2", "m-3"]

    assert about(thoughts, "m-1") == {
        "media_id": "m-1",
        "decision": "take",
        "thought": "Стол накрывают, гости ещё собираются — с этого вечер и начинается.",
        "story_part": 1,
        "story_part_title": "Сборы",
    }
    assert about(thoughts, "m-2") == {
        "media_id": "m-2",
        "decision": "take",
        "thought": "Вносят торт — главный момент вечера, держу подольше и смотрю на свечи.",
        "story_part": 2,
        "story_part_title": "Торт",
        "focus_x": 0.3,
        "focus_y": 0.05,
        "focus_w": 0.25,
        "focus_h": 0.5,
    }
    assert about(thoughts, "m-3") == {
        "media_id": "m-3",
        "decision": "skip",
        "reason": director.LEFT_OUT,
        "thought": "собака не про этот вечер",
    }, "причина — кодом, как везде в журнале, а слова модели о ней — мыслью: их и покажут людям"


def test_a_plan_written_before_the_director_spoke_is_still_a_plan(described):
    """Расширение, а не новый формат: записанный ответ без мыслей, частей и рамок разбирается тем
    же разбором. Порядок и длины из него те же, что были, `accent` становится мыслью о кадре, а
    частей и рамок в рассказе просто нет — выдумывать их за модель некому.
    """
    film, thoughts, _ = directed(described, "gemini_plan.json", story=STORY)

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    assert [piece.seconds for piece in film] == [3.0, 4.0, 3.0]
    assert about(thoughts, "m-2") == {
        "media_id": "m-2",
        "decision": "take",
        "thought": "вносят торт — главный момент вечера",
        "story_part": None,
        "story_part_title": None,
    }


def test_the_same_plan_comes_back_whole_from_the_other_provider(described):
    """Поставщик на слова режиссёра не влияет: тот же расширенный план, записанный ответом OpenAI,
    читается ровно так же — разбор живёт у двери к модели, а не у каждого адаптера."""
    film, thoughts, _ = directed(
        described, "openai_plan.json", provider=providers.OpenAI, story=STORY
    )

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    assert thoughts[0]["thought"].startswith("Три части по кадру")
    assert about(thoughts, "m-2")["focus_w"] == 0.5
    assert [one.get("story_part") for one in thoughts] == [None, 1, 2, 3]


def test_what_the_director_made_up_does_not_reach_the_journal(described):
    """Ответ модели — недоверенный ввод, а мысль увидят все участники воспоминания. Чужой кадр в
    плане, часть «нулевая», рамка за краем кадра и рамка из трёх долей — всё это выдумки, и до
    экрана они не доходят; мысль о кадре при этом остаётся, потерять её из-за кривой рамки было бы
    хуже.
    """
    film, thoughts, net = directed(described, "gemini_plan_wild.json", story=STORY)

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    assert "m-404" not in asked(net), "модели такого кадра не показывали"
    assert not [one for one in thoughts if one.get("media_id") == "m-404"], (
        "и рассказывать о нём нечего: в журнале только кадры воспоминания"
    )
    # Рамка либо целая и внутри кадра, либо никакой: 1.4 — не доля, 0.9 + 0.4 вылезает за край,
    # а из трёх долей рамку не собрать.
    assert not [one for one in thoughts if any(key.startswith("focus") for key in one)]
    assert about(thoughts, "m-1")["story_part"] is None, "«часть 0» — не номер части"
    assert about(thoughts, "m-2")["story_part"] == 2, "а число строкой — номер"
    assert about(thoughts, "m-3")["thought"] is None, "мысль списком — не мысль"
    # Длина, которую нельзя прочитать числом, оставляет кусок таким, каким его отрезал отсев.
    assert [piece.seconds for piece in film] == [3.0, 4.0, selection.PHOTO_SECONDS]


def test_what_the_database_cannot_hold_costs_the_plan_nothing_but_itself(described):
    """За ответ заплачено, и значение, которого база не примет или которое не число, не стоит ни
    ответа, ни ролика. `"NaN"` строкой и `1e999` — не длины: кусок идёт длиной отсева, а не
    «не числом» сквозь все ограничения. Целое длиннее double — не доля кадра, и рамки нет. NUL и
    половинка суррогатной пары выпадают из слов, а сами слова остаются."""
    film, thoughts, _ = directed(described, "gemini_plan_unholdable.json", story=STORY)

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    assert [piece.seconds for piece in film] == [selection.PHOTO_SECONDS] * 2 + [3.0]
    assert thoughts[0] == {"thought": "План на три части"}
    assert about(thoughts, "m-2") == {
        "media_id": "m-2",
        "decision": "take",
        "thought": "Вносят торт",
        "story_part": None,
        "story_part_title": "Торт",
    }
    assert about(thoughts, "m-3")["focus_w"] == 0.3, "годная рамка рядом остаётся"
    ((status, overview),) = described.execute(
        "SELECT status, response->>'overview' FROM ai_requests WHERE operation = 'plan_montage'"
    ).fetchall()
    assert (status, overview) == ("answered", "План на три части")


# --- one shot per moment ----------------------------------------------------


def twin_of(piece, media_id, sharpness, seconds_later=1.0):
    """Another shot of the same moment: the shutter held down, a second apart at most."""
    return dataclasses.replace(
        piece,
        media_id=media_id,
        sharpness=sharpness,
        captured_at=piece.captured_at + datetime.timedelta(seconds=seconds_later),
    )


def test_a_burst_of_one_moment_reaches_the_director_as_one_shot(described):
    """Two shots of the same second are one moment, and the model is shown the sharper of them —
    the отбор's own дедуп, run before the material becomes a question. A director handed both can
    put them side by side in the plan, and the second of them is paid for on the way out as well."""
    describe(described, "m-9", "собака под столом, кадр секундой позже")
    blurrier = twin_of(MATERIAL[2], "m-9", sharpness=1.0)

    film, _, net = directed(described, pieces=[*MATERIAL, blurrier], story=STORY)

    assert "m-9" not in asked(net) and "секундой позже" not in asked(net)
    assert "собака под столом" in asked(net), "the sharpest of the burst stands for it"
    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    # И сколько кадров за ним стояло, кусок несёт на себе — журналу монтажа серия одна запись
    # «взял 1 из 2», ровно как на пути отсева (`story.plan`).
    assert [piece.series_size for piece in film] == [1, 1, 2]


def test_the_question_keeps_the_order_the_memory_is_read_in(described):
    """Порядок вопроса — тот, в котором воспоминание читается, и дедуп его не переставляет: свои
    кадры он возвращает своим порядком, первыми — те, у кого даты съёмки нет вовсе. Модели же
    материал показывают так, как он пришёл, и «третий кадр» в плане значит третий сверху."""
    describe(described, "m-9", "торт несут мимо окна")
    undated = dataclasses.replace(shot("m-9", 7), captured_at=None)

    _, _, net = directed(described, pieces=[MATERIAL[0], undated, *MATERIAL[1:]], story=STORY)

    question = asked(net)
    assert sorted(("m-1", "m-9", "m-2", "m-3"), key=question.index) == ["m-1", "m-9", "m-2", "m-3"]


def test_the_shot_the_organizer_asked_for_is_the_one_its_burst_is_shown_by(described):
    """«Включить» beats sharpness inside a burst here exactly as it does in `selection.choose`: the
    organizer named the blurrier shot, so that is the one moment's shot the director is offered —
    and the sharper one next to it is not also offered."""
    describe(described, "m-9", "собака под столом, кадр секундой позже")
    blurrier = twin_of(MATERIAL[2], "m-9", sharpness=1.0)

    film, _, net = directed(described, pieces=[*MATERIAL, blurrier], include=["m-9"])

    assert "секундой позже" in asked(net), "the shot that was asked for"
    assert "m-3" not in asked(net), "and not the sharper one beside it"
    # The recorded plan names m-3, which is not material any more; «включить» puts m-9 in itself.
    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-9"]


# --- what the organizer said ------------------------------------------------


def test_a_struck_out_piece_is_never_even_shown_to_the_director(described):
    """«Исключить» is not a filter over the answer — the piece does not go into the question, so
    there is nothing for the model to name. Named here anyway (the recorded plan has all three),
    which is exactly the case this rule has to survive: a model that says otherwise."""
    blurred = dataclasses.replace(shot("m-4", 15), rejected=selection.BLUR)
    film, thoughts, net = directed(described, pieces=[*MATERIAL, blurred], exclude=["m-3", "m-4"])

    assert [piece.media_id for piece in film] == ["m-1", "m-2"]
    assert "m-3" not in asked(net) and "собака" not in asked(net)
    # Слово организатора видно и на этом пути — «мимо, вычеркнут», как у отсева; а смазанному
    # второго «мимо» нет: о нём рассказал отсев.
    assert (about(thoughts, "m-3")["decision"], about(thoughts, "m-3")["reason"]) == (
        "skip",
        selection.EXCLUDED,
    )
    assert not [one for one in thoughts if one.get("media_id") == "m-4"]


def test_told_both_at_once_the_piece_stays_out_of_the_film_too(described):
    """The same contradiction `selection.choose` settles the same way: named on both of the
    organizer's own lists, a piece is struck out before the director ever sees it — «исключить» is
    what cannot be undone once a video is made, so it is the one that wins here as well."""
    film, _, net = directed(described, include=["m-3"], exclude=["m-3"])

    assert [piece.media_id for piece in film] == ["m-1", "m-2"]
    assert "m-3" not in asked(net) and "собака" not in asked(net)


def test_a_piece_the_organizer_asked_for_is_in_the_film_even_if_the_plan_forgot_it(described):
    """«Включить» is not the model's to refuse. The recorded plan names three pieces and this is a
    fourth, so the only way it can be in the film is the rule putting it back."""
    describe(described, "m-4", "все машут в камеру")
    film, thoughts, net = directed(described, pieces=[*MATERIAL, shot("m-4", 15)], include=["m-4"])

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3", "m-4"]
    assert '"must_include": true' in asked(net), "and it was asked for, not only forced afterwards"
    # И в журнале это «беру» при любом плане: модель о кадре не сказала ничего, а он в ролике.
    (about_it,) = [one for one in thoughts if one.get("media_id") == "m-4"]
    assert (about_it["decision"], about_it.get("thought")) == ("take", None)


def test_a_piece_the_selection_turned_down_is_in_the_film_when_the_organizer_asks_for_it(described):
    """The same overruling the отбор honours: too dark for the algorithm, wanted by the person who
    was there. It is offered to the director like everything else, and put back if the plan skips
    it — and it goes in with nothing about it rejected."""
    dark = dataclasses.replace(MATERIAL[2], rejected=selection.EXPOSURE)

    film, _, net = directed(described, pieces=[*MATERIAL[:2], dark], include=["m-3"])

    assert [piece.media_id for piece in film] == ["m-1", "m-2", "m-3"]
    assert [piece.rejected for piece in film] == [None] * 3
    assert "собака под столом" in asked(net)


def test_what_the_selection_turned_down_and_nobody_asked_for_is_not_offered(described):
    """The other way round: the director is offered what a person would be offered, and a blurred
    frame nobody named is not on that list."""
    blurred = dataclasses.replace(MATERIAL[2], rejected=selection.BLUR)

    film, _, net = directed(described, pieces=[*MATERIAL[:2], blurred])

    assert [piece.media_id for piece in film] == ["m-1", "m-2"]
    assert "m-3" not in asked(net)


def test_a_file_with_no_frame_in_it_stays_out_even_when_the_organizer_asked_for_it(described):
    """The one word «включить» cannot overrule, here as in the отбор: there is no picture to put on
    screen, and a director cannot be shown one either."""
    broken = dataclasses.replace(MATERIAL[2], rejected=selection.UNREADABLE)

    film, _, net = directed(described, pieces=[*MATERIAL[:2], broken], include=["m-3"])

    assert [piece.media_id for piece in film] == ["m-1", "m-2"]
    assert "m-3" not in asked(net)


# --- when there is no director ----------------------------------------------


def test_without_a_cloud_model_nobody_is_asked_and_the_selection_cuts_the_video(conn, monkeypatch):
    """Today's state, and the one thing this module must not change: no model, no plan, and the
    montage is the montage it has always been (`montage.py` falls back to `selection.choose`)."""
    monkeypatch.setattr(ai, "from_env", lambda *_, **__: None)

    assert director.direct(conn, MATERIAL, 60.0, ref="montage-3") == (None, ())
    assert conn.execute("SELECT count(*) FROM ai_requests").fetchone()[0] == 0


def test_material_nobody_has_described_yet_is_left_to_the_selection(conn):
    """A memory that became ready before there was a key, with the hourly sweep not yet through it.
    Ordering frames it has never seen is what the model would be paid for here, and the algorithm
    at least looks at the pixels."""
    film, _, net = directed(conn, story=STORY)

    assert film is None
    assert net.requests == [], "and nobody was asked anything"
    assert conn.execute("SELECT count(*) FROM ai_requests").fetchone()[0] == 0


def test_a_plan_with_nothing_of_ours_in_it_falls_back_instead_of_making_an_empty_video(described):
    """A model that answered about material it was never shown. Better the отбор's video than none
    — and `montage.py` reads None as «nobody directed this»."""
    describe(described, "m-9", "единственное, что у нас есть")

    film, *_ = directed(described, pieces=[shot("m-9", 0)])

    assert film is None


def test_the_call_that_planned_the_film_is_in_the_journal_under_the_montage(described):
    """Every question put to a model is written down (doc-03), the director's like the other two —
    and this one is about a montage, so that is what `ref` names."""
    directed(described, story=STORY)

    assert described.execute("SELECT operation, ref, status FROM ai_requests").fetchall() == [
        ("plan_montage", "montage-3", "answered")
    ]
