INSTRUCTION = (
    "You analyze files shot by people at the same event. Answer in English and reply with a "
    "JSON object only."
)
PHOTO_PROMPT = (
    "Look closely at the photo. Reply with a JSON object:\n"
    '{"summary": "...", "description": "...", "scoreImportance": 0.0, "scoreQuality": 0.0, '
    '"tags": {"key": "value"}}\n'
    "summary is one sentence for the participant who ordered the movie. description is a detailed "
    "observation: who and how many people, what they are doing, faces and expressions, what is in "
    "their hands, the place, light and colour, what is cut off or accidentally in frame, and what "
    "a person choosing a shot would care about. tags is a flat object of string keys and string "
    "values; choose the keys yourself and put there whatever else helps a director: shot size, "
    "role, people, faces, emotion, where the main subject is, where they face, what is moving, "
    "quality flags."
)
VIDEO_PROMPT = (
    "These are frames from one video, taken at seconds {seconds}. Reply with a JSON object:\n"
    '{"summary": "...", "description": "...", "scoreImportance": 0.0, "scoreQuality": 0.0, '
    '"tags": {"key": "value"}, '
    '"frames": [{"description": "...", "tags": {"key": "value"}}]}\n'
    "summary is one sentence for the participant. description is a detailed observation of the "
    "whole video. tags is a flat object of string keys and string values. Go through the frames in "
    "the same order and describe what happens in each; frames must have exactly one entry per "
    "frame given. In the frame tags, choose the keys yourself: shot size, role, people, faces, "
    "emotion, subject position, camera motion, quality."
)
SEGMENT_PROMPT = (
    "These are frames from one segment of a video, taken at seconds {seconds}. The video before "
    "this segment: {context}\nReply with a JSON object:\n"
    '{"context": "...", "description": "...", "scoreQuality": 0.0, "tags": {"key": "value"}}\n'
    "context is one or two sentences about the whole video so far, updated with this segment. "
    "description says what happens in this segment. tags is a flat object of string keys and "
    "string values: shot size, role, people, faces, emotion, subject position, camera motion, "
    "quality, and anything else a director would use."
)
FINAL_VIDEO_PROMPT = (
    "These are descriptions of consecutive segments of one video:\n{segments}\n"
    "Reply with a JSON object for the whole video:\n"
    '{"summary": "...", "description": "...", "scoreImportance": 0.0, '
    '"tags": {"key": "value"}}\n'
    "summary is one sentence for the participant. description is a detailed observation of the "
    "whole video. tags is a flat object of string keys and string values."
)
SOUND_PROMPT = (
    "Listen to the sound of a video from an event. Reply with a JSON object:\n"
    '{"description": "...", "tags": {"key": "value"}, '
    '"sounds": [{"secStart": 0.0, "secEnd": 0.0, "description": "...", "tags": {"key": "value"}}], '
    '"words": [{"secStart": 0.0, "secEnd": 0.0, "description": "...", "tags": null}]}\n'
    "description is a short description of the soundscape. sounds are non-speech events: laughter, "
    "applause, music, cheering, singing, noise. words are the speech, transcribed. Times are in "
    "seconds inside the recording."
)
AUDIO_PROMPT = (
    "Listen to the recording. Reply with a JSON object:\n"
    '{"summary": "...", "description": "...", "scoreImportance": 0.0, "scoreQuality": 0.0, '
    '"tags": {"key": "value"}, '
    '"sounds": [{"secStart": 0.0, "secEnd": 0.0, "description": "...", "tags": {"key": "value"}}], '
    '"words": [{"secStart": 0.0, "secEnd": 0.0, "description": "...", "tags": null}]}\n'
    "summary is one sentence for the participant. description is a detailed observation of what "
    "is heard: voices, laughter, music, noise, what is happening. tags is a flat object of string "
    "keys and string values; choose the keys yourself. sounds are non-speech events. words are the "
    "speech, transcribed."
)
MUSIC_PROMPT = (
    "This is a music track. Reply with a JSON object:\n"
    '{"summary": "...", "description": "...", "score": 0.0, "tags": {"key": "value"}, '
    '"sounds": [{"secStart": 0.0, "secEnd": 0.0, "description": "...", "tags": {"key": "value"}}], '
    '"words": [{"secStart": 0.0, "secEnd": 0.0, "description": "...", "tags": null}]}\n'
    "summary is one sentence for the participant. description covers the mood, instruments and how "
    "the track develops. tags is a flat object of string keys and string values, for example bpm, "
    "key, energy, use. sounds are musical sections with their mood and instruments. words are the "
    "lyrics, transcribed."
)
