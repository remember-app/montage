"""Remember's montage worker: the background process behind everything the app cannot do itself.

    montage/          the queue and what every job shares — storage, ffmpeg, the journal
    montage/ai/       the cloud model: the door, the providers behind it, what it answers
    montage/media/    the jobs about one uploaded file: preview, size and camera, отсев, речь
    montage/video/    the jobs about one ordered montage, and the pipeline that assembles it

Run it with `python -m montage run`; see README.md.
"""
