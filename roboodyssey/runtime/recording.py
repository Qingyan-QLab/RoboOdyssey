"""CPU MP4 encoder for private review frames; never imports simulator or CUDA."""
import json
import math
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np


class RecordingError(RuntimeError):
    pass


def validate_video_options(options):
    options = dict(options)
    if options.get('layout', 'front_overview') not in {'front', 'front_overview'}:
        raise ValueError('video layout must be front or front_overview')
    fps, seconds = options.get('fps', 10), options.get('event_seconds', .5)
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError('video fps must be between 1 and 60')
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not .1 <= seconds <= 10:
        raise ValueError('video event_seconds must be between .1 and 10')
    encoder = shutil.which(options.get('ffmpeg', 'ffmpeg'))
    if not encoder:
        raise ValueError('CPU ffmpeg executable not found')
    return {**options, 'ffmpeg': encoder, 'fps': fps, 'event_seconds': seconds,
            'layout': options.get('layout', 'front_overview')}


def rgb_image(array):
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or not array.shape[0] or not array.shape[1]:
        raise RecordingError('invalid review RGB dimensions')
    if array.dtype != np.uint8:
        raise RecordingError('review RGB must be uint8')
    return Image.fromarray(array[..., :3])


def wrap(draw, text, font, width):
    lines = []
    for paragraph in str(text).splitlines() or ['']:
        line = ''
        for char in paragraph:
            if draw.textbbox((0, 0), line + char, font=font)[2] > width and line:
                lines.append(line); line = ''
            line += char
        lines.append(line)
    return lines


class SessionRecorder:
    """One event holds one sampled view; video time is explicitly not wall time.

    All encoder arguments are a fixed CPU libx264 profile. Files are finalized on
    normal exceptions/Ctrl-C; a killed process leaves the partial file and index.
    """
    def __init__(self, output, **options):
        self.options = validate_video_options(options)
        self.output = Path(output)
        self.process = None
        self.closed = False
        self.frames = 0
        self.latest = {}
        self.index = (self.output / 'video_index.jsonl').open('x', encoding='utf-8')
        self.error_log = (self.output / 'video_encoder.log').open('xb')
        self.partial = self.output / 'review.partial.mp4'
        self.width, self.height = 1280, 960
        try:
            self.font = ImageFont.truetype('DejaVuSans.ttf', 20)
            self.small = ImageFont.truetype('DejaVuSans.ttf', 16)
        except OSError:
            self.font = self.small = ImageFont.load_default()
        self._manifest('recording')

    def _manifest(self, status, **extra):
        (self.output / 'video_manifest.json').write_text(json.dumps({
            'schema_version': 1, 'status': status, 'private': True,
            'timeline': 'event_sampled', 'description': 'Each event holds a sampled camera frame; inference waits are compressed. See wall-clock timestamps in the index.',
            'encoder': 'libx264 (CPU)', 'fps': self.options['fps'], 'frames': self.frames,
            'duration_seconds': self.frames / self.options['fps'], 'layout': self.options['layout'],
            'size': [self.width, self.height], **extra}, ensure_ascii=False, indent=2) + '\n')

    def _start(self):
        fps = self.options['fps']
        self.process = subprocess.Popen([
            self.options['ffmpeg'], '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
            '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.partial)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_log)

    def capture(self, event, frames=None):
        if self.closed:
            raise RecordingError('recorder already closed')
        if frames is not None:
            required = {'front', 'overview'} if self.options['layout'] == 'front_overview' else {'front'}
            if not required <= frames.keys() or any(frames[k] is None for k in required):
                raise RecordingError('requested review camera frame missing')
            self.latest = {k: rgb_image(frames[k]) for k in required}
        canvas = Image.new('RGB', (self.width, self.height), '#13202b')
        draw = ImageDraw.Draw(canvas)
        title = f"PRIVATE REVIEW | {event['session_id']} | {event.get('task_id') or '-'} | {event['event']}"
        draw.text((20, 14), title, font=self.font, fill='white')
        draw.text((20, 46), f"wall={event['utc']} | event={event['event_id']} | sampled timeline", font=self.small, fill='#a9c8d9')
        panels = [('front', (20, 80, 620, 450)), ('overview', (660, 80, 620, 450))]
        if self.options['layout'] == 'front':
            panels = [('front', (20, 80, 1240, 450))]
        for name, (x, y, w, h) in panels:
            if name in self.latest:
                panel = ImageOps.contain(self.latest[name], (w, h - 28))
                canvas.paste(panel, (x + (w-panel.width)//2, y + 28))
            draw.text((x, y), name + (' (no frame yet)' if name not in self.latest else ''), font=self.small, fill='white')
        text = str(event.get('instruction') or event.get('reason_code') or '')
        lines = wrap(draw, text, self.font, 1240)
        for i, line in enumerate(lines[:14]):
            draw.text((20, 550 + i*25), line, font=self.font, fill='white')
        if len(lines) > 14:
            draw.text((20, 908), 'Full instruction in runtime_events.jsonl', font=self.small, fill='#ffd877')
        status = f"success={event.get('success')} | reason={event.get('reason_code') or '-'} | action={event.get('action') or '-'}"
        draw.text((20, 936), status[:145], font=self.small, fill='#ffd877')
        if self.process is None:
            self._start()
        repeat = max(1, round(self.options['fps'] * self.options['event_seconds']))
        first = self.frames
        try:
            for _ in range(repeat):
                self.process.stdin.write(canvas.tobytes())
                self.frames += 1
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RecordingError('CPU video encoder failed; see video_encoder.log') from exc
        entry = {k: event.get(k) for k in ('event_id', 'event', 'scene_id', 'session_id', 'task_id', 'utc', 'monotonic_ns')}
        entry.update(frame_start=first, frame_end_exclusive=self.frames, video_seconds=first/self.options['fps'],
                     new_camera_sample=frames is not None, video_file='review.mp4')
        self.index.write(json.dumps(entry) + '\n'); self.index.flush()
        return entry

    def close(self, run_status='completed'):
        if self.closed:
            return
        self.closed = True
        error = None
        try:
            if self.process:
                try:
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    code = self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=5)
                    raise RecordingError('video encoder did not finalize within 30 seconds')
                if code:
                    raise RecordingError(f'video encoder exited with {code}; see video_encoder.log')
                self.partial.rename(self.output / 'review.mp4')
            self._manifest('finalized' if self.process else 'empty', run_status=run_status,
                           file='review.mp4' if self.process else None)
        except Exception as exc:
            error = exc
            self._manifest('error', run_status=run_status, error=str(exc), file=self.partial.name)
        finally:
            self.index.close(); self.error_log.close()
        if error:
            raise error
