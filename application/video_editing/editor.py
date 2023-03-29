from moviepy.editor import *


class VideoEditor:

    def __init__(self, filename):
        self.clip = VideoFileClip(filename)

    def concatenate(self, other):
        self.clip = concatenate_videoclips([self.clip, other.clip])

    def resize(self, width=None, height=None):
        if width is not None and height is not None:
            self.clip = self.clip.resize((width, height))
        elif width is not None:
            self.clip = self.clip.resize(width / self.clip.w)
        elif height is not None:
            self.clip = self.clip.resize(height / self.clip.h)

    def trim(self, start_time=None, end_time=None):
        if start_time is not None and end_time is not None:
            self.clip = self.clip.subclip(start_time, end_time)
        elif start_time is not None:
            self.clip = self.clip.subclip(start_time)
        elif end_time is not None:
            self.clip = self.clip.subclip(0, end_time)

    def speedup(self, factor):
        self.clip = self.clip.speedx(factor)

    def slowmotion(self, factor):
        self.clip = self.clip.fx(vfx.speedx, factor)

    def write(self, filename):
        self.clip.write_videofile(filename)
