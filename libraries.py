import cv2

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # return cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)    



from threading import Thread
import sys
# import cv2
import time

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


#multithread video decode and downscale to 720p
class FileVideoStream: 
    # def __init__(self, path, transform=None, queue_size=128):
    def __init__(self, path, transform=None, queue_size=196):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        # self.stream = cv2.VideoCapture(path, cv2.CAP_MSMF)

        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True
                    
                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                if frame is not None:
                    frame = ResizeWithAspectRatio(frame, width=1280) #slow function, about 2.5ms
                    # add the frame to the queue
                    self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()    

# Frametime histogram buckets per target FPS: (min_ms, max_ms, label, BGR colour).
# A frametime falls in a bucket when min_ms <= frametime < max_ms.
# Ranges are chosen so the ideal frametime (16.67ms / 8.33ms) lands in "Perfect".
HISTOGRAM_BUCKETS = {
    60: [
        (0, 18, "Perfect", (93, 232, 130)),      # Green - includes ideal 16.67ms
        (18, 25, "Good", (93, 232, 130)),        # Green - still smooth
        (25, 35, "Fair", (0, 255, 255)),         # Yellow - noticeable
        (35, 50, "Stutter", (0, 165, 255)),      # Orange - severe
        (50, 999, "Bad", (0, 0, 255)),           # Red - critical
    ],
    120: [
        (0, 9, "Perfect", (93, 232, 130)),       # Green - includes ideal 8.33ms
        (9, 12, "Good", (93, 232, 130)),         # Green - still smooth
        (12, 17, "Fair", (0, 255, 255)),         # Yellow - noticeable
        (17, 25, "Stutter", (0, 165, 255)),      # Orange - severe
        (25, 999, "Bad", (0, 0, 255)),           # Red - critical
    ],
}


def get_histogram_buckets(target_fps):
    """Return the histogram bucket definitions for a target FPS (60 or 120)."""
    return HISTOGRAM_BUCKETS[target_fps]


def categorize_frametime(frametime_ms, buckets):
    """Return the index of the bucket a frametime falls in, or None if outside all buckets."""
    for i, (min_ft, max_ft, _label, _color) in enumerate(buckets):
        if min_ft <= frametime_ms < max_ft:
            return i
    return None


def count_frames_per_bucket(frametimes, buckets):
    """Count how many frametimes fall into each bucket."""
    counts = [0] * len(buckets)
    for ft in frametimes:
        i = categorize_frametime(ft, buckets)
        if i is not None:
            counts[i] += 1
    return counts


class FpsCounter:
    """Rolling one-second window of frame flags (1 = unique frame, 0 = repeated frame).

    The window holds target_fps samples, so it always spans one second of video
    and the sum of the window is the current FPS.
    """

    def __init__(self, target_fps):
        self.window = [0] * target_fps

    def record(self, unique):
        """Record one video frame and return the updated FPS."""
        self.window.pop(0)
        self.window.append(1 if unique else 0)
        return sum(self.window)
