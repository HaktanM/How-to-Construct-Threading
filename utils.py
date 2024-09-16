import cv2


class VideoReader:
    def __init__(self, video_path, start_time, end_time) -> None:
        # Open the video file
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # Get frames per second (fps) and total number of frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * 60 * self.fps)  # Which minute to start
        end_frame = int(end_time * 60 * self.fps)      # Which minute to stop

        # Ensure that the end frame does not exceed the total frames
        self.end_frame = min(end_frame, self.total_frames)

        # Set the video to start at the 10th minute
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        self.frameCounter = start_frame
        
    def getNextFrame(self):
        if self.frameCounter < self.end_frame:
            ret, frame = self.cap.read()
            self.frameCounter = self.frameCounter + 1
            return ret, frame
        else:
            print("HAS RETUEND FALSE IN VIDEO READER")
            return False, None
        
class VideoCreator:
    def __init__(self, video_name ,fps) -> None:
        self.initialized = False
        self.video = None
        self.video_name = video_name
        self.fps = fps
    
    def feedFrame(self, frame):
        if self.initialized:
            self.video.write(frame)
        else:
            height, width, layers = frame.shape
            self.video = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            self.initialized = True
            
    def release(self):
        self.video.release()
            