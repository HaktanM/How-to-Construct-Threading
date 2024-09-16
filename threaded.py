from ultralytics import YOLO
import cv2
import utils
import time
import threading


class ResultBuffer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.results = []
    def append(self,result):
        # Lock the array
        self.lock.acquire()
        try:
            self.results.append(result)
        finally:
            # Unlock the array
            self.lock.release()
            
    def readNext(self):
        self.lock.acquire()
        try:
            if len(self.results)>0:
                # Read and delete the first item
                result = self.results.pop(0)
            else:
                result = None
        finally:
            # Unlock the array
            self.lock.release()
            
        return result
    
    
class Manager:
    def __init__(self) -> None:
        # Load the video
        self.video = utils.VideoReader("cars.mp4", 11, 12)
        
        # Save the output
        self.video_out = utils.VideoCreator("annotated_cars_threaded.mp4",self.video.fps)

        # Load YOLOv8 model (you can also load YOLOv5 similarly)
        self.model = YOLO('yolov8n.pt')  # You can choose 'yolov8n.pt' for YOLOv8
        
        # Result Buffer
        self.result_buffer = ResultBuffer()
        
        # Flag if YOLO thread is ended
        self.yolo_finished = False

        # Flag if Tracker thread is ended
        self.track_finished = False
    
    def yolo_thread(self):
        while True:
            while len(self.result_buffer.results)>100:
                time.sleep(0.1)
                
            ret, frame = self.video.getNextFrame()
            if not ret:
                print(f"YOLO THREAD IS ENDED")
                break
            else:
                result = self.model(frame, verbose = False)
                self.result_buffer.append(result)
        self.yolo_finished = True
            
    def tracker_thread(self):
        while True:
            result = self.result_buffer.readNext()            
            if result is not None:
                annotated_frame = result[0].plot()
                self.video_out.feedFrame(annotated_frame)
                time.sleep(0.01)
            else:
                if self.yolo_finished: # If result is None and yolo thread is ended, we can end the track thread as well
                    break
                time.sleep(0.1)  # Wait if there's no result
                
        self.video_out.release()
        self.track_finished = True 
    
if __name__ == "__main__":
    
    manager = Manager()
    
    start_time = time.monotonic()
    
    # Create threads
    producer_thread = threading.Thread(target=manager.yolo_thread, args=())
    consumer_thread = threading.Thread(target=manager.tracker_thread, args=())
    
    # Start threads
    producer_thread.start()
    consumer_thread.start()
    
    # Wait for threads to finish
    producer_thread.join()
    consumer_thread.join()
    
    while True:
        if manager.track_finished:
            break
        else:
            time.sleep(0.1)
            
    stop_time = time.monotonic()
    elapsed_time = stop_time - start_time
    print(f"Elapsed Time : {elapsed_time}")