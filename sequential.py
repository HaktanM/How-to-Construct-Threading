from ultralytics import YOLO
import cv2
import utils
import time

path_to_input_video = ".../<video_name>.mp4"  # path to your favorite video

# Load the video
video = utils.VideoReader(path_to_input_video, 0, 1) # 0 1 represents the startin and stopping video time.

# Save the output
video_out = utils.VideoCreator("annotated_cars_seqeuntial.mp4", video.fps)

# Load YOLOv8 model (you can also load YOLOv5 similarly)
model = YOLO('yolov8n.pt')  # You can choose 'yolov8n.pt' for YOLOv8

start_time = time.monotonic()
while True:
    
    ret, frame = video.getNextFrame()
    if not ret:
        video_out.release()
        print(f"Error reading frame")
        break
    
    results = model(frame, verbose = False)
    annotated_frame = results[0].plot()
    
    video_out.feedFrame(annotated_frame)
    time.sleep(0.01) # This represent the time required for your complicated tracker



# # Release the video capture and close windows
video_out.release()
end_time = time.monotonic()
elpased_time = end_time - start_time




cv2.destroyAllWindows()

print(f"Total Elapsed Time : {elpased_time}")
