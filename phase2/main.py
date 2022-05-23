import lane
import yolo
import sys
from moviepy.editor import VideoFileClip

def lane_and_yolo(img):    
    img = lane.vid_pipeline(img)
    img = yolo.detect_yolo(img)
    return img

def lane_and_yolo_pipeline(input_file, output_file, tiny):
    yolo.load_weights(tiny)
    project_video = VideoFileClip(input_file)
    white_clip = project_video.fl_image(lane_and_yolo) 
    white_clip.write_videofile(output_file, audio=False, threads=8, preset='ultrafast')

if __name__ == "__main__":
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    mode = sys.argv[1]    
    if(mode == '--both'):
        lane_and_yolo_pipeline(input_file, output_file, False)
    elif(mode == '--both-tiny'):
        lane_and_yolo_pipeline(input_file, output_file, True)
    elif(mode == '--yolo'):
        yolo.pipeline_yolo_only(input_file, output_file, False)
    elif(mode == '--yolo-tiny'):
        yolo.pipeline_yolo_only(input_file, output_file, True)
    elif(mode == '--lane-debug'):
        lane.pipeline_lane_only(input_file, output_file, 1)
    elif(mode == '--lane-no-debug'):
        lane.pipeline_lane_only(input_file, output_file, 0)


