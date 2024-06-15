## Importing Relevant Packages
import cv2
import pickle
import os
import supervision as sv
from ultralytics import YOLO
import numpy as np


## READ VIDEO- presented 24fps
def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    
    return frames

## SAVE VIDEO
def save_video(output_video_frames,output_video_path):
    fourcc= cv2.VideoWriter_fourcc(*'XVID')

    #Width and height of frame
    out=cv2.VideoWriter(output_video_path,fourcc,24,(output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

## Other ADDITINAL FUNCTIONS
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2] -bbox[0]

## CLASS- TRACKER (WITH ONE CONSTRUCTOR AND ALL TRACKING FUNCTIONS)
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

## FRAME DETECTION
    def detect_frames(self, frames):
        batch_size=20
        detections=[]

        for i in range(0,len(frames),batch_size):
            detections_batch= self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch 
        return detections
    
## OBJECT TRACKING
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections=self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names 
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #Convert detections to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to player object#
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            
            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3].tolist()
                track_id = frame_detection[4].tolist()

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3].tolist()

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)



        return tracks

## ANNOTATIONS AND DRAWINGS
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width=40
        rectangle_height=20
        x1_rect= x_center - rectangle_width/2
        x2_rect= x_center + rectangle_width/2
        y1_rect= (y2 - rectangle_height//2) +15
        y2_rect= (y2 + rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                            (int(x1_rect),int(y1_rect)),
                            (int(x2_rect),int(y2_rect)),
                            (235, 232, 234),
                            cv2.FILLED
                            )
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, #Font Size
                (0,0,0), #Color Black
                2 #Thickness
            )

        return frame

    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])

        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame

    def draw_annotations(self,video_frames,tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"],(255,255,255),track_id)

            #Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))


            #Draw Ball
            for track_id, ball in ball_dict.items():
                frame= self.draw_triangle(frame, ball["bbox"],(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames


## MAIN- FIRST FUNCTION THAT IS CALLED AND CALLS OTHER FUNCTIONS AND CLASSES.
def main():
    
    # Read video
    video_frames=read_video('input_videos/FA_Cup_2024.mp4')

    #Initialize Tracker
    tracker= Tracker('models/best.pt')

    tracks= tracker.get_object_tracks(video_frames,
                                      read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')
    
    #Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    #Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()