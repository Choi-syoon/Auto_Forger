import os
import cv2
import math
import numpy as np
import mediapipe as mp

class dataset:
    def __init__(self):
        self._path = ""
        self.src_path = ""
        self.padded_path = ""

    class Video:
        def __init__(self) -> None:
            pass

        def keypoint_extraction(self, filename):
            full_filename = os.path.join(self.PADDED_SAVE_PATH, filename)
            cap = cv2.VideoCapture(full_filename)

            if self.MODEL == "pose":
                solution = mp.solutions.pose.Pose
            elif self.MODEL == "holistic":
                solution = mp.solutions.holistic.Holistic
            else:
                raise ValueError("Invalid Model")
            with solution(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                    while True:
                        opened, image = cap.read()
                        if not opened:
                            break

                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)

                        if self.MODEL == "pose" and results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                self.keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                            self.keypoints_list.append(self.keypoints)

                            np.save(f"{self.FEATURE_SAVE_PATH}/{os.path.splitext(os.path.basename(filename))[0]}_pose.npy", self.keypoints_list) 

                        elif self.MODEL == "holistic" and (results.face_landmarks or
                                                      results.left_hand_landmarks or
                                                      results.right_hand_landmarks or
                                                      results.pose_landmarks):
                            keypoints = {'face': [],
                                         'left_hand': [],
                                         'right': [],
                                         'pose': []}

                            if results.pose_landmarks:
                                for landmark in results.pose_landmarks.landmark:
                                    keypoints['pose'].append([
                                        landmark.x, landmark.y, landmark.z
                                    ])

                            if results.face_landmarks:
                                for landmark in results.face_landmarks.landmark:
                                    keypoints['face'].append([
                                        landmark.x, landmark.y, landmark.z
                                    ])

                            if results.left_hand_landmarks:
                                for landmark in results.left_hand_landmarks.landmark:
                                    keypoints['left_hand'].append([
                                        landmark.x, landmark.y, landmark.z
                                    ])

                            if results.right_hand_landmarks:
                                for landmark in results.right_hand_landmarks.landmark:
                                    keypoints['right_hand'].append([
                                        landmark.x, landmark.y, landmark.z
                                    ])

                            self.keypoints_list.append(keypoints)
                            np.save(f"{self.FEATURE_SAVE_PATH}/{os.path.splitext(os.path.basename(filename))[0]}_holistic.npy", self.keypoints_list)

                    cap.release()

        class Pad:
            def video_length(self, mode=""):    # min, max mode
                current_len = 0
                target_len = 0
                target_file = ""
                
                if mode == 'max':
                    comparison_op = lambda x, y: x >= y
                    target_len = 0
                elif mode == 'min':
                    comparison_op = lambda x, y: x <= y
                    target_len = float('inf')
                else:
                    raise ValueError("Invalid mode. Please use 'max' or 'min'.")

                for filename in os.listdir(self._path):
                    __full_path = os.path.join(self.src_path, filename)
                    cap = cv2.VideoCapture(__full_path)
                    current_len = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
                    
                    if comparison_op(current_len, target_len):
                        target_len = current_len
                        target_file = filename

                return target_len, target_file
            
        