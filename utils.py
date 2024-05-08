import os
import cv2
import math
import numpy as np

class dataset:
    def __init__(self):
        self._path = ""
        self.src_path = ""
        self.padded_path = ""

    class Video:
        def __init__(self) -> None:
            pass
        
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