import cv2
import numpy as np

def preprocess(frame, input_width, input_height):
   img = cv2.resize(frame, (input_width, input_height))
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = img.astype(np.float32) / 255.0
   img = np.transpose(img, (2, 0, 1))
   img = np.expand_dims(img, axis=0)


   return np.ascontiguousarray(img)
