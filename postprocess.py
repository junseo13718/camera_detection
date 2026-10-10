import numpy as np
import cv2

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

def postprocess(output, frame_shape, names, input_width, input_height):


   frame_height, frame_width = frame_shape[:2]


   scale_x = frame_width / input_width
   scale_y = frame_height / input_height


   if output.shape[1] == 84:
       predictions = output[0].T


   elif output.shape[2] == 84:
       predictions = output[0]


   else:
       raise ValueError(
           f"지원하지 않는 출력 형태: {output.shape}"
       )


   boxes = []
   confidences = []
   class_ids = []


   for prediction in predictions:


       class_scores = prediction[4:]


       class_id = int(np.argmax(class_scores))
       confidence = float(class_scores[class_id])


       if confidence < CONF_THRESHOLD:
           continue


       cx, cy, w, h = prediction[:4]


       x1 = int((cx - w / 2) * scale_x)
       y1 = int((cy - h / 2) * scale_y)


       x2 = int((cx + w / 2) * scale_x)
       y2 = int((cy + h / 2) * scale_y)


       x1 = max(0, min(x1, frame_width - 1))
       y1 = max(0, min(y1, frame_height - 1))


       x2 = max(0, min(x2, frame_width - 1))
       y2 = max(0, min(y2, frame_height - 1))


       if x2 <= x1 or y2 <= y1:
           continue


       boxes.append([
           x1,
           y1,
           x2 - x1,
           y2 - y1
       ])


       confidences.append(confidence)
       class_ids.append(class_id)


   detections = []


   for class_id in set(class_ids):


       selected = [
           i for i, value in enumerate(class_ids)
           if value == class_id
       ]


       class_boxes = [boxes[i] for i in selected]
       class_scores = [confidences[i] for i in selected]


       indices = cv2.dnn.NMSBoxes(
           class_boxes,
           class_scores,
           CONF_THRESHOLD,
           NMS_THRESHOLD
       )


       for index in np.array(indices).reshape(-1):


           original_index = selected[int(index)]


           detections.append((
               boxes[original_index],
               confidences[original_index],
               names[class_id]
           ))


   return detections