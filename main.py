import cv2

from load_model import load_model
from postprocess import postprocess
from preprocess import preprocess


def main():


   session, input_name, names, input_width, input_height = load_model()


   cap = cv2.VideoCapture(0)


   if not cap.isOpened():
       print("Cannot open camera")
       return


   print("Camera opened successfully")


   while True:


       ret, frame = cap.read()


       if not ret:
           print("Cannot read frame")
           break


       frame = cv2.flip(frame, 1)


       img = preprocess(
           frame,
           input_width,
           input_height
       )


       outputs = session.run(
           None,
           {input_name: img}
       )


       detections = postprocess(
           outputs[0],
           frame.shape,
           names,
           input_width,
           input_height
       )


       object_counts = {}


       for box, confidence, class_name in detections:


           x, y, w, h = box


           cv2.rectangle(
               frame,
               (x, y),
               (x + w, y + h),
               (255, 0, 0),
               2
           )


           cv2.putText(
               frame,
               f"{class_name} {confidence:.2f}",
               (x, max(20, y - 10)),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.6,
               (0, 255, 0),
               2
           )


           object_counts[class_name] = (
               object_counts.get(class_name, 0) + 1
           )


       cv2.putText(
           frame,
           f"Total Objects: {len(detections)}",
           (20, 40),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.9,
           (0, 0, 255),
           2
       )


       y = 80


       for name, count in object_counts.items():


           cv2.putText(
               frame,
               f"{name}: {count}",
               (20, y),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,
               (0, 255, 255),
               2
           )


           y += 30


       cv2.putText(
           frame,
           "Press Q to quit",
           (20, frame.shape[0] - 20),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255, 255, 255),
           2
       )


       cv2.imshow("YOLO ONNX Detection", frame)


       if cv2.waitKey(1) & 0xFF == ord("q"):
           break


   cap.release()
   cv2.destroyAllWindows()




if __name__ == "__main__":
   main()


