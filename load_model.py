import ast
import onnxruntime as ort

def load_model():
   session = ort.InferenceSession(
       "yolo11n.onnx",
       providers=["CPUExecutionProvider"]
   )


   input_info = session.get_inputs()[0]
   output_info = session.get_outputs()[0]


   input_height = input_info.shape[2]
   input_width = input_info.shape[3]


   print("Input:", input_info.shape)
   print("Output:", output_info.shape)


   metadata = session.get_modelmeta().custom_metadata_map
   names = ast.literal_eval(metadata["names"])


   if isinstance(names, list):
       names = dict(enumerate(names))


   return session, input_info.name, names, input_width, input_height
