import onnxruntime as rt
import numpy as np
ort_session = rt.InferenceSession("mymodel.onnx")
input_names = [i.name for i in ort_session.get_inputs()]
output_names = [i.name for i in ort_session.get_outputs()]
batch = {input_names[0]: np.random.randn(1, 1, 28, 28).astype(np.float32)}
out = ort_session.run(output_names, batch)
print(out)

import onnxruntime as rt
sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "optimized_model.onnx"

session = rt.InferenceSession("mymodel.onnx", sess_options)