import MNN
import numpy as np

interpreter = MNN.Interpreter("yolo12n_320.mnn")
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, input_data, MNN.Tensor_DimensionType_Caffe)
input_tensor.copyFrom(tmp_input)

# Run inference
interpreter.runSession(session)
output_tensor = interpreter.getSessionOutput(session)
print("Inference successful!")