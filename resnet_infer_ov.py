from openvino.runtime import Core
import cv2 
import numpy as np
from PIL import Image

core = Core()
core.set_property({'CACHE_DIR': './cache'})
resnet50 = core.compile_model("resnet50.xml", "GPU.1")
output_node = resnet50.outputs[0]
# Resize
img = cv2.resize(cv2.imread("cat.jpg"), [224,224])
# Layout: HWC -> NCHW
blob = np.expand_dims(np.transpose(img, (2,0,1)), 0)

result = resnet50(blob)[output_node]
print(np.argmax(result))


