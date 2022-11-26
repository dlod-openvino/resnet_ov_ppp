from openvino.runtime import Core
import cv2 
import numpy as np

core = Core()
core.set_property({'CACHE_DIR': './cache/ppp'}) # 使用模型缓存技术

resnet50_ppp = core.compile_model("resnet50_ppp.xml", "CPU")
output_node = resnet50_ppp.outputs[0]

blob = np.expand_dims(cv2.imread("cat.jpg"),0)

result = resnet50_ppp(blob)[output_node]
print(np.argmax(result))


