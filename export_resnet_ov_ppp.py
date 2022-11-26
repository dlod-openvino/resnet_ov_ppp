from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type, serialize

# ========  Step 0: read original model =========
core = Core()
model = core.read_model("resnet50.onnx")

# ======== Step 1: Preprocessing ================
ppp = PrePostProcessor(model)
# Declare section of desired application's input format
ppp.input("images").tensor() \
    .set_element_type(Type.u8) \
    .set_spatial_dynamic_shape() \
    .set_layout(Layout('NHWC')) \
    .set_color_format(ColorFormat.BGR)

# Specify actual model layout
ppp.input("images").model().set_layout(Layout('NCHW'))

# Explicit preprocessing steps. Layout conversion will be done automatically as last step
ppp.input("images").preprocess() \
    .convert_element_type()     \
    .convert_color(ColorFormat.RGB) \
    .resize(ResizeAlgorithm.RESIZE_LINEAR) \
    .mean([123.675, 116.28, 103.53]) \
    .scale([58.624, 57.12, 57.375])

# Dump preprocessor
print(f'Dump preprocessor: {ppp}')
model = ppp.build()

# ======== Step 2: Save the model ================
serialize(model, 'resnet50_ppp.xml', 'resnet50_ppp.bin')