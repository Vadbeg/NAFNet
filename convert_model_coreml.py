

import coremltools as ct
import torch
import numpy as np
from coremltools.models.neural_network import quantization_utils

from convert_model_torchscript import imread, img2tensor


if __name__ == '__main__':
    model_script = torch.jit.load('nafnet_reds_64_torchscript.pt')

    shapes = [(1, 3, 256 * i, 256 * i) for i in range(1, 5)]
    input_a_shape = ct.EnumeratedShapes(shapes=shapes)

    model_coreml = ct.convert(
        model_script,
        inputs=[
            ct.ImageType(shape=input_a_shape, scale=1/255.),
        ]
    )
    print("Finished model conversion to CoreML...")

    spec = model_coreml.get_spec()

    current_input_names = model_coreml.input_description._fd_spec

    old_input_name = current_input_names[0].name
    new_input_name = 'image'
    ct.utils.rename_feature(
        spec, old_input_name, new_input_name, rename_outputs=True
    )

    current_output_names = model_coreml.output_description._fd_spec

    old_name = current_output_names[0].name
    new_name = 'result'
    ct.utils.rename_feature(
        spec, old_name, new_name, rename_outputs=True
    )
    new_model = ct.models.MLModel(spec)

    print("Testing model inferece...")
    input_path = 'demo_input/plates.png'
    img_input = imread(input_path)
    inp1 = img2tensor(img_input)
    inp1 = inp1.unsqueeze(dim=0)
    inp1_numpy = inp1.detach().cpu().numpy()
    print("Saving model...")

    new_model.save("nafnet_reds_64_fp32.mlmodel")