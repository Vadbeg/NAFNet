
import gdown
import torch

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
from basicsr.models.image_restoration_model import ImageRestorationModel
import numpy as np
import cv2
import matplotlib.pyplot as plt


def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1)
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('NAFNet output', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)


def single_image_inference(model, img, save_path):
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          print('use grids')
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      sr_img = tensor2img([visuals['result']])
      imwrite(sr_img, save_path)


def download_all_files():
    gdown.download(
        'https://drive.google.com/uc?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X',
        "./experiments/pretrained_models/",
        quiet=False
    )

    gdown.download(
        'https://drive.google.com/uc?id=1kWjrGsAvh4gOA_gn7rB9vnnQVfRINwEn',
        "demo_input/",
        quiet=False
    )
    gdown.download(
        'https://drive.google.com/uc?id=1xdfmGUKNDXtnWakyxcGq2nh8m18vHhSI',
        "demo_input/",
        quiet=False
    )


if __name__ == '__main__':
    download_files = False
    if download_files:
        download_all_files()

    opt_path = 'options/test/REDS/NAFNet-width64.yml'

    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet: ImageRestorationModel = create_model(opt)

    input_path = 'demo_input/blurry-reds-0.jpg'
    img_input = imread(input_path)
    img_input = cv2.resize(img_input, (256, 256))
    inp1 = img2tensor(img_input)
    inp1 = inp1.unsqueeze(dim=0)

    input_path = 'demo_input/blurry-reds-1.jpg'
    img_input = imread(input_path)
    img_input = cv2.resize(img_input, (512, 512))
    inp2 = img2tensor(img_input)
    inp2 = inp2.unsqueeze(dim=0)

    network = NAFNet.net_g
    network = network.eval()

    model_script = torch.jit.trace(
        network,
        example_inputs=inp1,
        check_inputs=[inp1]
    )

    result = network(inp2)
    result_torchscript = model_script(inp2)

    try:
        np.testing.assert_allclose(
            result_torchscript.detach().numpy(),
            result.detach().numpy(),
            rtol=1e-03,
            atol=1e-05
        )
    except AssertionError as e:
        print(e)
        print("Torchscript model failed")

    model_script.save('nafnet_reds_64_torchscript.pt')
