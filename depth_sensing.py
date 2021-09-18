"""01. Predict depth from a single image with pre-trained Monodepth2 models
===========================================================================

This is a quick demo of using GluonCV Monodepth2 model for KITTI on real-world images.
Please follow the `installation guide <../../index.html#installation>`__
to install MXNet and GluonCV if not yet.
"""
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv

import argparse
import time
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import scipy.misc

from PIL import Image


# using cpu
ctx = mx.cpu(0)

# Video Stream:
video_source = "Webcam"

# From CAMERA
if(video_source == "Webcam"):

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()


original_height, original_width = frame.shape[:2]
feed_height = 96
feed_width = 320
#feed_height = 192
#feed_width = 640


model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_mono_640x192', #monodepth2_resnet18_kitti_stereo_640x192 monodepth2_resnet18_posenet_kitti_mono_640x192
                                    pretrained_base=False, ctx=ctx, pretrained=True)

while True:
    # Read a new frame
    ok, frame = cap.read()

    raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(raw_img)

    img = img.resize((feed_width, feed_height), pil.LANCZOS)
    img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

    outputs = model.predict(img)
    disp = outputs[("disp", 0)]
    disp_resized = mx.nd.contrib.BilinearResize2D(disp, height=int(original_height), width=int(original_width))

    disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    depth_frame = np.asarray(im)
    output = np.concatenate((depth_frame, frame), axis=0)
    cv2.imshow('frame', output)

    k = cv2.waitKey(1)

    if k == 27:  # If escape was pressed exit
        cv2.destroyAllWindows()
        break