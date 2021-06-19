
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

from PIL import Image

from object_detection import finding_object, object_detection


# Video Stream:
video_source = "Webcam" #"Ip Esp32 Stream" # Webcam
process_type = "Depth sensing" # Object detection Depth sensing

if(video_source == "Ip Esp32 Stream"):
    host = "192.168.1.159"  # ESP32 IP in local network
    port = 80  # ESP32 Server Port
    stream_port = 81

# From CAMERA
if (video_source == "Webcam"):

    cap = cv2.VideoCapture(1)
    ok, frame = cap.read()

if (video_source == "Ip Esp32 Stream"):

    cap = cv2.VideoCapture('http://' + host + ':' + str(stream_port) + '/stream')
    ok, frame = cap.read()
    print("Reading video stream from " + 'http://' + host + ':' + str(stream_port) + '/stream' + " - OK")


# Depth sensing params
if(process_type == "Object detection"):
    original_height, original_width = frame.shape[:2]
    feed_height = 96
    feed_width = 320
    # feed_height = 192
    # feed_width = 640
    # using cpu
    ctx = mx.cpu(0)
    model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_mono_640x192', #monodepth2_resnet18_kitti_stereo_640x192 monodepth2_resnet18_posenet_kitti_mono_640x192
                                        pretrained_base=False, ctx=ctx, pretrained=True)


if(process_type == "Object detection"): object = object_detection(name='helmet')
count = 0
while True:

    ret, frame = cap.read()

    original_height, original_width = frame.shape[:2]

    if(process_type == "Object detection"):
        # Get number of SIFT matches
        new_frame, object_found, object = finding_object(frame, 15, count, object)
        if object_found:
            cv2.putText(new_frame, 'Object Found', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow('Object Detector using SIFT', new_frame)

    if (process_type == "Depth sensing"):
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

    count += 1
    if (count == 10): count = 0

cap.release()
cv2.destroyAllWindows()