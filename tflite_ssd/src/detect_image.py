# Download from https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/python/label_image.py
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite 
from tflite_runtime.interpreter import load_delegate
import pdb

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


class SSDTflite(object):
    def __init__(self, 
        model_file="weights/detect.tflite",
        label_file="labelmap.txt",
      ):
        self.label_file = label_file 
        self.interpreter = tflite.Interpreter(model_path=model_file,
                            #experimental_delegates=[load_delegate('')] 
            )
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width  = self.input_details[0]['shape'][2]

    def __call__(self, img, output_name=None):
        img = img.resize((self.width, self.height))
        if len(img.size) < 4:
            # add N dim
            input_data = np.expand_dims(img, axis=0)
        else:
            input_data = img
        
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        start_time = time.time()
        self.interpreter.invoke()
        stop_time = time.time()
        
        # Obtain outputs, based on https://www.tensorflow.org/lite/models/object_detection/overview
        output_locs  = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_cls   = self.interpreter.get_tensor(self.output_details[1]['index'])
        output_scs   = self.interpreter.get_tensor(self.output_details[2]['index'])
        output_n_dcs = self.interpreter.get_tensor(self.output_details[3]['index'])
        
        # Obtain top_k (6)
        output_scs   = np.squeeze(output_scs)
        self.top_k   = output_scs.argsort()[-5:][::-1]
        labels       = load_labels(self.label_file)
        
        if output_name:
            self._saveResult(img, output_locs, output_cls, output_scs, labels, output_name) 
        

    def _saveResult(self, orig_img, locs, cls, scores, labels, output_name):
        # Draw boundaries
        draw = ImageDraw.Draw(img)
        #fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        orig_w, orig_h = img.size
        for i in self.top_k:
            box = locs[0][i]
            x0  = int(box[1] * orig_w)
            y0  = int(box[0] * orig_h)
            x1  = int(box[3] * orig_w)
            y1  = int(box[2] * orig_h)
            draw.rectangle([x0, y0, x1, y1], outline=(120, 250,120,128))    

        # Label 
        cl_name = labels[int(cls[0][i])]
        # Draw text 
        txt_x0 = min(max(x0,5), orig_w -20)
        txt_y0 = min(max(y0,5), orig_h -20)
        draw.text((txt_x0, txt_y0), cl_name, fill=(255,120,120,128)) 
        print('[Object] Prob: {:08.6f}: {}'.format(float(scores[i]), cl_name))
        print('--> Pos: {}\n'.format(locs[0, i]*self.width))

        img.save(output_name)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='/tmp/grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='/tmp/labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '-o',
      '--output',
      default='',
      help='Output image name')
  args = parser.parse_args()
  if not args.output:
    output_name = args.image.split("/")[-1]
    output_name = "output_" + output_name
  else:
    output_name = args.output 
  
  img = Image.open(args.image)

  for i in range(2000):
    print(i)
    ssd_model = SSDTflite(args.model_file, args.label_file)

    ssd_model(img, output_name)


