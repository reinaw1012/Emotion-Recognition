#!/usr/bin/python3
# -*- coding: utf-8 -*-

#MIT License
#
#Copyright (c) 2018 Iván de Paz Centeno
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# IMPORTANT:
#
# This code is derivated from the MTCNN implementation of David Sandberg for Facenet
# (https://github.com/davidsandberg/facenet/)
# It has been rebuilt from scratch, taking the David Sandberg's implementation as a reference.
# The code improves the readibility, fixes several mistakes in the definition of the network (layer names)
# and provides the keypoints of faces as outputs along with the bounding boxes.
#

import cv2
import numpy as np
import pkg_resources
import tensorflow as tf
from mtcnn.layer_factory import LayerFactory
from mtcnn.network import Network
from mtcnn.exceptions import InvalidImage

__author__ = "Iván de Paz Centeno"


class PNet(Network):
    """
    Network to propose areas with faces.
    """
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, None, None, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=10, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=16, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_conv(name='conv4-1', kernel_size=(1, 1), channels_output=2, stride_size=(1, 1), relu=False)
        layer_factory.new_softmax(name='prob1', axis=3)

        layer_factory.new_conv(name='conv4-2', kernel_size=(1, 1), channels_output=4, stride_size=(1, 1),
                               input_layer_name='prelu3', relu=False)

    def _feed(self, image):
        print('Running Pnet!')
        return self._session.run(['pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'], feed_dict={'pnet/input:0': image})


class RNet(Network):
    """
    Network to refine the areas proposed by PNet
    """

    def _config(self):

        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 24, 24, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=28, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=48, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(2, 2), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_fully_connected(name='fc1', output_count=128, relu=False)  # shouldn't the name be "fc1"?
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)   # shouldn't the name be "fc2-1"?
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu4')

    def _feed(self, image):
        print('Running Rnet!')
        return self._session.run(['rnet/fc2-2/fc2-2:0', 'rnet/prob1:0'], feed_dict={'rnet/input:0': image})


class ONet(Network):
    """
    Network to retrieve the keypoints
    """
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 48, 48, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_max_pool(name='pool3', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv4', kernel_size=(2, 2), channels_output=128, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc1', output_count=256, relu=False)
        layer_factory.new_prelu(name='prelu5')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu5')

        layer_factory.new_fully_connected(name='fc2-3', output_count=10, relu=False, input_layer_name='prelu5')

    def _feed(self, image):
        print('Running Onet!')
        return self._session.run(['onet/fc2-2/fc2-2:0', 'onet/fc2-3/fc2-3:0', 'onet/prob1:0'],
                                 feed_dict={'onet/input:0': image})


class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """
    def __init__(self, pad_result: tuple=None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result


class MTCNN(object):
    """
    Allows to perform MTCNN Detection ->
        a) Detection of faces (with the confidence probability)
        b) Detection of keypoints (left eye, right eye, nose, mouth_left, mouth_right)
    """

    def __init__(self, weights_file: str=None, min_face_size: int=20, steps_threshold: list=None,
                 scale_factor: float=0.709):
        """
        Initializes the MTCNN.
        :param weights_file: file uri with the weights of the P, R and O networks from MTCNN. By default it will load
        the ones bundled with the package.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]

        if weights_file is None:
            weights_file = pkg_resources.resource_stream('mtcnn', 'data/mtcnn_modify_weights.npy')

        self.__min_face_size = min_face_size
        self.__steps_threshold = steps_threshold
        self.__scale_factor = scale_factor

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__session = tf.Session(config=config, graph=self.__graph)

            weights = np.load(weights_file).item()
            self.__pnet = PNet(self.__session, False)
            self.__pnet.set_weights(weights['PNet'])

            self.__rnet = RNet(self.__session, False)
            self.__rnet.set_weights(weights['RNet'])

            self.__onet = ONet(self.__session, False)
            self.__onet.set_weights(weights['ONet'])

            self.__train_writer = tf.summary.FileWriter('tfboard_test', self.__graph)

    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self.__scale_factor, factor_count)]
            min_layer = min_layer * self.__scale_factor
            factor_count += 1

        return scales

    @staticmethod
    def __scale_image(image, scale: float):
        """
        Scales the image to a given scale.
        :param image:
        :param scale:
        :return:
        """
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    @staticmethod
    def __generate_bounding_box(imap, reg, scale, t):
        #t is threshold, the cut-off percentage for face-like identification
        #imap is the percentage of whether or not it is like a face
        #reg is the coordinates of the bounding box, with 3 dimensions: 
        #   the (x,y) coordinate of the 12x12 kernel, and the location of the bounding box within that kernel
        #scale is the list of scales to resize the image

        # use heatmap to generate bounding boxes
        stride = 2
        cellsize = 12

        #Get an array of bounding box coordinates (for each shift of the 12x12 kernel)
        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:, :, 0])
        dy1 = np.transpose(reg[:, :, 1])
        dx2 = np.transpose(reg[:, :, 2])
        dy2 = np.transpose(reg[:, :, 3])

        #Create an index of which bounding boxes have a face in it
        y, x = np.where(imap >= t)

        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)

        #Create an array of which bounding boxes have which percentage certainty
        score = imap[(y, x)]

        #Using the index, pick out the bounding boxes certain with faces in it
        #reg = [[dx1, dy1, dx2, dy2],[dx1, dy1, dx2, dy2]...]
        reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

        if reg.size == 0:
            reg = np.empty(shape=(0, 3))

        #Index of how many strides the 12x12 kernel has taken
        bb = np.transpose(np.vstack([y, x]))

        #Convert the inside bounding boxes to real image size
        #q1 is top left corner of 12x12 kernel, q2 is bottom right corner
        q1 = np.fix((stride * bb + 1)/scale)
        q2 = np.fix((stride * bb + cellsize)/scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

        return boundingbox, reg

    @staticmethod
    def __nms(boxes, threshold, method):
        """
        Non Maximum Suppression.

        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        #x1 and y1 are values from q1, top left corner of the 12x12 kernel, resized down to the real image
        #x2 and y2 are values from q2, bottom right corner
        #s is score
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            #x1,x2,y1,y2[i] is largest bounding box
            #x1,x2,y1,y2[idx] is all other bounding boxes
            #Pick box with highest percentage
            i = sorted_s[-1]
            #Set up counter
            pick[counter] = i
            counter += 1
            #Create array with rest of the boxes
            idx = sorted_s[0:-1]

            #Picking intersection:
            #xx1,yy1 is top left corner
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            #xx2,yy2 is bottom right corner
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            #Intersection width and height
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            #Intersection area
            inter = w * h

            if method is 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                #Using this, since method is 'Union'
                #Percentage of intersection over total area
                o = inter / (area[i] + area[idx] - inter)
            #If there is too much intersection (i.e. percentage is too high), it gets deleted from sorted_s
            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    @staticmethod
    def __pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        #If the bounding box is too much to the right, shift bottom right corner of the bounding box in to the screen
        #ex = ex - (ex-w) = w
        #Shift edx in the same way: by -(ex-w)
        #Original edx (bottom right corner based on bounding box) is tmpw, then shift to the left by -ex+w
        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        #If the bounding box is too much to the left, shift top left corner of the bounding box in to the screen
        #x (originally negative) will become 1
        #dx will move over approximately the amount x moved by
        #For example if x was -50, then dx will shift by 2-(-50)=52
        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1


        #dx, dy are the coordinates of the top left corner of the bounding box BASED ON COORDINATES OF THE BOUNDING BOX
        #edx and edy are the coordinates of the bottom right corner of the bounding box BASED ON COORDINATES OF THE BOUNDING BOX
        #x, y are the coordinates of the top left corner of the bounding box BASED ON THE IMAGE
        #ex and ey are the coordinates of the bottom right corner of the bounding box BASED ON THE IMAGE
        #tmph and tmpw are original bounding box width and heigh
        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def __rerec(bbox):
        # convert bbox to square
        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        l = np.maximum(w, h)
        #Shift top left corner up/left
        bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
        #Expand from top left corner
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
        return bbox

    @staticmethod
    def __bbreg(boundingbox, reg):
        # calibrate bounding boxes
        #boundingbox=pnet result
        #reg=rnet result
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        #stack vertically then turn, so that there are numerous rows of [x1, y1, x2, y2]
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox

    def detect_faces(self, img) -> list:
        """
        Detects bounding boxes from the specified image.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """
        if img is None or not hasattr(img, "shape"):
            raise InvalidImage("Image not valid.")

        height, width, _ = img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self.__min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__stage1, self.__stage2, self.__stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in stages:
            result = stage(img, result[0], result[1])

        [total_boxes, points] = result

        #print("total_boxes:",total_boxes)
        #print("points:", points)

        #total_boxes: [x1,y1,x2,y2,score],[x1,y1,x2,y2,score]
        #points: [x1,y1],[x2,y2]...

        bounding_boxes = []
        #print("zip:", zip(total_boxes, points.T))

        for bounding_box, keypoints in zip(total_boxes, points.T):

            bounding_boxes.append({
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1],
                    'keypoints': {
                        'left_eye': (int(keypoints[0]), int(keypoints[5])),
                        'right_eye': (int(keypoints[1]), int(keypoints[6])),
                        'nose': (int(keypoints[2]), int(keypoints[7])),
                        'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                        'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                    }
                }
            )

        return bounding_boxes

    def __stage1(self, image, scales: list, stage_status: StageStatus):
        """
        First stage of the MTCNN.
        :param image:
        :param scales:
        :param stage_status:
        :return:
        """
        total_boxes = np.empty((0, 9))
        status = stage_status

        for scale in scales:
            scaled_image = self.__scale_image(image, scale)

            #ADDITIONAL:
            #Scaled image: 3 dimensions, heigh, width, rgb
            #print("Scaled image[0][0]:" , scaled_image[0][0][0])

            #Add 1 dimension
            img_x = np.expand_dims(scaled_image, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))

            #ADDITIONAL:
           #print("Transposed image:",img_y[0][0][0][0])

            out = self.__pnet.feed(img_y)
            #ADDITIONAL:
            #print("Pnet out:", out)

            #4 dimensions in total: x coordinate of 12x12 kernel, y coordinate of kernel, and percentage+coordinates of bounding box
            #out0 is the coordinates of the bounding boxes
            #out1 is the percentage of whether or not it is a face

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))
            #ADDITIONAL:
            #print("out[0]:",out0.shape())
            #print("out[1]:",out1.shape())

            #Generates bounding box, returns coordinates of 12x12 kernel, score array, and bounding box
            boxes, _ = self.__generate_bounding_box(out1[0, :, :, 1].copy(),
                                                    out0[0, :, :, :].copy(), scale, self.__steps_threshold[0])

            # inter-scale nms, takes the 12x12 kernel with the highest score (probability) and gets rid of overlapping ones
            pick = self.__nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                #deletes everything not in pick
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numboxes = total_boxes.shape[0]

        if numboxes > 0:
            pick = self.__nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]

            #Total_boxes: 0,1-q1, 2,3-q2, 4-score, 5,6,7,8-reg (between 0 and 1)
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]

            #Coordinates of bounding boxes
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = self.__rerec(total_boxes.copy())

            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

        return total_boxes, status

    def __stage2(self, img, total_boxes, stage_status:StageStatus):
        """
        Second stage of the MTCNN.
        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """

        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, stage_status

        # second stage
        tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

        #for each bounding box:
        for k in range(0, num_boxes):
            tmp = np.zeros((int(stage_status.tmph[k]), int(stage_status.tmpw[k]), 3))

            #Take the portion of the image within the bounding box and copy the image over to tmp, the bounding box
            #If bounding box out of bounds, only copy the portion of image within bounds and pad everything else with 0
            tmp[stage_status.dy[k] - 1:stage_status.edy[k], stage_status.dx[k] - 1:stage_status.edx[k], :] = \
                img[stage_status.y[k] - 1:stage_status.ey[k], stage_status.x[k] - 1:stage_status.ex[k], :]

            #resize tmp to 24x24 and pass into tempimg
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)

            else:
                return np.empty(shape=(0,)), stage_status


        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        #feed the 24x24 bounding box images into rnet
        out = self.__rnet.feed(tempimg1)

        #print("Rnet out:",out)

        #out0 = coordinates, out1=probability
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])

        score = out1[1, :]

        ipass = np.where(score > self.__steps_threshold[1])

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            pick = self.__nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = self.__rerec(total_boxes.copy())

        return total_boxes, stage_status

    def __stage3(self, img, total_boxes, stage_status: StageStatus):
        """
        Third stage of the MTCNN.

        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """
        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, np.empty(shape=(0,))

        total_boxes = np.fix(total_boxes).astype(np.int32)

        status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                             width=stage_status.width, height=stage_status.height)

        tempimg = np.zeros((48, 48, 3, num_boxes))

        for k in range(0, num_boxes):

            tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

            tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                return np.empty(shape=(0,)), np.empty(shape=(0,))

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self.__onet.feed(tempimg1)
        #print("Onet out:", out)
        #out0 = coordinates of bounding boxes
        #out1 = facial landmark coordinates
        #out2 = probabilities/scores
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])

        score = out2[1, :]

        points = out1

        ipass = np.where(score > self.__steps_threshold[2])

        points = points[:, ipass[0]]

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1

        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1

        if total_boxes.shape[0] > 0:
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
            pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            #points=[x1,y1],[x2,y2]...
            points = points[:, pick]

        return total_boxes, points

    def __del__(self):
        #self.__train_writer.close()
        self.__session.close()
