# MIT License
#
# Copyright (c) 2017 PXL University College
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Clusters similar faces from input folder together in folders based on euclidean distance matrix

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import facenet
import align.detect_face
from sklearn.cluster import DBSCAN
import cv2

class FaceRecognition():
    def __init__(self, args):
        self.image_size = args.image_size
        self.margin = args.margin
        self.create_network_face_detection(args.gpu_memory_fraction)
        graph = tf.Graph()
        with graph.as_default(): 
            self.sess = tf.Session()
            with self.sess.as_default():
                facenet.load_model(args.model)
            self.images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
            self.embeddings_op = self.sess.graph.get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

    def create_network_face_detection(self, gpu_memory_fraction):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

    def face_encodings(self, img): 
        images = self.align_data(img)
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        try:
            emb = self.sess.run(self.embeddings_op, feed_dict=feed_dict)
        except:
            return []
        return [e for e in emb]

    def face_distance(self, e1, e2):
        d = np.sum((e1-e2)**2)**0.5
        return d 

    def compare_faces(self, encoding_list, encoding, max_distance=1.0):
        distance_data = np.array([self.face_distance(e,encoding) for e in encoding_list])
        min_distance = np.min(distance_data)
        min_distance_idx = np.argmin(distance_data)
        if min_distance < max_distance: return min_distance_idx
        else: return -1
        

    def detect_face(self, src_img):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        bounding_boxes, key_points = align.detect_face.detect_face(src_img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        return bounding_boxes, key_points

    def align_data(self, src_img):
        img_list = []
        img_size = np.asarray(src_img.shape)[0:2]
        bounding_boxes, key_points = self.detect_face(src_img)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                    bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                    bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                    cropped = src_img[bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = cv2.resize(cropped, dsize=(self.image_size, self.image_size))
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)
        if len(img_list) > 0:
            images = np.stack(img_list)
            return images
        else:
            return None


    def load_image_file(self,imgfile):
        img = cv2.imread(imgfile)
        return img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=1)
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--largest_cluster_only', action='store_true',
                        help='This argument will make that only the biggest cluster is saved.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)


fr = FaceRecognition(parse_arguments(sys.argv[1:]))

def load_known_img(known_dir='/data/face/known'):
    encoding_list = []
    label_list = []
    for label in os.listdir(known_dir):
        sum_encoding = np.zeros([512],dtype=np.float32)
        num = 0.0
        for img in os.listdir('%s/%s'%(known_dir,label)):
            img = fr.load_image_file("%s/%s/%s"%(known_dir,label,img))
            encodings = fr.face_encodings(img)
            if len(encodings) == 1:
                sum_encoding += encodings[0]
                num += 1
        encoding_list.append(sum_encoding/num)
        label_list.append(label)
    return encoding_list, label_list

encoding_list,label_list = load_known_img()

def recognition(img_file):
    img = fr.load_image_file(img_file)
    encodings = fr.face_encodings(img)
    labels=[]
    for encoding in encodings:
        idx = fr.compare_faces(encoding_list, encoding, max_distance=1.0)
        if idx > -1:
            label = label_list[idx]
            labels.append(label)
    return labels

if __name__=='__main__':
    for img_file in os.listdir('/data/face/unknown'):
        labels = recognition('/data/face/unknown/%s'%img_file)
        print(img_file, labels)


