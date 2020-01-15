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

def main(args):
    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet.load_model(args.model)
            image_list, imgnames = load_images_from_folder(args.data_dir)
            images, cropped, name_list = align_data(image_list, imgnames, args.image_size, args.margin, pnet, rnet, onet)

            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            #计算tsne
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)
            tsne.fit_transform(emb)
            Y = 5*tsne.embedding_
            Y -= np.min(Y, axis=0)-100
            Y *= 800/np.max(Y,axis=0)
            Y = Y.astype(np.int32)
            print('-----------tsne:')
            print(Y)
            data = np.zeros([1000,1000,3], dtype=np.float32)
            for i in range(Y.shape[0]):
                r,c = Y[i]
                name = name_list[i]
                cv2.circle(data, (r,c), radius=1, color=(0,0,255), thickness=2)
                cv2.putText(data,name, (r,c), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color=(255,255,255))
            cv2.imwrite('tsne.jpg', data) 
                
            
            nrof_images = len(images)

            matrix = np.zeros((nrof_images, nrof_images))

            print('')
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                #print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    matrix[i][j] = dist
                    #print('  %1.4f  ' % dist, end='')
                #print('')

            print('')

            # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
            db = DBSCAN(eps=args.cluster_threshold, min_samples=args.min_cluster_size, metric='precomputed')
            db.fit(matrix)
            labels = db.labels_

            # get number of clusters
            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            print('No of clusters:', no_clusters)

            if no_clusters > 0:
                if args.largest_cluster_only:
                    largest_cluster = 0
                    for i in range(no_clusters):
                        print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                        if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                            largest_cluster = i
                    print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
                    cnt = 1
                    for i in np.nonzero(labels == largest_cluster)[0]:
                        cv2.imwrite(os.path.join(args.out_dir, str(cnt) + '.png'), cropped[i])
                        cnt += 1
                else:
                    print('Saving all clusters')
                    for i in range(no_clusters):
                        cnt = 1
                        print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                        path = os.path.join(args.out_dir, str(i))
                        if not os.path.exists(path):
                            os.makedirs(path)
                            for j in np.nonzero(labels == i)[0]:
                                cv2.imwrite(os.path.join(path, str(cnt) + '.png'), cropped[j])
                                cnt += 1
                        else:
                            for j in np.nonzero(labels == i)[0]:
                                cv2.imwrite(os.path.join(path, str(cnt) + '.png'), cropped[j])
                                cnt += 1


def align_data(image_list, imgnames, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    name_list = []
    cropped_list = []

    for x in range(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        img = image_list[x]
        name = imgnames[x]
        bounding_boxes, key_points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        SP = bounding_boxes.shape
        draw=img.copy()
        for i in range(SP[0]):
            b = bounding_boxes[i]
            cv2.rectangle(draw, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0,255,0), 2)
        for pt in key_points.transpose().reshape([-1,2,5]):
            for i in range(5):
                cv2.circle(draw, (pt[0,i],pt[1,i]), radius=1, color=(0,0,255), thickness=2)
        cv2.imwrite('rectangle/%s.jpg'%x, draw)        
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    cropped_list.append(cropped)
                    aligned = cv2.resize(cropped, dsize=(image_size, image_size))
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)
                    name_list.append(name)
    if len(img_list) > 0:
        images = np.stack(img_list)
        return (images,cropped_list,name_list)
    else:
        return None


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images = []
    imgnames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            imgnames.append(filename)
    return images, imgnames


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('data_dir', type=str,
                        help='The directory containing the images to cluster into folders.')
    parser.add_argument('out_dir', type=str,
                        help='The output directory where the image clusters will be saved.')
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


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
