from styx_msgs.msg import TrafficLight
import rospkg
import tensorflow as tf
import os
import sys
import numpy as np
from functools import partial

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        # tensorflow localization/detection model
        r = rospkg.RosPack()
    	path = r.get_path('tl_detector')
        detect_model = path+'frozen_inference_graph.pb'

        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        # configuration for possible GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # load frozen tensorflow detection model and initialize
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # read model from the model file
            with tf.gfile.GFile(detect_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.tf_session = tf.Session(graph=self.detection_graph, config=config)
            # get the placeholder of input image, output scores, classes.
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.predict = True

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        predict = TrafficLight.UNKNOWN
        if self.predict is not None:
            # expand image dimensions
            image_expanded = np.expand_dims(image, axis=0)
            # run detection
            (scores, classes, num) = self.tf_session.run(
                [self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            # reduce the dimensions
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # calculate prediction
            c = 5
            predict = self.clabels[c]
            cc = classes[0]
            confidence = scores[0]
            print('id:', cc)
            print('confidence:', confidence)
            print('predict is:', predict)
            if cc > 0 and cc < 4 and confidence is not None and confidence > THRESHOLD:
                c = cc
                predict = self.clabels[c]
        print('Light:', predict)

        return predict
