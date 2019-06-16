from styx_msgs.msg import TrafficLight
import cv2
import os
import tensorflow as tf
import numpy as np
from PIL import ImageDraw
LIGHT_DICT = {
    "0":{ # Seems to not be in use. remove?
        "color":(0,0,0),
        "name":'UNKNOWN',
        "tr":TrafficLight.UNKNOWN
    },
    "1":{
        "color":(0,255,0),
        "name":'GREEN',
        "tr":TrafficLight.GREEN
    },
    "2":{
        "color":(0,0,255),
        "name":'RED',
        "tr":TrafficLight.RED
    },
    "3":{
        "color":(0,255,255),
        "name":'YELLOW',
        "tr":TrafficLight.YELLOW
    },
    "4":{
        "color":(255,255,255),
        "name":'UNKNOWN',
        "tr":TrafficLight.UNKNOWN
    }
}

draw_image=True

class TLClassifier(object):
    def __init__(self):
                #TODO load classifier
        self.is_site = False ## False = simulator
        if self.is_site:
            path = "./models/ssd_udacity/frozen_inference_graph.pb"
        else:
            path = "./models/ssd_sim/frozen_inference_graph.pb"
        print("intit ml")
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
 
        self.counter = 0


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        This node takes in data from the /image_color, /current_pose, and 
        /base_waypoints topics and publishes the locations to stop for red traffic
         lights to the /traffic_waypoint topic.
        """    
        print("do it")
        #maybe we need
        t_ligth=TrafficLight.UNKNOWN
        image_np = np.expand_dims(np.asarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), dtype=np.uint8), 0)
        with tf.Session(graph=self.graph) as sess:         
            (boxes, scores, classes, num) = sess.run(
                    [self.detect_boxes, self.detect_scores, self.detect_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np})
            print('SCORES')
            print(scores[0])
            print('CLASSES')
            print(classes[0])
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            confidence_cutoff = 0.3
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)
            if len(classes)>0:
                best_index = np.argmax(scores)
                t_ligth = LIGHT_DICT[str(int(classes[best_index]))]["tr"]

            if draw_image:
                # The current box coordinates are normalized to a range between 0 and 1.
                # This converts the coordinates actual location on the image.
                height, width,_ = image.shape
                box_coords = to_image_coords(boxes, height, width)
                # Each class with be represented by a differently colored box
                image=draw_boxes(image, box_coords, classes, scores)
                self.counter += 1 
                pt=os.path.abspath("./i{}.png".format(self.counter))
                print(pt)
                cv2.imwrite(pt,image)
        #TODO implement light color prediction
    
        return t_ligth

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, scores,thickness=4):
    """Draw bounding boxes on the image"""
    #draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        print(bot,left,top,right)
        class_id = int(classes[i])
        
        text = LIGHT_DICT[str(class_id)]["name"] + " p:" +  str(scores[i])
        color = LIGHT_DICT[str(class_id)]["color"]
        image = cv2.rectangle(image,(left,bot ), (right, top) ,color,2)
        image = cv2.putText(image, text, (right, bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return image
