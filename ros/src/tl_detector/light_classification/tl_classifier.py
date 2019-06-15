from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

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
        #TODO implement light color prediction
        
        #return 
        return TrafficLight.UNKNOWN
