import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import pickle
import time
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Twist, Point
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

class ReadSignNode(Node):
    def __init__(self):
        super().__init__('read_sign_node')
        self.bridge = CvBridge()
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "Raw Image")
        self.ret_Publish= self.create_publisher(Int8, '/ret', 10)
        self.Pleasework = Point()

        # Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value # Image Window Title
        
        # Only create image frames if we are not running headless (_display_image sets this)
        if self._display_image:
            # Set Up Image Viewing
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
            cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
        
        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.knn_model = None

        # Load the saved KNN model from pickle file
        with open('knn_model_new.pkl', 'rb') as f:
            self.knn_model = pickle.load(f)

        # Declare that the minimal_video_subscriber node is subscribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
                CompressedImage,
                '/image_raw/compressed',
                self._image_callback,
                image_qos_profile)
        self._video_subscriber # Prevents unused variable warning.

    def _image_callback(self, CompressedImage):    
        # The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        self.image_process(self._imgBGR)
        if self._display_image:
            self.show_image(self._imgBGR)
    def show_image(self, img):
        cv2.imshow(self._titleOriginal, img)
		# Cause a slight delay so image is displayed
        self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.

    def image_process (self, _imgBGR):
        img = cv2.cvtColor(self._imgBGR, cv2.COLOR_BGR2HSV)
        upper_red = np.array([25, 255, 255])
        lower_red = np.array([0, 100, 50])
        upper_green = np.array([90, 255, 255])
        lower_green = np.array([45, 100, 40])
        upper_blue = np.array([140, 255, 255])
        lower_blue = np.array([90, 40, 30])

        thresh_red = cv2.inRange(img, lower_red, upper_red)
        thresh_green = cv2.inRange(img, lower_green, upper_green)
        thresh_blue = cv2.inRange(img, lower_blue, upper_blue)

        mask = thresh_red | thresh_green | thresh_blue

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros((40, 40), dtype=np.uint8)

        # Calculate the center of the image
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        # Find the contour closest to the center
        closest_contour = None
        min_distance = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small contours
                # Calculate the centroid of the contour
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Calculate the distance from the center of the image
                distance = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
                
                # Check if this contour is closer to the center and has larger area
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour
            else:
                return np.zeros((40, 40), dtype=np.uint8)

        # Get the bounding box of the closest contour
        # if closest_contour is not None:

        x, y, w, h = cv2.boundingRect(closest_contour)
        cropped_img = opening[y:y+h, x:x+w]
            # Resize the cropped image to a fixed shape
        # cropped_img = np.array(cv2.resize(cropped_img, (40, 40)))

            # # Display images for debugging
            # cv2.imshow("cropped", cropped_img)
            # cv2.imshow("hsv", img)
            # cv2.imshow("closing", closing)
            # cv2.waitKey(0)


        # Resize the cropped image to a fixed shape
        cropped_img = np.array(cv2.resize(cropped_img, (40,40)))
        cropped_img = cropped_img.flatten().reshape(1,40*40)
        cropped_img = cropped_img.astype(np.float32)
        ret = self.knn_model.predict(cropped_img)
        self.get_logger().info('Return: %s' % ret)
        msg = Int8()
        msg.data = int(ret[0])
        self.ret_Publish.publish(msg)
        # time.sleep(0.2)

    



def main(args=None):
    rclpy.init(args=args)
    read_sign_node = ReadSignNode()
    rclpy.spin(read_sign_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()