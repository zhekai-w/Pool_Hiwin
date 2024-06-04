import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

#import tools for strategy 
import matplotlib.pyplot as plt
import random
import math
import py_pubsub.pool_strategy as ps




class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'topic',
            self.listener_callback,
            10)
        
        self.subscription  # prevent unused variable warning
        self.flat = []
        self.objectballx = []
        self.objectbally = []
        self.confidence = []

    def listener_callback(self, ballmsg):
        self.objectballx = []
        self.objectbally = []
        self.confidence = []

        print("hi")
        self.get_logger().info('I heard ball location:{}\n'.format(ballmsg.data))
        self.flat = ballmsg.data
        cueindex = 0
       
        #convert flat array to usable array 
        # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
        for i in range(0,len(ballmsg.data),4):
            if ballmsg.data[i] == 0:
                self.confidence.append(ballmsg.data[i+1])
                self.objectballx.append(ballmsg.data[i+2])
                self.objectbally.append(ballmsg.data[i+3])
            else:
                cueindex = i
        self.confidence.append(ballmsg.data[cueindex+1])
        self.objectballx.append(ballmsg.data[cueindex+2])
        self.objectbally.append(ballmsg.data[cueindex+3])
        print("objectball x:\n",self.objectballx)
        print("objectball y:\n",self.objectbally)
        n = len(self.objectballx)

        ValidRoute, bestrouteindex = ps.main(self.objectballx[-1],self.objectbally[-1], self.objectballx[0:n], self.objectbally[0:n],n-1)
        print('All valid route:\n',ValidRoute)
        print('Best route index:',bestrouteindex)
        print('Best Route:\n',ValidRoute[bestrouteindex])
        
        # remember to publish the clear array 

                
def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()
    # self.get_logger().info('testing')
    print("test")
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    