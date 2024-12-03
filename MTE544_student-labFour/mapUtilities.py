import matplotlib.pyplot as plt
import numpy as np 

from sensor_msgs.msg import LaserScan
from math import floor
import math

from math import pi as M_PI
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PointStamped, Quaternion, Point
from utilities import *

class mapManipulator(Node):


    def __init__(self, filename_: str = "room.yaml", laser_sig=0.01):
        
        
        super().__init__('likelihood_field')
        
        filenameYaml=None
        filenamePGM=None
        if ".pgm" in filename_:
            
            filenamePGM=filename_
            filenameYaml=filename_.replace(".pgm", ".yaml")
            
        elif ".yaml" in filename_:
            
            filenameYaml=filename_
            filenamePGM=filename_.replace(".yaml", ".pgm")
        
        else:
            filenameYaml=filename_ + ".yaml"
            filenamePGM=filename_+".pgm"

        

        width, height, max_value, pixels = self.read_pgm(filenamePGM)


        self.width = width
        self.height = height
        
        
        

        self.image_array = np.array(pixels).reshape((height, width))
        self.o_x, self.o_y, self.res, self.thresh = self.read_description(filenameYaml)

        self.laser_sig=laser_sig
        
        self.likelihood_msg=None

        
    def getAllObstacles(self):
        image_array=self.image_array.T

        
        
        
        indices = np.where(image_array < 10)
        
        return [self.cell_2_position([i, j]) for i, j in zip(indices[0], indices[1])]

    def getLikelihoodField(self):
        return self.likelihood_field
    
    def getMetaData(self):
        return self.o_x, self.o_y, self.res, self.thresh
            
    def getMap(self):
        return self.image_array
    
    def timer_callback(self):
        if self.likelihood_msg is None: 
            return
        self.map_publisher.publish(self.likelihood_msg)
        
    def read_pgm(self, filename):
        with open(filename, 'rb') as f:
            # Check if it's a PGM file
            header = f.readline().decode().strip()
            
            if header != 'P5':
                raise ValueError('Invalid PGM file format')

            # Skip comments
            line = f.readline().decode().strip()
            while line.startswith('#'):
                line = f.readline().decode().strip()

            # Read width, height, and maximum gray value
            width, height = map(int, line.split())
            max_value = int(f.readline().decode().strip())

            # Read the image data
            image_data = f.read()

        # Convert image data to a list of pixel values
        pixels = [x for x in image_data]

        return width, height, max_value, pixels

    def plot_pgm_image(self, image_array, astar_path, odom_path, title):
        
        # Convert paths to pixel coordinates
        astar_cell = [self.position_2_cell(p) for p in astar_path]
        odom_cell = [self.position_2_cell(p) for p in odom_path]

        X_MIN = 75
        Y_MIN = 75

        # Plot grids
        for i in range(0, image_array.shape[0]-96, 3):
            plt.plot([i-0.5 for j in range(100, 225)], [p-100 for p in range(100, 225)], color='lightgray', alpha=0.5, label="_nolegend_")
        for i in range(0, image_array.shape[1]-200, 3):
            plt.plot([p-100 for p in range(100, 300)], [i-0.5 for j in range(100, 300)], color='lightgray', alpha=0.5, label="_nolegend_")

        # Plot the image
        plt.imshow(image_array[X_MIN:200, Y_MIN:275], cmap='gray')
        plt.axis('off')
        plt.title(title)

        # Plot the two paths
        plt.plot([p[0]-Y_MIN for p in astar_cell], [p[1]-X_MIN for p in astar_cell])
        plt.plot([p[0]-Y_MIN for p in odom_cell[1:-1]], [p[1]-X_MIN for p in odom_cell[1:-1]])

        # Add start and end points
        plt.scatter(odom_cell[0][0]-Y_MIN, odom_cell[0][1]-X_MIN, color='orange')
        plt.scatter(odom_cell[-1][0]-Y_MIN, odom_cell[-1][1]-X_MIN, color='orange', marker='x')

        plt.legend(["Target Path (A Star)", "Actual Path (Odom)", "Robot Start", "Robot End"])
        plt.grid()
        plt.show()



    def read_description(self, filenameYAML):
        import re

        # Open and read the YAML file
        with open(filenameYAML, 'r') as file:
            yaml_content = file.readlines()

            # Extract the desired fields
            threshold = None
            origin_x = None
            origin_y = None
            resolution = None

            for line in yaml_content:
                if 'occupied_thresh' in line:
                    threshold = float(re.findall(r'\d+\.\d+', line)[0])
                elif 'origin' in line:
                    origin_values = re.findall(r'-?\d+\.\d+', line)
                    origin_x = float(origin_values[0])
                    origin_y = float(origin_values[1])
                elif 'resolution' in line:
                    resolution = float(re.findall(r'\d+\.\d+', line)[0])
        return origin_x, origin_y, resolution, threshold


    def getOrigin(self):
        return np.array([self.o_x, self.o_y])
    
    def getResolution(self):
        return self.res
    
    def cell_2_position(self, pix):
        i,j= pix
        return self.o_x + i*self.getResolution(),    (self.height - j) * self.getResolution()  + self.o_y  
    
    
    def position_2_cell(self, pos):
        x,y = pos
        return floor( (-self.o_x + x)/self.getResolution()), -floor( -self.height + (-self.o_y + y)/self.getResolution() )


    def make_likelihood_field(self):
        
        image_array=self.image_array

        from sklearn.neighbors import KDTree
        
        

        indices = np.where(image_array < 10)
        
        occupied_points = [self.cell_2_position([i, j]) for i, j in zip(indices[0], indices[1])]
        all_positions = [self.cell_2_position([i, j]) for i in range(image_array.shape[0]) for j in range(image_array.shape[1])]

        kdt=KDTree(occupied_points)

        dists=kdt.query(all_positions, k=1)[0][:]
        probabilities=np.exp( -(dists**2) / (2*self.laser_sig**2))
        
        likelihood_field=probabilities.reshape(image_array.shape)
        
        likelihood_field_img=np.array(255-255*probabilities.reshape(image_array.shape), dtype=np.int32)
        
        self.likelihood_img=likelihood_field_img
        
        self.occ_points=np.array(occupied_points)
        
                
        self.plot_pgm_image(likelihood_field_img)

        self.likelihood_field = likelihood_field
        
        return likelihood_field
                
    
    def _numpy_to_data(self, data):
        """
        Convert the numpy array containing grid data to a python
        list suitable for use as the data field in an OccupancyGrid
        message.
        """
        flat_grid = data.reshape((data.size,)) * 100
        data_ = set(np.array(np.round(flat_grid), dtype='int'))

        return data_
    
    def to_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"

        likelihoodField = self.getLikelihoodField()

        grid.info.resolution = self.getResolution()  # Set the resolution (m/cell)
        grid.info.width = self.height
        grid.info.height = self.width
        grid.info.origin = Pose()  # Set the origin of the map (geometry_msgs/Pose)
        
        gridOrigin = self.cell_2_position([0, self.height])
        

        grid.info.origin.orientation.w = np.cos(-np.pi/4)
        grid.info.origin.orientation.z = np.sin(-np.pi/4)
        offset = -self.height*self.getResolution()



        grid.info.origin.position.x, grid.info.origin.position.y = self.getOrigin()[0], +self.getOrigin()[1] - offset


        #grid.info.origin.orientation.w = np.cos(np.pi/2)
        #grid.info.origin.orientation.z = np.sin(np.pi/2)
        # Flatten the likelihood field and scale it to [0, 100], set unknown as -1
        
        normalized_likelihood = np.clip(likelihoodField.T * 100, 0, 100)
        
        # Convert to integers and ensure values are within the range [-128, 127]
        # In ROS, 0 means unknown, so we remap our values from [0, 100] to [1, 101]
        # and then subtract 1 to have [0, 100] for the message
        
        grid.data = [int(value) for value in normalized_likelihood.flatten()]
        grid.data = list(grid.data)


        return grid


    def calculate_score(self,x,y):
        try:
            return self.likelihood_field[self.position_2_cell(x,y)]
        except IndexError:
            return 0
        
        
        
    def map_localation_query(self, laser_msg: LaserScan):
        
        # this part was for kidnapping section of the particle filter algorithm
        # the complete algorithm is in this repo
        # https://github.com/aalghooneh/ros2-particlefilter-python

        pass
            





     



import argparse
if __name__=="__main__":
    


    rclpy.init()

    parser=argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="room.yaml", help='the absolute path to argument')
    parser.add_argument('--std', type=float, help='the std', default=0.7)


    args = parser.parse_args()

    MAP_UTILITIS=mapManipulator(args.map, args.std)

    test = "location1-True"
    astar_f = "CSVs/" + test + "-astar-path.csv"
    odom_f = "CSVs/" + test + "-robotPose-odom.csv"

    astar = []
    odom = []

    # Read in astar and odom path data
    with open(astar_f) as f:
        f.readline()
        for l in f.readlines():
            l2 = l.split(",")
            astar.append((float(l2[0]), float(l2[1])))
    with open(odom_f) as f:
        f.readline()
        for l in f.readlines():
            l2 = l.split(',')
            odom.append((float(l2[0]), float(l2[1])))

    # Plot complete image
    MAP_UTILITIS.plot_pgm_image(MAP_UTILITIS.image_array, astar, odom, 'Location 1 - Euclidian Distance')

    #rclpy.spin(MAP_UTILITIS)


# Usage example

