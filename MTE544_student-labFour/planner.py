
from mapUtilities import *
from a_star import *
from utilities import Logger, STARTTIME, EUCLIDIAN

POINT_PLANNER=0; TRAJECTORY_PLANNER=1

class planner:
    def __init__(self, type_, mapName="room"):

        self.type=type_
        self.mapName=mapName
        self.astar_logger=Logger(f"CSVs/{STARTTIME}-{EUCLIDIAN}-astar-path.csv" , ["x", "y"])

    
    def plan(self, startPose, endPose):
        
        if self.type==POINT_PLANNER:
            return self.point_planner(endPose)
        
        elif self.type==TRAJECTORY_PLANNER:
            self.costMap=None
            self.initTrajectoryPlanner()
            return self.trajectory_planner(startPose, endPose)


    def point_planner(self, endPose):
        return endPose

    def initTrajectoryPlanner(self):


        # Create the cost-map, the laser_sig is
        # the standard deviation for the gausian for which
        # the mean is located on the occupant grid.
         
        # Laser sig of 0.7 helped avoid walls, making the path stay very far from obstacles
        self.m_utilites=mapManipulator(laser_sig=0.7)
            
        self.costMap=self.m_utilites.make_likelihood_field()
        

    def trajectory_planner(self, startPoseCart, endPoseCart):


        # This is to convert the cartesian coordinates into the 
        # the pixel coordinates of the map image, remmember,
        # the cost-map is in pixels. You can by the way, convert the pixels
        # to the cartesian coordinates and work by that index, the a_star finds
        # the path regardless. 

        # Convert the cell pixels into the cartesian coordinates using provided function
        startPose=self.m_utilites.position_2_cell(startPoseCart)
        endPose=self.m_utilites.position_2_cell(endPoseCart)

        # Run a-star using the search function
        astar_path = search(self.costMap, startPose, endPose)
        
        # Store and log the generated path for the robot to follow
        Path = list(map(self.m_utilites.cell_2_position, astar_path))
        for p in Path:
            self.astar_logger.log_values(p)

        # Return the path to be used in decisions.py
        return Path




if __name__=="__main__":

    m_utilites=mapManipulator()
    
    map_likelihood=m_utilites.make_likelihood_field()

    # you can use this part of the code to test your 
    # search algorithm regardless of the ros2 hassles
    
