from lrhc_control.utils.rhc_viz.rhc2viz_base import RhcToVizBridgeBase

import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock

from rhcviz.utils.handshake import RHCVizHandshake

class RhcToVizBridge(RhcToVizBridgeBase):

    def _init_ros_pubs(self, id: str):
        
        self.node=rospy.init_node(self.rhcviz_basename + "_" + self._remap_namespace+f"_{id}")

        self.rhc_q_pub = rospy.Publisher(self.ros_names.rhc_q_topicname(basename=self.rhcviz_basename, 
                                            namespace=self._remap_namespace),
                            Float64MultiArray, 
                            queue_size=10)
        
        self.rhc_refs_pub = rospy.Publisher(self.ros_names.rhc_refs_topicname(basename=self.rhcviz_basename, 
                                            namespace=self._remap_namespace),
                            Float64MultiArray, 
                            queue_size=10)
        if self._with_agent_refs:
            self.hl_refs_pub = rospy.Publisher(self.ros_names.hl_refs_topicname(basename=self.rhcviz_basename, 
                                            namespace=self._remap_namespace),
                            Float64MultiArray, 
                            queue_size=10)
            
        self.robot_q_pub = rospy.Publisher(self.ros_names.robot_q_topicname(basename=self.rhcviz_basename, 
                                            namespace=self._remap_namespace), 
                            Float64MultiArray, 
                            queue_size=10)
        
        self.robot_jntnames_pub = rospy.Publisher(self.ros_names.robot_jntnames(basename=self.rhcviz_basename, 
                                            namespace=self._remap_namespace),
                            String, 
                            queue_size=10)       

        self.rhc_jntnames_pub = rospy.Publisher(self.ros_names.rhc_jntnames(basename=self.rhcviz_basename, 
                                            namespace=self._remap_namespace),
                            String, 
                            queue_size=10)  
        
        if self._pub_stime:
            self.simtime_pub = rospy.Publisher("/clock", 
                                Clock, 
                                queue_size=10)
        
        self.handshaker = RHCVizHandshake(handshake_topic=self.ros_names.handshake_topicname(basename=self.rhcviz_basename, 
                    namespace=self._remap_namespace),
                is_server=True)
    
    def close(self):
        super().close()
        # rospy.signal_shutdown("closing rhc2viz bridge")
    
    def pub_stime(self, stime: float):
        # Create a Clock message
        self._ros_clock.clock = rospy.Time.from_sec(stime)  # Convert float time to rospy.Time

        # Publish the Clock message
        self.simtime_pub.publish(self._ros_clock)