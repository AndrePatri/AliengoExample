from lrhc_control.utils.rhc_viz.rhc2viz_base import RhcToVizBridgeBase

import rclpy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
# from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, LivelinessPolicy
from rclpy.qos import QoSProfile

from rhcviz.utils.handshake import RHCVizHandshake

class RhcToViz2Bridge(RhcToVizBridgeBase):

    def _init_ros_pubs(self, id: str):
        if not rclpy.ok():
            rclpy.init()
        
        self._qos_settings = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # BEST_EFFORT
            durability=DurabilityPolicy.TRANSIENT_LOCAL, # VOLATILE
            history=HistoryPolicy.KEEP_LAST, # KEEP_ALL
            depth=10,  # Number of samples to keep if KEEP_LAST is used
            liveliness=LivelinessPolicy.AUTOMATIC,
            # deadline=1000000000,  # [ns]
            # partition='my_partition' # useful to isolate communications
            )

        self.node = rclpy.create_node(self.rhcviz_basename + "_" + self._remap_namespace+f"_{id}")

        self.rhc_q_pub = self.node.create_publisher(Float64MultiArray, 
                                            self.ros_names.rhc_q_topicname(basename=self.rhcviz_basename, 
                                                                        namespace=self._remap_namespace),
                                            qos_profile=self._qos_settings)

        self.rhc_refs_pub = self.node.create_publisher(Float64MultiArray, 
                                            self.ros_names.rhc_refs_topicname(basename=self.rhcviz_basename, 
                                                                        namespace=self._remap_namespace),
                                            qos_profile=self._qos_settings)

        if self._with_agent_refs:
            self.hl_refs_pub = self.node.create_publisher(Float64MultiArray, 
                                            self.ros_names.hl_refs_topicname(basename=self.rhcviz_basename, 
                                                                        namespace=self._remap_namespace),
                                            qos_profile=self._qos_settings)
            
        self.robot_q_pub = self.node.create_publisher(Float64MultiArray, 
                                            self.ros_names.robot_q_topicname(basename=self.rhcviz_basename, 
                                                                    namespace=self._remap_namespace),
                                            qos_profile=self._qos_settings)
        
        self.robot_jntnames_pub = self.node.create_publisher(String, 
                                            self.ros_names.robot_jntnames(basename=self.rhcviz_basename, 
                                                                namespace=self._remap_namespace),
                                            qos_profile=self._qos_settings)       

        self.rhc_jntnames_pub = self.node.create_publisher(String, 
                                            self.ros_names.rhc_jntnames(basename=self.rhcviz_basename, 
                                                                namespace=self._remap_namespace),
                                            qos_profile=self._qos_settings)   
        
        if self._pub_stime:
            self.simtime_pub = self.node.create_publisher(Clock, 
                                                "clock",
                                                qos_profile=self._qos_settings)
        
        self.handshaker = RHCVizHandshake(handshake_topic=self.ros_names.handshake_topicname(basename=self.rhcviz_basename, 
                                                                                namespace=self._remap_namespace),
                                node=self.node,
                                is_server=True)