from rhcviz.utils.handshake import RHCVizHandshake
from rhcviz.utils.namings import NamingConventions
from rhcviz.utils.string_list_encoding import StringArray

from lrhc_control.controllers.rhc.horizon_based.utils.math_utils import hor2w_frame,base2world_frame

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal
from control_cluster_bridge.utilities.shared_data.sim_data import SharedEnvInfo

from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from control_cluster_bridge.utilities.homing import RobotHomer
import numpy as np

from EigenIPC.PyEigenIPC import dtype

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

import rclpy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock

# from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, LivelinessPolicy
from rclpy.qos import QoSProfile

from typing import List

from perf_sleep.pyperfsleep import PerfSleep
import time 

class RhcToViz2Bridge:

    # bridge from shared mem to ROS 2
    
    def __init__(self, 
            namespace: str, 
            remap_ns: str = None,
            verbose = False,
            vlevel: VLevel = VLevel.V1,
            rhcviz_basename = "RHCViz",
            robot_selector: List = [0, None],
            with_agent_refs = False,
            rhc_refs_in_h_frame: bool = False,
            agent_refs_in_h_frame: bool = False,
            env_idx: int = 0,
            sim_time_trgt: float = None,
            srdf_homing_file_path: str = None,
            abort_wallmin: float = 5.0,
            use_static_idx: bool = True,
            update_dt: float = 0.1,
            pub_stime: float = True):
        
        self._srdf_homing_file_path=srdf_homing_file_path # used to retrieve homing
        self._homer=None
        self._some_jnts_are_missing=False

        self._pub_stime=pub_stime
        self._sim_time_trgt = sim_time_trgt
        if self._sim_time_trgt is None:
            self._sim_time_trgt = np.inf # basically run indefinitely

        self._current_index = env_idx
        # self._use_static_idx = True if env_idx >= 0 else False
        self._use_static_idx =use_static_idx

        self._rhc_refs_in_hor_frame = rhc_refs_in_h_frame
        self._agent_refs_in_h_frame = agent_refs_in_h_frame
        
        self._robot_selector = robot_selector

        self._with_agent_refs = with_agent_refs

        self.verbose = verbose
        self.vlevel = vlevel

        self.namespace = namespace # defines uniquely the kind of controller 
        # (associated with a specific robot)
        self._remap_namespace=remap_ns
        if self._remap_namespace is None: # allow publishing with different namespace
            self._remap_namespace=self.namespace
        # ros stuff
        self.ros_names = NamingConventions() # rhcviz naming conventions
        self.rhcviz_basename = rhcviz_basename 
        
        self.cluster_size = None
        self.jnt_names_robot = None
        self.jnt_names_rhc = None

        self.rhc_internal_clients = None
        self.robot_state = None
        self.rhc_refs = None
        self.agent_refs = None
        self._sim_data = None

        self._update_counter = 0
        self._print_frequency = 100

        self._missing_homing=None
        
        self._is_running = False
        self._closed=False

        self._safety_abort_walldt=abort_wallmin # [min]. If using stime, abort
        # if it hasn't changed in the last _safety_abort_walldt minutes
        self._abort_stime_res=1e-4 # [s] if stime was constant withing this res for 
        # more than _safety_abort_walldt, abort
        self._stime_before=-1.0

        self._initialize()
        
        self._update_dt=update_dt
        self.init(update_dt=self._update_dt)

    def __del__(self):
        self.close()

    def _check_selector(self):
        
        if not isinstance(self._robot_selector, List):

            exception = f"robot_selector should be a List!"

            Journal.log(self.__class__.__name__,
                "_parse_selector",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if not len(self._robot_selector) == 2:

            exception = f"robot_selector should be a List of two values: [start_index, end_idx]!"

            Journal.log(self.__class__.__name__,
                "_parse_selector",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if not isinstance(self._robot_selector[0], int) or \
            not self._robot_selector[0] >= 0 or \
            not self._robot_selector[0] < self.cluster_size:

            exception = f"first index should be a positive integer, not bigger than{self.cluster_size - 1}!"

            Journal.log(self.__class__.__name__,
                "_parse_selector",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if not isinstance(self._robot_selector[1], (int, type(None))):

            if self._robot_selector[1] is not None:

                if self._robot_selector[1] <= self._robot_selector[0]:

                    exception = f"second index should be bigger than first index{self._robot_selector[0]}, but got {self._robot_selector[1]}!"

                    Journal.log(self.__class__.__name__,
                        "_parse_selector",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            
                if not self._robot_selector[0] < self.cluster_size:

                    exception = f"second index should be a positive integer, not bigger than{self.cluster_size - 1}!"

                    Journal.log(self.__class__.__name__,
                        "_parse_selector",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
        
        if self._robot_selector[1] is None:

            self._robot_selector[1] = self.cluster_size -1

        self._robot_indexes = list(range(self._robot_selector[0], 
                                    self._robot_selector[1]+1))

    def _initialize(self):
        
        if not rclpy.ok():
            rclpy.init()

        from datetime import datetime
        time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')

        self.node = rclpy.create_node(self.rhcviz_basename + "_" + self._remap_namespace+f"_{time_id}")
        self._qos_settings = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # BEST_EFFORT
            durability=DurabilityPolicy.TRANSIENT_LOCAL, # VOLATILE
            history=HistoryPolicy.KEEP_LAST, # KEEP_ALL
            depth=10,  # Number of samples to keep if KEEP_LAST is used
            liveliness=LivelinessPolicy.AUTOMATIC,
            # deadline=1000000000,  # [ns]
            # partition='my_partition' # useful to isolate communications
            )

        self.handshaker = RHCVizHandshake(handshake_topic=self.ros_names.handshake_topicname(basename=self.rhcviz_basename, 
                                                                                        namespace=self._remap_namespace),
                                        node=self.node,
                                        is_server=True)
        
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

        self.simtime_pub=None
        if self._pub_stime:
            self.simtime_pub = self.node.create_publisher(Clock, 
                                                "clock",
                                                qos_profile=self._qos_settings)

        # sim data
        self._sim_data = SharedEnvInfo(namespace=self.namespace,
                                is_server=False,
                                safe=False,
                                verbose=self.verbose,
                                vlevel=self.vlevel)
        self._sim_data.run()
        self._sim_datanames = self._sim_data.param_keys
        self._simtime_idx = self._sim_datanames.index("cluster_time")
        self._ros_clock = Clock()

        # robot state
        self.robot_state = RobotState(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=self.verbose,
                                vlevel=self.vlevel)
        self.robot_state.set_q_remapping(q_remapping=[1, 2, 3, 0]) # remapping from w, i, j, k
        self.robot_state.run()
        # to rviz conventions (i, k, k, w)
        self.rhc_refs = RhcRefs(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=self.verbose,
                                vlevel=self.vlevel)
        self.rhc_refs.rob_refs.set_q_remapping(q_remapping=[1, 2, 3, 0]) # remapping from w, i, j, k
        # to rviz conventions (i, k, k, w)
        self.rhc_refs.run()

        self.rhc_cmds = RhcCmds(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=self.verbose,
                                vlevel=self.vlevel) 
        self.rhc_cmds.set_q_remapping(q_remapping=[1, 2, 3, 0]) # remapping from w, i, j, k
        # to rviz and horizon's conventions (i, k, k, w)
        self.rhc_cmds.run()

        if self._with_agent_refs:
            self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=True,
                                verbose=self.verbose,
                                vlevel=self.vlevel)
            self.agent_refs.rob_refs.set_q_remapping(q_remapping=[1, 2, 3, 0]) # remapping from w, i, j, k
            # to rviz conventions (i, k, k, w)
            self.agent_refs.run()

        self.cluster_size = self.robot_state.n_robots()
        self.jnt_names_robot = self.robot_state.jnt_names()
            
        self._check_selector()

        # env selector
        if not self._use_static_idx:
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = self.verbose, 
                    vlevel = self.vlevel,
                    safe = False,
                    dtype=dtype.Int)
            self.env_index.run()

        rhc_internal_config = RhcInternal.Config(is_server=False, 
                        enable_q=True)
        # rhc internal data
        self.rhc_internal_clients = []
        for i in range(len(self._robot_indexes)):
            self.rhc_internal_clients.append(RhcInternal(config=rhc_internal_config,
                                                namespace=self.namespace,
                                                rhc_index=self._robot_indexes[i],
                                                verbose=self.verbose,
                                                vlevel=self.vlevel,
                                                safe=False))
            self.rhc_internal_clients[self._robot_indexes[i]].run()

        # publishing joint names on topic 
        string_array = StringArray()
        self.jnt_names_robot_encoded = string_array.encode(self.jnt_names_robot) # encoding 
        # jnt names in a ; separated string

        self.handshaker.set_n_nodes(self.rhc_internal_clients[0].q.n_cols) # signal to RHViz client
        # the number of nodes of the RHC problem

        if self._srdf_homing_file_path is not None:
            self._homer= RobotHomer(srdf_path=self._srdf_homing_file_path, 
                            jnt_names=self.jnt_names_robot)

        self.jnt_names_rhc = self.rhc_internal_clients[0].jnt_names() # assumes all controllers work on the same robot
        rhc_jnt_names_set=set(self.jnt_names_rhc)
        env_jnt_names_set=set(self.jnt_names_robot)
        missing_jnts=list(env_jnt_names_set-rhc_jnt_names_set)
        if not len(missing_jnts)==0:
            self.jnt_names_rhc=self.jnt_names_rhc+missing_jnts
            self._some_jnts_are_missing=True
            self._missing_homing=np.zeros((len(missing_jnts),self.rhc_internal_clients[0].q.n_cols))
            if self._homer is not None:
               for node in range(self.rhc_internal_clients[0].q.n_cols):
                   self._missing_homing[:, node]=np.array(self._homer.get_homing_vals(jnt_names=missing_jnts))
                
        self.jnt_names_rhc_encoded = string_array.encode(self.jnt_names_rhc) # encoding 
        # jnt names ifor rhc controllers   
    
    def init(self, update_dt: float = 0.01):

        self._is_running = True

        self._update_dt = update_dt
        self._start_time = 0.0
        self._elapsed_time = 0.0

        self._time_to_sleep_ns = 0

        info = f": initializing bridge with (realtime) update dt {self._update_dt} s and sim time target {self._sim_time_trgt} s"
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)

        self._reset()

    def _reset(self):
        self._sim_time = 0 
        self._sim_time_init = self._sim_data.get()[self._simtime_idx].item()
        self._stime_before=self._sim_time # record stime now

        self._safety_check_start_time = time.monotonic() 
        self._sporadic_log_freq=500
        self._log_counter=0

    def run(self, sim_time: float = None):

        if sim_time is not None:
            self._sim_time_trgt=sim_time
        else:
            self._sim_time_trgt=np.inf

        self._reset() # reset counters
        # run bridge for some sim time
        finished=False
        while not finished and (self._is_running and not self._closed):
            try:
                finished=not self.step()
            except KeyboardInterrupt:
                break
        
        return finished
        

    def step(self):

        t_before_update = time.monotonic() 
                
        self._update() # update data on ROS

        if self._log_counter%self._sporadic_log_freq==0:
            Journal.log(self.__class__.__name__,
                "run",
                f"elapsed sim time {round(self._sim_time,2)}/{self._sim_time_trgt} s.",
                LogType.INFO)

        self._log_counter+=1
        # check if we need to stop
        if (t_before_update-self._safety_check_start_time)*1.0/60.0>=self._safety_abort_walldt:
            # every self._safety_abort_walldt [min]
            self._safety_check_start_time=time.monotonic() 
            if (self._sim_time-self._stime_before)<=self._abort_stime_res:
                warn=f"terminating rhc2viz bridge due to timeout!\t" + \
                    f"No sim time update detected over {self._safety_abort_walldt} min. \n" +  \
                    f"stime before: {self._stime_before} s, stime now: {self._sim_time}"
                Journal.log(self.__class__.__name__,
                    "run",
                    warn,
                    LogType.WARN)
                return False

            self._stime_before=self._sim_time # record stime now

        if self._sim_time >= self._sim_time_trgt:
            Journal.log(self.__class__.__name__,
                "run",
                f"terminating rhc2viz bridge ({self._sim_time}>={self._sim_time_trgt})",
                LogType.INFO)
            return False

        self._elapsed_time = time.monotonic() - t_before_update
        self._time_to_sleep_ns = int((self._update_dt - self._elapsed_time) * 1e+9) # [ns]
        if self._time_to_sleep_ns < 0:
            warning = f"Could not match desired (realtime) update dt of {self._update_dt} s. " + \
                f"Elapsed time to update {self._elapsed_time}."
            Journal.log(self.__class__.__name__,
                "run",
                warning,
                LogType.WARN,
                throw_when_excep = True)
        else:
            PerfSleep.thread_sleep(self._time_to_sleep_ns) 
            
        return True

    def _update(self):
        
        success = False

        if not self._use_static_idx:
            self.env_index.synch_all(read=True, retry=True)
            self._current_index = self.env_index.get_numpy_mirror()[0, 0].item()

        if self._current_index in self._robot_indexes:

            if self._is_running:
            
                # read from shared memory
                self.robot_state.synch_from_shared_mem()
                self.rhc_refs.rob_refs.synch_from_shared_mem()
                self.rhc_cmds.synch_from_shared_mem()
                if self._with_agent_refs:
                    self.agent_refs.rob_refs.synch_from_shared_mem()
                self.rhc_internal_clients[self._current_index].synch(read=True)
                self._sim_time = self._sim_data.get()[self._simtime_idx].item() - self._sim_time_init
                self._publish()
            
        else:

            warning = f"Current env index {self._current_index} is not within {self._robot_indexes}!\n" + \
                "No update will be performed"

            Journal.log(self.__class__.__name__,
                "update",
                warning,
                LogType.WARN,
                throw_when_excep = True)

        return success

    def close(self):

        if not self._closed:
            if not self.rhc_internal_clients is None:
                for i in range(len(self.rhc_internal_clients)):
                    self.rhc_internal_clients[i].close() # closes servers

            if not self.robot_state is None:
                self.robot_state.close()
            if not self.rhc_refs is None:
                self.rhc_refs.close()
            if not self.rhc_cmds is None:
                self.rhc_cmds.close()

            self.node.destroy_node()
            # rclpy.shutdown()
            self._closed=True
            self._is_running=False
    
    def _sporadic_log(self,
                calling_methd: str,
                msg: str,
                logtype: LogType = LogType.INFO):

        if self.verbose and \
            (self._update_counter+1) % self._print_frequency == 0: 
            
            Journal.log(self.__class__.__name__,
                calling_methd,
                msg,
                logtype,
                throw_when_excep = True)
    
    def _contains_nan(self,
                    data: np.ndarray):
        
        return np.isnan(data).any()
    
    def _publish(self):
        
        if self.simtime_pub is not None:
            self._ros_clock.clock.sec = int(self._sim_time)
            self._ros_clock.clock.nanosec = int((self._sim_time - self._ros_clock.clock.sec)*1e9)

            self.simtime_pub.publish(self._ros_clock)
        # continously publish also joint names 
        self.robot_jntnames_pub.publish(String(data=self.jnt_names_robot_encoded))
        
        self.rhc_jntnames_pub.publish(String(data=self.jnt_names_rhc_encoded))

        # publish rhc_q
        rhc_actual=self.rhc_internal_clients[self._current_index].q.get_numpy_mirror()[:, :]
        rhc_missing=self._missing_homing
        
        if rhc_missing is None:
            rhc_q=rhc_actual.flatten()
        else:
            rhc_q_tot = np.concatenate((rhc_actual, rhc_missing), axis=0)
            rhc_q =rhc_q_tot.flatten()

        root_q_robot = self.robot_state.root_state.get(data_type="q_full",robot_idxs=self._current_index)
        jnts_q_robot = self.robot_state.jnts_state.get(data_type="q")[self._current_index, :]

        robot_q = np.concatenate((root_q_robot, jnts_q_robot), axis=0)

        # rhc refs
        rhc_ref_pose = self.rhc_refs.rob_refs.root_state.get(data_type="q_full",robot_idxs=self._current_index)
        rhc_ref_twist= self.rhc_refs.rob_refs.root_state.get(data_type="twist",robot_idxs=self._current_index)
        
        if self._rhc_refs_in_hor_frame:
            rhc_ref_twist_h = rhc_ref_twist.copy().reshape(-1, 1)
            # using orientaton q (remapped to horizon's and rviz convetions) internal to the controller
            # (this HAS to match inside the controller!)
            rhc_q_cmd=self.rhc_cmds.root_state.get(data_type="q",robot_idxs=self._current_index).reshape(-1, 1)
            hor2w_frame(t_h=rhc_ref_twist.reshape(-1, 1), 
                        q_b=rhc_q_cmd, 
                        t_out=rhc_ref_twist_h)
            rhc_refs = np.concatenate((rhc_ref_pose, rhc_ref_twist_h.flatten()), axis=0)
        else:
            rhc_refs = np.concatenate((rhc_ref_pose, rhc_ref_twist), axis=0)

        # high lev refs
        if self._with_agent_refs:
            if self._agent_refs_in_h_frame:
                hl_ref_pose = self.agent_refs.rob_refs.root_state.get(data_type="q_full",robot_idxs=self._current_index).numpy()
            
                hl_ref_twist= self.agent_refs.rob_refs.root_state.get(data_type="twist",robot_idxs=self._current_index).numpy()
                agent_ref_twist_h = hl_ref_twist.copy().reshape(-1, 1)
                hor2w_frame(t_h=hl_ref_twist.reshape(-1, 1), 
                        q_b=self.robot_state.root_state.get(data_type="q",
                        robot_idxs=self._current_index).reshape(-1, 1), 
                        t_out=agent_ref_twist_h)
                hl_refs = np.concatenate((hl_ref_pose, agent_ref_twist_h.flatten()), axis=0)
            else: # in base frame
                
                hl_ref_pose = self.agent_refs.rob_refs.root_state.get(data_type="q_full",robot_idxs=self._current_index).numpy()
                hl_ref_twist_base_loc= self.agent_refs.rob_refs.root_state.get(data_type="twist",robot_idxs=self._current_index).numpy()

                agent_ref_twist_w = hl_ref_twist_base_loc.copy().reshape(-1, 1)
                
                base2world_frame(t_b=hl_ref_twist_base_loc.reshape(-1, 1), 
                    q_b=self.robot_state.root_state.get(data_type="q",
                        robot_idxs=self._current_index).reshape(-1, 1), 
                    t_out=agent_ref_twist_w)
                hl_refs = np.concatenate((hl_ref_pose, agent_ref_twist_w.flatten()), axis=0)

        # publish rhc q
        if not self._contains_nan(rhc_q):
            self.rhc_q_pub.publish(Float64MultiArray(data=rhc_q))
        else:
            self._sporadic_log(calling_methd="_publish", 
                        msg="rhc q data contains some NaN. That data will not be published")
            
        # publish robot_q
        if not self._contains_nan(robot_q):
            self.robot_q_pub.publish(Float64MultiArray(data=robot_q))
        else:
            self._sporadic_log(calling_methd="_publish", 
                        msg="robot q data contains some NaN. That data will not be published")
        
        # publish rhc_refs
        if not self._contains_nan(rhc_refs):
            self.rhc_refs_pub.publish(Float64MultiArray(data=rhc_refs))
        else:
            self._sporadic_log(calling_methd="_publish", 
                        msg="rhc refs data contains some NaN. That data will not be published")
        
        # publish hl_refs
        if self._with_agent_refs:
            if not self._contains_nan(hl_refs):
                self.hl_refs_pub.publish(Float64MultiArray(data=hl_refs))
            else:
                self._sporadic_log(calling_methd="_publish", 
                            msg="high-level refs data contains some NaN. That data will not be published")
