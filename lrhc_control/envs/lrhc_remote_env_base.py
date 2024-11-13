from lrhc_control.controllers.rhc.lrhc_cluster_server import LRhcClusterServer
from lrhc_control.utils.shared_data.remote_stepping import RemoteStepperClnt
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetClnt
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetRequest
from lrhc_control.utils.jnt_imp_control_base import JntImpCntrlBase
from lrhc_control.utils.hybrid_quad_xrdf_gen import get_xrdf_cmds
from lrhc_control.utils.xrdf_gen import generate_srdf, generate_urdf
from lrhc_control.utils.math_utils import quaternion_difference
from lrhc_control.utils.custom_arg_parsing import extract_custom_xacro_args, merge_xacro_cmds

from control_cluster_bridge.utilities.homing import RobotHomer
from control_cluster_bridge.utilities.shared_data.jnt_imp_control import JntImpCntrlData

from SharsorIPCpp.PySharsorIPC import VLevel, Journal, LogType

from typing import List, Union, Dict, TypeVar

import os
import signal
import time

import numpy as np
import torch

from abc import ABC, abstractmethod

JntImpCntrlChild = TypeVar('JntImpCntrlChild', bound='JntImpCntrlBase')

class LRhcEnvBase(ABC):

    def __init__(self,
                robot_names: List[str],
                robot_urdf_paths: List[str],
                robot_srdf_paths: List[str],
                jnt_imp_config_paths: List[str],
                n_contacts: List[int],
                cluster_dt: List[float],
                use_remote_stepping: List[bool],
                name: str = "LRhcEnvBase",
                num_envs: int = 1,
                debug = False,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                n_init_step: int = 0,
                timeout_ms: int = 60000,
                env_opts: Dict = None,
                use_gpu: bool = True,
                dtype: torch.dtype = torch.float32,
                dump_basepath: str = "/tmp",
                override_low_lev_controller: bool = False):

        # checks on input args
        # type checks
        if not isinstance(robot_names, List):
            exception = "robot_names must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(robot_urdf_paths, List):
            exception = "robot_urdf_paths must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(robot_srdf_paths, List):
            exception = "robot_srdf_paths must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(cluster_dt, List):
            exception = "cluster_dt must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(use_remote_stepping, List):
            exception = "use_remote_stepping must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(n_contacts, List):
            exception = "n_contacts must be a list (of integers)!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(jnt_imp_config_paths, List):
            exception = "jnt_imp_config_paths must be a list paths!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
        # dim checks
        if not len(robot_urdf_paths) == len(robot_names):
            exception = f"robot_urdf_paths has len {len(robot_urdf_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(robot_srdf_paths) == len(robot_names):
            exception = f"robot_srdf_paths has len {len(robot_srdf_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(cluster_dt) == len(robot_names):
            exception = f"cluster_dt has len {len(cluster_dt)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(use_remote_stepping) == len(robot_names):
            exception = f"use_remote_stepping has len {len(use_remote_stepping)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(robot_srdf_paths) == len(robot_names):
            exception = f"robot_srdf_paths has len {len(robot_srdf_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(jnt_imp_config_paths) == len(robot_names):
            exception = f"jnt_imp_config_paths has len {len(jnt_imp_config_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        self._name=name
        self._num_envs=num_envs
        self._debug=debug
        self._verbose=verbose
        self._vlevel=vlevel
        self._force_reconnection=True
        self._timeout_ms=timeout_ms
        self._use_gpu=use_gpu
        self._device = "cuda" if self._use_gpu else "cpu"
        self._dtype=dtype
        self._robot_names=robot_names
        self._env_opts={}
        self._env_opts["deact_when_failure"]=True
        self._env_opts.update(env_opts)

        self.step_counter = 0 # global step counter
        self._n_init_steps = n_init_step # n steps to be performed before applying solutions from control clusters
        self._srdf_dump_paths = robot_srdf_paths
        self._homers = {} 
        self._homing = None
        self._jnt_imp_cntrl_shared_data = {}
        self._jnt_imp_controllers = {}
        self._jnt_imp_config_paths = {}

        # control cluster data
        self.cluster_sim_step_counters = {}
        self.cluster_servers = {}
        self._trigger_sol = {}
        self._wait_sol = {}
        self._cluster_dt = {}
        self._robot_urdf_paths={}
        self._robot_srdf_paths={}
        self._contact_names={}
        self._num_contacts={}

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._cluster_dt[robot_name]=cluster_dt[i]
            self._robot_urdf_paths[robot_name]=robot_urdf_paths[i]
            self._robot_srdf_paths[robot_name]=robot_srdf_paths[i]
            self._contact_names[robot_name]=None
            self._num_contacts[robot_name]=n_contacts[i]
            self._jnt_imp_config_paths[robot_name]=jnt_imp_config_paths[i]
        # db data
        self.debug_data = {}
        self.debug_data["time_to_step_world"] = np.nan
        self.debug_data["time_to_get_states_from_sim"] = np.nan
        self.debug_data["cluster_sol_time"] = {}
        self.debug_data["cluster_state_update_dt"] = {}
        self.debug_data["sim_time"] = {}
        self.debug_data["cluster_time"] = {}
        
        self._env_timer = time.perf_counter()

        # remote sim stepping options
        self._timeout = timeout_ms # timeout for remote stepping
        self._use_remote_stepping = use_remote_stepping
        # should use remote stepping
        self._remote_steppers = {}
        self._remote_resetters = {}
        self._remote_reset_requests = {}
        self._is_first_trigger = {}

        self._closed = False
             
        self._descr_dump_path=dump_basepath+"/"+f"{self.__class__.__name__}"
        self._urdf_dump_paths = {}
        self._srdf_dump_paths = {}
        self.xrdf_cmd_vals = [] # by default empty, needs to be overriden by
        # child class

        self._override_low_lev_controller=override_low_lev_controller

        self._root_p = {}
        self._root_q = {}
        self._jnts_q = {} 
        self._root_p_prev = {} # used for num differentiation
        self._root_q_prev = {} # used for num differentiation
        self._jnts_q_prev = {} # used for num differentiation
        self._root_v_prev = {} # used for num differentiation
        self._root_omega_prev = {} # used for num differentiation
        self._root_p_default = {} 
        self._root_q_default = {}
        self._jnts_q_default = {}
        
        self._gravity_normalized = {}
        self._gravity_normalized_base_loc = {}

        self._root_v = {}
        self._root_v_base_loc = {}
        self._root_v_default = {}
        self._root_omega = {}
        self._root_omega_base_loc = {}
        self._root_omega_default = {}
        self._root_a = {}
        self._root_a_base_loc = {}
        self._root_alpha = {}
        self._root_alpha_base_loc = {}

        self._jnts_v = {}
        self._jnts_v_default = {}
        self._jnts_eff = {}
        self._jnts_eff_default = {}

        self._root_pos_offsets = {} 
        self._root_q_offsets = {} 

        self._parse_env_opts()

        self._pre_setup() # child's method

        self._init_world() # after this point all info from sim or robot is 
        # available
        
        self._setup()

        signal.signal(signal.SIGINT, self.signal_handler)   

    def signal_handler(self, sig, frame):
        Journal.log(self.__class__.__name__,
            "signal_handler",
            "received SIGINT -> cleaning up",
            LogType.WARN)
        self.close()
    
    def __del__(self):
        self.close()
    
    def close(self) -> None:
        if not self._closed:
            for i in range(len(self._robot_names)):
                if self._robot_names[i] in self.cluster_servers:
                    self.cluster_servers[self._robot_names[i]].close()
                if self._use_remote_stepping[i]: # remote signaling
                    if self._robot_names[i] in self._remote_reset_requests:
                        self._remote_reset_requests[self._robot_names[i]].close()
                        self._remote_resetters[self._robot_names[i]].close()
                        self._remote_steppers[self._robot_names[i]].close()
                if self._robot_names[i] in self._jnt_imp_cntrl_shared_data:
                    jnt_imp_shared_data=self._jnt_imp_cntrl_shared_data[self._robot_names[i]]
                    if jnt_imp_shared_data is not None:
                        jnt_imp_shared_data.close()
            self._close()
            self._closed=True
    
    def _setup(self) -> None:
    
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]

            # normalized gravity vector
            self._gravity_normalized[robot_name]=torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._gravity_normalized[robot_name][:, 2]=-1.0
            self._gravity_normalized_base_loc[robot_name]=self._gravity_normalized[robot_name].detach().clone()

            self.cluster_sim_step_counters[robot_name]=0
            self._is_first_trigger[robot_name] = True
            if not isinstance(self._cluster_dt[robot_name], (float)):
                exception = f"cluster_dt[{i}] should be a float!"
                Journal.log(self.__class__.__name__,
                    "_setup",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            self._cluster_dt[robot_name] = self._cluster_dt[robot_name]
            self._trigger_sol[robot_name] = True # allow first trigger
            self._wait_sol[robot_name] = False

            # initialize a lrhc cluster server for communicating with rhc controllers
            self.cluster_servers[robot_name] = LRhcClusterServer(cluster_size=self._num_envs, 
                        cluster_dt=self._cluster_dt[robot_name], 
                        control_dt=self.physics_dt(), 
                        jnt_names=self._robot_jnt_names(robot_name=robot_name), 
                        n_contacts=self._n_contacts(robot_name=robot_name),
                        contact_linknames=self._contact_names[robot_name], 
                        verbose=self._verbose, 
                        vlevel=self._vlevel,
                        debug=self._debug, 
                        robot_name=robot_name,
                        use_gpu=self._use_gpu,
                        force_reconnection=self._force_reconnection,
                        timeout_ms=self._timeout)
            self.cluster_servers[robot_name].run()
            self.debug_data["cluster_sol_time"][robot_name] = np.nan
            self.debug_data["cluster_state_update_dt"][robot_name] = np.nan
            self.debug_data["sim_time"][robot_name] = np.nan
            # remote sim stepping
            if self._use_remote_stepping[i]:
                self._remote_steppers[robot_name] = RemoteStepperClnt(namespace=robot_name,
                                                            verbose=self._debug,
                                                            vlevel=self._vlevel)
                self._remote_resetters[robot_name] = RemoteResetClnt(namespace=robot_name,
                                                            verbose=self._debug,
                                                            vlevel=self._vlevel)
                self._remote_reset_requests[robot_name] = RemoteResetRequest(namespace=robot_name,
                                                                    n_env=self._num_envs,
                                                                    is_server=True,
                                                                    verbose=self._debug,
                                                                    vlevel=self._vlevel, 
                                                                    force_reconnection=self._force_reconnection, 
                                                                    safe=False)
                self._remote_steppers[robot_name].run()
                self._remote_resetters[robot_name].run()
                self._remote_reset_requests[robot_name].run()
            else:
                self._remote_steppers[robot_name] = None
                self._remote_reset_requests[robot_name] = None
                self._remote_resetters[robot_name] = None

            self._homers[robot_name] = RobotHomer(srdf_path=self._srdf_dump_paths[robot_name], 
                            jnt_names=self._robot_jnt_names(robot_name=robot_name),
                            filter=True)
            robot_homing=torch.from_numpy(self._homers[robot_name].get_homing().reshape(1,-1))
            if "cuda" in self._device:
                robot_homing=robot_homing.cuda()
            self._homing=robot_homing.repeat(self._num_envs, 1)

            self._jnts_q_default[robot_name] = self._homing
            self._set_jnts_homing(robot_name=robot_name)
            self._set_root_to_defconfig(robot_name=robot_name)
            self._reset_sim()
            self._init_safe_cluster_actions(robot_name=robot_name)

            self._jnt_imp_controllers[robot_name] = self._generate_jnt_imp_control(robot_name=robot_name)
            self._jnt_imp_cntrl_shared_data[robot_name] = JntImpCntrlData(is_server=True, 
                                            n_envs=self._num_envs, 
                                            n_jnts=len(self._robot_jnt_names(robot_name=robot_name)),
                                            jnt_names=self._robot_jnt_names(robot_name=robot_name),
                                            namespace=robot_name, 
                                            verbose=self._verbose, 
                                            force_reconnection=self._force_reconnection,
                                            vlevel=self._vlevel,
                                            use_gpu=self._use_gpu,
                                            safe=False)
            self._jnt_imp_cntrl_shared_data[robot_name].run()

            for n in range(self._n_init_steps): # run some initialization steps
                self._step_sim()
            self._read_jnts_state_from_robot(robot_name=robot_name,
                env_indxs=None)
            self._read_root_state_from_robot(robot_name=robot_name,
                    env_indxs=None)
            
            # write some inits for all robots
            self._update_root_offsets(robot_name)
            self._synch_default_root_states(robot_name=robot_name)
            epsi=0.08 # adding a bit of height to avoid initial penetration
            self._root_p_default[robot_name][:, 2]=self._root_p_default[robot_name][:, 2]+epsi
            
            self._reset(env_indxs=None,
                robot_name=robot_name,
                reset_cluster=True,
                reset_cluster_counter=False,
                randomize=True)
            self._set_startup_jnt_imp_gains(robot_name=robot_name)

            control_cluster=self.cluster_servers[robot_name]
            self._write_state_to_cluster(robot_name=robot_name)
            control_cluster.write_robot_state()
            control_cluster.pre_trigger()
            to_be_activated=control_cluster.get_inactive_controllers()
            if to_be_activated is not None:
                control_cluster.activate_controllers(
                    idxs=to_be_activated)       

            if self._use_remote_stepping[i]:
                self._wait_for_remote_step_req(robot_name=robot_name)

            control_cluster.trigger_solution()

        self._setup_done=True

    def step(self) -> bool:
        success=True

        if self._debug:
            self._pre_step_db()
            self._env_timer=time.perf_counter()
            self._step_sim()
            self.debug_data["time_to_step_world"] = \
                time.perf_counter() - self._env_timer
            self._post_sim_step_db()
        else:
            self._pre_step()
            self._step_sim()
            self._post_sim_step()

        return success
    
    def render(self, mode:str="human") -> None:
        self._render_sim(mode)

    def reset(self,
        env_indxs: torch.Tensor = None,
        reset_cluster: bool = False,
        reset_cluster_counter = False,
        randomize: bool = False,
        reset_sim: bool = False) -> None:

        for i in range(len(self._robot_names)):
            robot_name=self._robot_names[i]
            self._reset(robot_name=robot_name,
                env_indxs=env_indxs,
                randomize=randomize,
                reset_cluster=reset_cluster,
                reset_cluster_counter=reset_cluster_counter)
            self._set_startup_jnt_imp_gains(robot_name=robot_name,
                env_indxs=env_indxs)
            
        if reset_sim:
            self._reset_sim()

    def _reset_cluster(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            reset_cluster_counter: bool = False):
        
        control_cluster = self.cluster_servers[robot_name]

        control_cluster.reset_controllers(idxs=env_indxs)

        self._write_state_to_cluster(robot_name=robot_name,
            env_indxs=env_indxs)
        control_cluster.write_robot_state() # writes to shared memory

        if reset_cluster_counter:
            self.cluster_sim_step_counters[robot_name] = 0 

    def _write_state_to_cluster(self, 
        robot_name: str, 
        env_indxs: torch.Tensor = None,
        base_loc: bool = True):
        
        if self._debug:
            if not isinstance(env_indxs, (torch.Tensor, type(None))):
                msg = "Provided env_indxs should be a torch tensor of indexes!"
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)
            
        control_cluster = self.cluster_servers[robot_name]
        # floating base
        rhc_state = control_cluster.get_state()
        # configuration
        rhc_state.root_state.set(data=self.root_p_rel(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="p", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_q(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # twist
        rhc_state.root_state.set(data=self.root_v(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="v", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_omega(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="omega", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # angular accc.
        rhc_state.root_state.set(data=self.root_a(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="a", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_alpha(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="alpha", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # gravity vec
        rhc_state.root_state.set(data=self.gravity(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="gn", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # joints
        rhc_state.jnts_state.set(data=self.jnts_q(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.jnts_state.set(data=self.jnts_v(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="v", robot_idxs = env_indxs, gpu=self._use_gpu) 
        rhc_state.jnts_state.set(data=self.jnts_eff(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="eff", robot_idxs = env_indxs, gpu=self._use_gpu) 
        
        # Updating contact state for selected contact links
        self._update_contact_state(robot_name=robot_name, env_indxs=env_indxs)
    
    def _update_contact_state(self, 
            robot_name: str, 
            env_indxs: torch.Tensor = None):

        for i in range(0, self.cluster_servers[robot_name].n_contact_sensors()):
            contact_link = self.cluster_servers[robot_name].contact_linknames()[i]
            f_contact = self._get_contact_f(robot_name=robot_name,
                contact_link=contact_link,
                env_indxs=env_indxs)
            if f_contact is not None:
                self.cluster_servers[robot_name].get_state().contact_wrenches.set(data=f_contact, data_type="f",
                                contact_name=contact_link, 
                                robot_idxs = env_indxs, 
                                gpu=self._use_gpu)
                    
    def _init_safe_cluster_actions(self,
                            robot_name: str):

        # this does not actually write on shared memory, 
        # but it's enough to get safe actions for the simulator before the 
        # cluster starts to receive data from the controllers
        control_cluster = self.cluster_servers[robot_name]
        rhc_cmds = control_cluster.get_actions()
        n_jnts = rhc_cmds.n_jnts()
        
        null_action = torch.zeros((self._num_envs, n_jnts), 
                        dtype=self._dtype,
                        device=self._device)
        rhc_cmds.jnts_state.set(data=self._homing, data_type="q", gpu=self._use_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="v", gpu=self._use_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="eff", gpu=self._use_gpu)

    def _pre_step_db(self) -> None:
        
        # cluster step logic here
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            if self._override_low_lev_controller:
                # if overriding low-lev jnt imp. this has to run at the highest
                # freq possible
                start=time.perf_counter()
                self._read_jnts_state_from_robot(robot_name=robot_name)
                self.debug_data["time_to_get_states_from_sim"]= time.perf_counter()-start
                self._write_state_to_jnt_imp(robot_name=robot_name)
                self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

            control_cluster = self.cluster_servers[robot_name]
            if control_cluster.is_cluster_instant(self.cluster_sim_step_counters[robot_name]):
                control_cluster.wait_for_solution() # this is blocking
                failed = control_cluster.get_failed_controllers(gpu=self._use_gpu)
                self._set_cluster_actions(robot_name=robot_name) # write last cmds to low level control
                # read state necessary for cluster
                start=time.perf_counter()
                self._read_root_state_from_robot(robot_name=robot_name, 
                    env_indxs=None)
                if not self._override_low_lev_controller:
                    # we can update the jnt state just at the rate at which the cluster needs it
                    self._read_jnts_state_from_robot(robot_name=robot_name, env_indxs=None)
                    self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
                # write last robot state to the cluster of controllers
                self.debug_data["time_to_get_states_from_sim"]= time.perf_counter()-start

                start=time.perf_counter()
                self._write_state_to_cluster(robot_name=robot_name, 
                    env_indxs=None)
                control_cluster.write_robot_state()
                self.debug_data["cluster_state_update_dt"][robot_name] = time.perf_counter()-start

                self._update_jnt_imp_cntrl_shared_data() # only if debug_mode_jnt_imp is enabled

                if self._use_remote_stepping[i]:
                    self._remote_steppers[robot_name].ack() # signal cluster stepping is finished
                    if failed is not None and self._env_opts["deact_when_failure"]: # deactivate robot completely 
                        self._deactivate(env_indxs=failed,
                            robot_name=robot_name)
                    self._process_remote_reset_req(robot_name=robot_name) # wait for remote reset request (blocking)
                    self._wait_for_remote_step_req(robot_name=robot_name)
                else:
                    if failed is not None:
                        self._reset(env_indxs=failed,
                            robot_name=robot_name,
                            reset_cluster=True,
                            reset_cluster_counter=False,
                            randomize=True)
                        self._set_startup_jnt_imp_gains(robot_name=robot_name,
                            env_indxs=failed)

                    control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

                control_cluster.pre_trigger() # performs pre-trigger steps, like retrieving
                # values of some rhc flags on shared memory
                control_cluster.trigger_solution() # trigger only active controllers

    def _pre_step(self) -> None:
        
        # cluster step logic here
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            if self._override_low_lev_controller:
                # if overriding low-lev jnt imp. this has to run at the highest
                # freq possible
                self._read_jnts_state_from_robot(robot_name=robot_name)
                self._write_state_to_jnt_imp(robot_name=robot_name)
                self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

            control_cluster = self.cluster_servers[robot_name]
            if control_cluster.is_cluster_instant(self.cluster_sim_step_counters[robot_name]):
                control_cluster.wait_for_solution() # this is blocking
                failed = control_cluster.get_failed_controllers(gpu=self._use_gpu)
                self._set_cluster_actions(robot_name=robot_name) # write last cmds to low level control
                # read state necessary for cluster
                self._read_root_state_from_robot(robot_name=robot_name, 
                    env_indxs=None)
                if not self._override_low_lev_controller:
                    # we can update the jnt state just at the rate at which the cluster needs it
                    self._read_jnts_state_from_robot(robot_name=robot_name, env_indxs=None)
                    self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
                # write last robot state to the cluster of controllers
                self._write_state_to_cluster(robot_name=robot_name, 
                    env_indxs=None)
                control_cluster.write_robot_state()

                if self._use_remote_stepping[i]:
                    self._remote_steppers[robot_name].ack() # signal cluster stepping is finished
                    if failed is not None and self._env_opts["deact_when_failure"]:
                        self._deactivate(env_indxs=failed,
                            robot_name=robot_name)
                    self._process_remote_reset_req(robot_name=robot_name) # wait for remote reset request (blocking)
                    self._wait_for_remote_step_req(robot_name=robot_name)
                else:
                    if failed is not None:
                        self._reset(env_indxs=failed,
                            robot_name=robot_name,
                            reset_cluster=True,
                            reset_cluster_counter=False,
                            randomize=True)
                        self._set_startup_jnt_imp_gains(robot_name=robot_name,
                            env_indxs=failed)
                    control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

                control_cluster.pre_trigger() # performs pre-trigger steps, like retrieving
                # values of some rhc flags on shared memory
                control_cluster.trigger_solution() # trigger only active controllers
                    
    def _post_sim_step_db(self) -> bool:

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            control_cluster = self.cluster_servers[robot_name]
            self.cluster_sim_step_counters[robot_name]+=1 # this has to be update with sim freq
            if self._debug:
                self.debug_data["sim_time"][robot_name]=self.cluster_sim_step_counters[robot_name]*self.physics_dt()
                self.debug_data["cluster_sol_time"][robot_name] = \
                    control_cluster.solution_time()
                
        self.step_counter +=1

    def _post_sim_step(self) -> bool:

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            self.cluster_sim_step_counters[robot_name]+=1
        self.step_counter +=1
                    
    def _reset(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False,
            reset_cluster: bool = False,
            reset_cluster_counter = False):
        
        # resets the state of target robot and env to the defaults
        self._reset_state(env_indxs=env_indxs, 
            robot_name=robot_name,
            randomize=randomize)
        # and jnt imp. controllers
        self._reset_jnt_imp_control(robot_name=robot_name,
                env_indxs=env_indxs)
        
        self._read_jnts_state_from_robot(robot_name=robot_name,
            env_indxs=env_indxs)
        self._read_root_state_from_robot(robot_name=robot_name,
                env_indxs=env_indxs)
        
        if reset_cluster: # reset controllers remotely
            self._reset_cluster(env_indxs=env_indxs,
                robot_name=robot_name,
                reset_cluster_counter=reset_cluster_counter)
        
    def _randomize_yaw(self,
            robot_name: str,
            env_indxs: torch.Tensor = None):

        root_q_default = self._root_q_default[robot_name]
        if env_indxs is None:
            env_indxs = torch.arange(root_q_default.shape[0])

        num_indices = env_indxs.shape[0]
        yaw_angles = torch.rand((num_indices,), 
                        device=root_q_default.device) * 2 * torch.pi  # uniformly distributed random angles
        
        # Compute cos and sin once
        cos_half = torch.cos(yaw_angles / 2)
        root_q_default[env_indxs, :] = torch.stack((cos_half, 
                                torch.zeros_like(cos_half),
                                torch.zeros_like(cos_half), 
                                torch.sin(yaw_angles / 2)), dim=1).reshape(num_indices, 4)

    def _deactivate(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        
        # deactivate jnt imp controllers for given robots and envs (makes the robot fall)
        self._jnt_imp_controllers[robot_name].deactivate(robot_indxs=env_indxs)
    
    def _n_contacts(self, robot_name: str) -> List[int]:
        return self._num_contacts[robot_name]
    
    def root_p(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_p[robot_name]
        else:
            return self._root_p[robot_name][env_idxs, :]

    def root_p_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        rel_pos = torch.sub(self.root_p(robot_name=robot_name,
                                            env_idxs=env_idxs), 
                self._root_pos_offsets[robot_name][env_idxs, :])
        return rel_pos
    
    def root_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_q[robot_name]
        else:
            return self._root_q[robot_name][env_idxs, :]

    def root_q_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        rel_q = quaternion_difference(self._root_q_offsets[robot_name][env_idxs, :], 
                            self.root_q(robot_name=robot_name,
                                            env_idxs=env_idxs))
        return rel_q
    
    def root_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_v=self._root_v[robot_name]
        if base_loc:
            root_v=self._root_v_base_loc[robot_name]
        if env_idxs is None:
            return root_v
        else:
            return root_v[env_idxs, :]
    
    def root_omega(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_omega=self._root_omega[robot_name]
        if base_loc:
            root_omega=self._root_omega_base_loc[robot_name]
        if env_idxs is None:
            return root_omega
        else:
            return root_omega[env_idxs, :]
    
    def root_a(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_a=self._root_a[robot_name]
        if base_loc:
            root_a=self._root_a_base_loc[robot_name]
        if env_idxs is None:
            return root_a
        else:
            return root_a[env_idxs, :]
    
    def root_alpha(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_alpha=self._root_alpha[robot_name]
        if base_loc:
            root_alpha=self._root_alpha_base_loc[robot_name]
        if env_idxs is None:
            return root_alpha
        else:
            return root_alpha[env_idxs, :]
        
    def gravity(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        gravity_loc=self._gravity_normalized[robot_name]
        if base_loc:
            gravity_loc=self._gravity_normalized_base_loc[robot_name]
        if env_idxs is None:
            return gravity_loc
        else:
            return gravity_loc[env_idxs, :]
    
    def jnts_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):
        
        if env_idxs is None:
            return self._jnts_q[robot_name]
        else:
            return self._jnts_q[robot_name][env_idxs, :]

    def jnts_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._jnts_v[robot_name]
        else:
            return self._jnts_v[robot_name][env_idxs, :]

    def jnts_eff(self,
            robot_name: str,
            env_idxs: torch.Tensor = None): # (measured) efforts

        if env_idxs is None:
            return self._jnts_eff[robot_name]
        else:
            return self._jnts_eff[robot_name][env_idxs, :]

    def _wait_for_remote_step_req(self,
            robot_name: str):
        if not self._remote_steppers[robot_name].wait(self._timeout):
            self.close()
            Journal.log(self.__class__.__name__,
                "_wait_for_remote_step_req",
                "Didn't receive any remote step req within timeout!",
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _process_remote_reset_req(self,
            robot_name: str):
        
        if not self._remote_resetters[robot_name].wait(self._timeout):
            self.close()
            Journal.log(self.__class__.__name__,
                "_process_remote_reset_req",
                "Didn't receive any remote reset req within timeout!",
                LogType.EXCEP,
                throw_when_excep = True)
            
        reset_requests = self._remote_reset_requests[robot_name]
        reset_requests.synch_all(read=True, retry=True) # read reset requests from shared mem
        to_be_reset = reset_requests.to_be_reset(gpu=self._use_gpu)
        if to_be_reset is not None:
            self._reset(env_indxs=to_be_reset,
                robot_name=robot_name,
                reset_cluster=True,
                reset_cluster_counter=False,
                randomize=True)
            self._set_startup_jnt_imp_gains(robot_name=robot_name,
                env_indxs=to_be_reset)
        control_cluster = self.cluster_servers[robot_name]
        control_cluster.activate_controllers(idxs=to_be_reset) # activate controllers
        # (necessary if failed)

        self._remote_resetters[robot_name].ack() # signal reset performed

    def _update_jnt_imp_cntrl_shared_data(self):
        if self._debug:
            for i in range(0, len(self._robot_names)):
                robot_name = self._robot_names[i]
                # updating all the jnt impedance data - > this may introduce some overhead
                imp_data = self._jnt_imp_cntrl_shared_data[robot_name].imp_data_view
                # set data
                imp_data.set(data_type="pos_err",
                        data=self._jnt_imp_controllers[robot_name].pos_err(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel_err",
                        data=self._jnt_imp_controllers[robot_name].vel_err(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="pos_gains",
                        data=self._jnt_imp_controllers[robot_name].pos_gains(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel_gains",
                        data=self._jnt_imp_controllers[robot_name].vel_gains(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="eff_ff",
                        data=self._jnt_imp_controllers[robot_name].eff_ref(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="pos",
                        data=self._jnt_imp_controllers[robot_name].pos(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="pos_ref",
                        data=self._jnt_imp_controllers[robot_name].pos_ref(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel",
                        data=self._jnt_imp_controllers[robot_name].vel(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel_ref",
                        data=self._jnt_imp_controllers[robot_name].vel_ref(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="eff",
                        data=self._jnt_imp_controllers[robot_name].eff(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="imp_eff",
                        data=self._jnt_imp_controllers[robot_name].imp_eff(),
                        gpu=self._use_gpu)
                # copy from GPU to CPU if using gpu
                if self._use_gpu:
                    imp_data.synch_mirror(from_gpu=True,non_blocking=True)
                    # even if it's from GPU->CPu we can use non-blocking since it's just for db 
                    # purposes
                # write copies to shared memory
                imp_data.synch_all(read=False, retry=False)

    def _set_startup_jnt_imp_gains(self,
            robot_name:str, 
            env_indxs: torch.Tensor = None):
        
        startup_p_gains=self._jnt_imp_controllers[robot_name].startup_p_gains()
        startup_d_gains=self._jnt_imp_controllers[robot_name].startup_d_gains()

        if env_indxs is not None:
            self._jnt_imp_controllers[robot_name].set_gains(robot_indxs=env_indxs,
                pos_gains=startup_p_gains[env_indxs, :], 
                vel_gains=startup_d_gains[env_indxs, :])
        else:
            self._jnt_imp_controllers[robot_name].set_gains(robot_indxs=env_indxs,
                pos_gains=startup_p_gains[:, :], 
                vel_gains=startup_d_gains[:, :])
            
    def _write_state_to_jnt_imp(self,
        robot_name: str):
     
        # always update ,imp. controller internal state (jnt imp control is supposed to be
        # always running)
        self._jnt_imp_controllers[robot_name].update_state(pos=self.jnts_q(robot_name=robot_name), 
            vel = self.jnts_v(robot_name=robot_name),
            eff = self.jnts_eff(robot_name=robot_name))

    def _set_cluster_actions(self, 
        robot_name):
        control_cluster = self.cluster_servers[robot_name]
        actions=control_cluster.get_actions()
        active_controllers=control_cluster.get_active_controllers(gpu=self._use_gpu)
        
        if active_controllers is not None:
            self._jnt_imp_controllers[robot_name].set_refs(
                pos_ref=actions.jnts_state.get(data_type="q", gpu=self._use_gpu)[active_controllers, :], 
                vel_ref=actions.jnts_state.get(data_type="v", gpu=self._use_gpu)[active_controllers, :], 
                eff_ref=actions.jnts_state.get(data_type="eff", gpu=self._use_gpu)[active_controllers, :],
                robot_indxs=active_controllers)            
    
    def _apply_cmds_to_jnt_imp_control(self, robot_name:str):

        self._jnt_imp_controllers[robot_name].apply_cmds()

    def _update_root_offsets(self, 
                    robot_name: str,
                    env_indxs: torch.Tensor = None):
        
        if self._debug:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):                
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "update_root_offsets",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)    
                if self._use_gpu:
                    if not env_indxs.device.type == "cuda":
                            error = "Provided env_indxs should be on GPU!"
                            Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                else:
                    if not env_indxs.device.type == "cpu":
                        error = "Provided env_indxs should be on CPU!"
                        Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                    "update_root_offsets",
                    f"updating root offsets " + for_robots,
                    LogType.STAT,
                    throw_when_excep = True)

        # only planar position used
        if env_indxs is None:
            self._root_pos_offsets[robot_name][:, 0:2]  = self._root_p[robot_name][:, 0:2]
            self._root_q_offsets[robot_name][:, :]  = self._root_q[robot_name]
            
        else:
            self._root_pos_offsets[robot_name][env_indxs, 0:2]  = self._root_p[robot_name][env_indxs, 0:2]
            self._root_q_offsets[robot_name][env_indxs, :]  = self._root_q[robot_name][env_indxs, :]

    def _reset_jnt_imp_control(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None):
        
        if self._debug:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    Journal.log(self.__class__.__name__,
                        "reset_jnt_imp_control",
                        "Provided env_indxs should be a torch tensor of indexes!",
                        LogType.EXCEP,
                        throw_when_excep = True)
                if self._use_gpu:
                    if not env_indxs.device.type == "cuda":
                            error = "Provided env_indxs should be on GPU!"
                            Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                else:
                    if not env_indxs.device.type == "cpu":
                        error = "Provided env_indxs should be on CPU!"
                        Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs)
                                
            if self._verbose:
                Journal.log(self.__class__.__name__,
                    "reset_jnt_imp_control",
                    f"resetting joint impedances " + for_robots,
                    LogType.STAT,
                    throw_when_excep = True)

        # resets all internal data, refs to defaults
        self._jnt_imp_controllers[robot_name].reset(robot_indxs=env_indxs)
        
        #restore jnt imp refs to homing            
        if env_indxs is None:                               
            self._jnt_imp_controllers[robot_name].set_refs(pos_ref=self._homing[:, :],
                robot_indxs = None)
        else:
            self._jnt_imp_controllers[robot_name].set_refs(pos_ref=self._homing[env_indxs, :],
                robot_indxs = env_indxs)

        # self._write_state_to_jnt_imp(robot_name=robot_name)
        # actually applies reset commands to the articulation
        self._write_state_to_jnt_imp(robot_name=robot_name)
        self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

    def _synch_default_root_states(self,
            robot_name: str,
            env_indxs: torch.Tensor = None):

        if self._debug:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "synch_default_root_states",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)  
                if self._use_gpu:
                    if not env_indxs.device.type == "cuda":
                            error = "Provided env_indxs should be on GPU!"
                            Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                else:
                    if not env_indxs.device.type == "cpu":
                        error = "Provided env_indxs should be on CPU!"
                        Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)  
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                            "synch_default_root_states",
                            f"updating default root states " + for_robots,
                            LogType.STAT,
                            throw_when_excep = True)

        if env_indxs is None:
            self._root_p_default[robot_name][:, :] = self._root_p[robot_name]
            self._root_q_default[robot_name][:, :] = self._root_q[robot_name]
        else:
            self._root_p_default[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :]
            self._root_q_default[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]

    def _generate_rob_descriptions(self, 
                    robot_name: str, 
                    urdf_path: str,
                    srdf_path: str):
        
        custom_xacro_args=extract_custom_xacro_args(self._env_opts)
        Journal.log(self.__class__.__name__,
                    "_generate_rob_descriptions",
                    "generating URDF for robot "+ f"{robot_name}, from URDF {urdf_path}...",
                    LogType.STAT,
                    throw_when_excep = True)
        xrdf_cmds=self._xrdf_cmds(robot_name=robot_name)
        xrdf_cmds=merge_xacro_cmds(prev_cmds=xrdf_cmds,
            new_cmds=custom_xacro_args)
        self._urdf_dump_paths[robot_name]=generate_urdf(robot_name=robot_name, 
            xacro_path=urdf_path,
            dump_path=self._descr_dump_path,
            xrdf_cmds=xrdf_cmds)
        Journal.log(self.__class__.__name__,
                    "_generate_rob_descriptions",
                    "generating SRDF for robot "+ f"{robot_name}, from SRDF {srdf_path}...",
                    LogType.STAT,
                    throw_when_excep = True)
        # we also generate SRDF files, which are useful for control
        self._srdf_dump_paths[robot_name]=generate_srdf(robot_name=robot_name, 
            xacro_path=srdf_path,
            dump_path=self._descr_dump_path,
            xrdf_cmds=xrdf_cmds)
    
    def _xrdf_cmds(self, robot_name:str):
        urdfpath=self._robot_urdf_paths[robot_name]
        # we assume directory tree of the robot package is like
        # robot-ros-pkg/robot_urdf/urdf/robot.urdf.xacro
        parts = urdfpath.split('/')
        urdf_descr_root_path = '/'.join(parts[:-2])
        cmds = get_xrdf_cmds(urdf_descr_root_path=urdf_descr_root_path) 
        return cmds
    
    def _reset_jnt_imp_control_gains(self,
        robot_name: str, 
        env_indxs: torch.Tensor = None):
        pass

    @abstractmethod
    def current_tstep(self) -> int:
        pass
    
    @abstractmethod
    def current_time(self) -> float:
        pass
    
    @abstractmethod
    def _sim_is_running(self) -> bool:
        pass
    
    @abstractmethod
    def _get_contact_f(self, 
        robot_name: str, 
        contact_link: str,
        env_indxs: torch.Tensor) -> torch.Tensor:
        return None
    
    @abstractmethod
    def physics_dt(self) -> float:
        pass
    
    @abstractmethod
    def rendering_dt(self) -> float:
        pass
    
    @abstractmethod
    def set_physics_dt(self, physics_dt:float):
        pass

    @abstractmethod
    def set_rendering_dt(self, rendering_dt:float):
        pass

    @abstractmethod
    def _robot_jnt_names(self, robot_name: str) -> List[str]:
        pass
    
    @abstractmethod
    def _read_root_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        pass
    
    @abstractmethod
    def _read_jnts_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        pass

    @abstractmethod
    def _init_robots_state(self):
        pass

    @abstractmethod
    def _reset_state(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False):
        pass
    
    @abstractmethod
    def _init_world(self):
        pass

    @abstractmethod
    def _reset_sim(self) -> None:
        pass
    
    @abstractmethod
    def _set_jnts_homing(self, robot_name: str):
        pass

    @abstractmethod
    def _set_root_to_defconfig(self, robot_name: str):
        pass

    @abstractmethod
    def _parse_env_opts(self):
        pass
    
    @abstractmethod
    def _pre_setup(self):
        pass

    @abstractmethod
    def _generate_jnt_imp_control(self) -> JntImpCntrlChild:
        pass

    @abstractmethod
    def _render_sim(self, mode:str="human") -> None:
        pass

    @abstractmethod
    def _close(self) -> None:
        pass

    @abstractmethod
    def _step_sim(self) -> None:
        pass