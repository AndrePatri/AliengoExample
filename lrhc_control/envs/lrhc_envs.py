
from EigenIPC.PyEigenIPC import VLevel, Journal, LogType

from lrhc_control.envs.lrhc_sim_env_base import LRhcEnvBase

class LRhcIsaacSimEnv(IsaacSimEnv):

    def __init__(self,
                headless: bool = True, 
                sim_device: int = 0, 
                enable_livestream: bool = False, 
                enable_viewport: bool = False,
                debug = False,
                timeout_ms: int = 60000):

        super().__init__(headless = headless, 
                sim_device = sim_device, 
                enable_livestream = enable_livestream, 
                enable_viewport = enable_viewport,
                debug = debug)

        # debug data
        self.debug_data = {}
        self.debug_data["time_to_step_world"] = np.nan
        self.debug_data["time_to_get_states_from_sim"] = np.nan
        self.debug_data["cluster_sol_time"] = {}
        self.debug_data["cluster_state_update_dt"] = {}
        self.debug_data["sim_time"] = {}
        self.debug_data["cluster_time"] = {}

        self._is_first_trigger = {}
        self.cluster_timers = {}
        self.env_timer = time.perf_counter()

        self._global_step_counter = 0
        self.cluster_sim_step_counters = {}
        self.cluster_servers = {}
        self._trigger_sol = {}
        self._wait_sol = {}
        self._cluster_dt = {}
        # remote simulation
        self._timeout = timeout_ms
        self._init_steps_done = False
        self._n_init_steps = 0 # n steps to be performed before waiting for remote stepping
        self._init_step_counter = 0
        self._use_remote_stepping = [] # whether the task associated with robot i should use remote stepping
        self._remote_steppers = {}
        self._remote_resetters = {}
        self._remote_reset_requests = {}
        
    def _pre_physics_step(self):
        
        self.task.get_states() # gets data from simulation (jnt imp control always needs updated state)

        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]
            control_cluster = self.cluster_servers[robot_name] # retrieve control
            active = None 
            just_activated = None
            just_deactivated = None                
            failed = None
            
            # 1) this runs @cluster_dt: wait + retrieve latest solution
            if control_cluster.is_cluster_instant(self.cluster_sim_step_counters[robot_name]) and \
                    self._init_steps_done:
                
                control_cluster.pre_trigger() # performs pre-trigger steps, like retrieving
                # values of some rhc flags on shared memory

                if not self._is_first_trigger[robot_name]: # first trigger we don't wait (there has been no trigger)
                    control_cluster.wait_for_solution() # this is blocking
                    if self.debug:
                        self.cluster_timers[robot_name] = time.perf_counter()
                    failed = control_cluster.get_failed_controllers(gpu=self.using_gpu)
                    if self._use_remote_stepping[i]:
                        self._remote_steppers[robot_name].ack() # signal cluster stepping is finished
                        if failed is not None: # deactivate robot completely (basically, deactivate jnt imp controller)
                            self.task.deactivate(env_indxs = failed,
                                robot_names = [robot_name])
                        self._process_remote_reset_req(robot_name=robot_name)
                    else:
                        # automatically reset and reactivate failed controllers if not running remotely
                        if failed is not None:
                            self.reset(env_indxs=failed,
                                    robot_names=[robot_name],
                                    reset_world=False,
                                    reset_cluster=True,
                                    randomize=False)
                        # activate inactive controllers
                        control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

            active = control_cluster.get_active_controllers(gpu=self.using_gpu)
            # 2) this runs at @simulation dt i.e. the highest possible rate,
            # using the latest available RHC actions
            self.task.pre_physics_step(robot_name = robot_name, 
                        actions = control_cluster.get_actions(),
                        env_indxs = active)
            # 3) this runs at @cluster_dt: trigger cluster solution
            if control_cluster.is_cluster_instant(self.cluster_sim_step_counters[robot_name]) and \
                    self._init_steps_done:
                if self._is_first_trigger[robot_name]:
                        # we update the all the default root state now. This state will be used 
                        # for future resets
                        self.task.synch_default_root_states(robot_name = robot_name)
                        self._is_first_trigger[robot_name] = False
                if self._use_remote_stepping[i]:
                        self._wait_for_remote_step_req(robot_name=robot_name)
                # handling controllers transition to running state
                just_activated = control_cluster.get_just_activated(gpu=self.using_gpu) 
                just_deactivated = control_cluster.get_just_deactivated(gpu=self.using_gpu)
                if just_activated is not None: # transition of some controllers from not active -> active
                    self.task.update_root_offsets(robot_name,
                                    env_indxs = just_activated) # we use relative state wrt to this state for the controllers
                    self.task.update_jnt_imp_control_gains(robot_name = robot_name, 
                                    jnt_stiffness = self.task.startup_jnt_stiffness, 
                                    jnt_damping = self.task.startup_jnt_damping, 
                                    wheel_stiffness = self.task.startup_wheel_stiffness, 
                                    wheel_damping = self.task.startup_wheel_damping,
                                    env_indxs = just_activated) # setting runtime state for jnt imp controller
                if just_deactivated is not None:
                    self.task.reset_jnt_imp_control(robot_name=robot_name,
                            env_indxs=just_deactivated)
                self._update_cluster_state(robot_name=robot_name, 
                                    env_indxs=active) # always update cluster state every cluster dt
                control_cluster.trigger_solution() # trigger only active controllers
            
    def _init_safe_cluster_actions(self,
                            robot_name: str):

        # this does not actually write on shared memory, 
        # but it's enough to get safe actions for the simulator before the 
        # cluster starts to receive data from the controllers
        control_cluster = self.cluster_servers[robot_name]
        rhc_cmds = control_cluster.get_actions()
        n_jnts = rhc_cmds.n_jnts()
        homing = self.task.homers[robot_name].get_homing()
        null_action = torch.zeros((self.task.num_envs, n_jnts), 
                        dtype=self.task.torch_dtype,
                        device=self.task.torch_device)
        rhc_cmds.jnts_state.set(data=homing, data_type="q", gpu=self.using_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="v", gpu=self.using_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="eff", gpu=self.using_gpu)

    def _post_physics_step(self):

        self._global_step_counter +=1
        if self._global_step_counter == self._n_init_steps and \
                not self._init_steps_done:
            self._init_steps_done = True # next cluster step we wait for connection to remote client
        for i in range(len(self.robot_names)):
            robot_name = self.robot_names[i]
            self.cluster_sim_step_counters[robot_name] += 1

            # if self._use_remote_stepping[i] and self._init_steps_done:
            #     self._signal_sim_env_is_ready(robot_name=robot_name) # signal sim is ready
           
            # if self.debug:
            self.debug_data["sim_time"][robot_name]=self.cluster_sim_step_counters[robot_name]*self.get_task().integration_dt()
            self.debug_data["cluster_sol_time"][robot_name] = \
                self.cluster_servers[robot_name].solution_time()
            self.debug_data["cluster_time"] = self.cluster_servers[robot_name]

    def step(self):
        
        self._pre_physics_step()

        # 3) simulation stepping (@ integration_dt) (for all robots and all environments)
        self._step_world()

        self._post_physics_step()

    def set_task(self, 
                task, 
                cluster_dt: List[float], 
                use_remote_stepping: List[bool],
                n_pre_training_steps = 0,
                backend="torch", 
                sim_params=None, 
                init_sim=True, 
                cluster_client_verbose = False, 
                cluster_client_debug = False,
                verbose: bool = True,
                vlevel: VLevel = VLevel.V1) -> None:

        super().set_task(task, 
                backend=backend, 
                sim_params=sim_params, 
                init_sim=init_sim)
        
        self.using_gpu = task.using_gpu

        self.robot_names = self.task.robot_names
        self.robot_pkg_names = self.task.robot_pkg_names
        self._use_remote_stepping = use_remote_stepping
        self._n_init_steps = n_pre_training_steps
        
        if not isinstance(cluster_dt, List):
            exception = "cluster_dt must be a list!"
            Journal.log(self.__class__.__name__,
                "set_task",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not (len(cluster_dt) == len(self.robot_names)):
            exception = f"cluster_dt length{len(cluster_dt)} does not match robot number {len(self.robot_names)}!"
            Journal.log(self.__class__.__name__,
                "set_task",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not (len(use_remote_stepping) == len(self.robot_names)):
            exception = f"use_remote_stepping length{len(use_remote_stepping)} does not match robot number {len(self.robot_names)}!"
            Journal.log(self.__class__.__name__,
                "set_task",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        # now the task and the simulation is guaranteed to be initialized
        # -> we have the data to initialize the cluster client
        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]
            self.cluster_sim_step_counters[robot_name] = 0
            self._is_first_trigger[robot_name] = True
            if not isinstance(cluster_dt[i], float):
                exception = f"cluster_dt should be a list of float values!"
                Journal.log(self.__class__.__name__,
                    "set_task",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self._cluster_dt[robot_name] = cluster_dt[i]
            self._trigger_sol[robot_name] = True # allow first trigger
            self._wait_sol[robot_name] = False
            if task.omni_contact_sensors[robot_name] is not None:
                n_contact_sensors = task.omni_contact_sensors[robot_name].n_sensors
                contact_names = task.omni_contact_sensors[robot_name].contact_prims
            else:
                n_contact_sensors = 4
                contact_names = None
                
            self.cluster_servers[robot_name] = LRhcClusterServer(cluster_size=task.num_envs, 
                        cluster_dt=self._cluster_dt[robot_name], 
                        control_dt=task.integration_dt(), 
                        jnt_names = task.robot_dof_names[robot_name], 
                        n_contacts = n_contact_sensors,
                        contact_linknames = contact_names, 
                        verbose = cluster_client_verbose, 
                        vlevel = vlevel,
                        debug = cluster_client_debug, 
                        robot_name=robot_name,
                        use_gpu = task.using_gpu,
                        force_reconnection=True,
                        timeout_ms=self._timeout)
            self.cluster_servers[robot_name].run()
            self._init_safe_cluster_actions(robot_name=robot_name)                    

            self.debug_data["cluster_sol_time"][robot_name] = np.nan
            self.debug_data["cluster_state_update_dt"][robot_name] = np.nan
            self.debug_data["sim_time"][robot_name] = np.nan
            self.debug_data["cluster_time"][robot_name] = np.nan
            if self.debug:
                self.cluster_timers[robot_name] = time.perf_counter()

            if self._use_remote_stepping[i]:
                self._remote_steppers[robot_name] = RemoteStepperClnt(namespace=robot_name,
                                                            verbose=verbose,
                                                            vlevel=vlevel)
                self._remote_resetters[robot_name] = RemoteResetClnt(namespace=robot_name,
                                                            verbose=verbose,
                                                            vlevel=vlevel)
                self._remote_reset_requests[robot_name] = RemoteResetRequest(namespace=robot_name,
                                                                    n_env=self.num_envs,
                                                                    is_server=True,
                                                                    verbose=verbose,
                                                                    vlevel=vlevel, 
                                                                    force_reconnection=True, 
                                                                    safe=False)
                self._remote_steppers[robot_name].run()
                self._remote_resetters[robot_name].run()
                self._remote_reset_requests[robot_name].run()
            else:
                self._remote_steppers[robot_name] = None
                self._remote_reset_requests[robot_name] = None
                self._remote_resetters[robot_name] = None
             
    def reset(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str]=None,
            reset_world: bool = False,
            reset_cluster: bool = False,
            reset_counter = False,
            randomize: bool = False):

        if reset_cluster:
            # reset controllers remotely
            self._reset_cluster(env_indxs=env_indxs,
                    robot_names=robot_names)
        if reset_world:
            self._world.reset()
        self.task.reset(env_indxs = env_indxs,
            robot_names = robot_names,
            randomize=randomize) # resets articulations in sim, the meas.
        # states and the jnt imp controllers
        rob_names = robot_names
        if rob_names is None:
            rob_names = self.robot_names
        # also reset the state of cluster using the reset state
        if reset_cluster:
            for i in range(len(rob_names)):
                self._update_cluster_state(robot_name=rob_names[i],
                                env_indxs=env_indxs)
        if reset_counter:
            for i in range(len(rob_names)):
                self.cluster_sim_step_counters[rob_names[i]] = 0
    
    def close(self):

        for i in range(len(self.robot_names)):
            self.cluster_servers[self.robot_names[i]].close()
            if self._use_remote_stepping[i]: # remote signaling
                self._remote_reset_requests[self.robot_names[i]].close()
                self._remote_resetters[self.robot_names[i]].close()
                self._remote_steppers[self.robot_names[i]].close()
        self.task.close() # performs closing steps for task
        super().close() # this has to be called last 
        # so that isaac's simulation is closed properly

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
        to_be_reset = reset_requests.to_be_reset(gpu=self.using_gpu)
        if to_be_reset is not None:
            self.reset(env_indxs=to_be_reset,
                robot_names=[robot_name],
                reset_world=False,
                reset_cluster=True,
                randomize=True)
        control_cluster = self.cluster_servers[robot_name]
        control_cluster.activate_controllers(idxs=to_be_reset) # activate controllers
        # (necessary if failed)

        self._remote_resetters[robot_name].ack() # signal reset performed

    def _step_world(self):
        if self.debug:
            self.env_timer = time.perf_counter()
        self._world.step(render=self._render)
        if self.debug:
            self.debug_data["time_to_step_world"] = time.perf_counter() - self.env_timer

    def _update_contact_state(self, 
                    robot_name: str, 
                    env_indxs: torch.Tensor = None):

        for i in range(0, self.cluster_servers[robot_name].n_contact_sensors()):
            contact_link = self.cluster_servers[robot_name].contact_linknames()[i]
            if not self.using_gpu:
                contact_sensors=self.task.omni_contact_sensors[robot_name]
                if contact_sensors is not None:
                    f_contact = self.task.omni_contact_sensors[robot_name].get(dt = self.task.integration_dt(), 
                                            contact_link = contact_link,
                                            env_indxs = env_indxs,
                                            clone = False)
                    # assigning measured net contact forces
                    self.cluster_servers[robot_name].get_state().contact_wrenches.set(data=f_contact, data_type="f",
                            contact_name=contact_link, robot_idxs = env_indxs, gpu=self.using_gpu)
            else:
                if (self.debug and ((self.cluster_sim_step_counters[robot_name] + 1) % 10000) == 0):
                    # sporadic warning
                    warning = f"Contact state from link {contact_link} cannot be retrieved in IsaacSim if using use_gpu_pipeline is set to True!"
                    Journal.log(self.__class__.__name__,
                        "_update_contact_state",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
        
    def _update_cluster_state(self, 
                    robot_name: str, 
                    env_indxs: torch.Tensor = None):
        
        if self.debug:
            if not isinstance(env_indxs, Union[torch.Tensor, None]):
                msg = "Provided env_indxs should be a torch tensor of indexes!"
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)
            if env_indxs is not None:
                if not len(env_indxs.shape) == 1:
                    msg = "Provided env_indxs should be a 1D torch tensor!"
                    raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)

        control_cluster = self.cluster_servers[robot_name]
        # floating base
        relative_pos = self.task.root_p_rel(robot_name=robot_name,
                                            env_idxs=env_indxs)
        rhc_state = control_cluster.get_state()
        rhc_state.root_state.set(data=relative_pos, 
                data_type="p", robot_idxs = env_indxs, gpu=self.using_gpu)
        rhc_state.root_state.set(data=self.task.root_q(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="q", robot_idxs = env_indxs, gpu=self.using_gpu)
        rhc_state.root_state.set(data=self.task.root_v(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="v", robot_idxs = env_indxs, gpu=self.using_gpu)
        rhc_state.root_state.set(data=self.task.root_omega(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="omega", robot_idxs = env_indxs, gpu=self.using_gpu)
        # joints
        rhc_state.jnts_state.set(data=self.task.jnts_q(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="q", robot_idxs = env_indxs, gpu=self.using_gpu)
        rhc_state.jnts_state.set(data=self.task.jnts_v(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="v", robot_idxs = env_indxs, gpu=self.using_gpu) 
        rhc_state.jnts_state.set(data=self.task.jnts_eff(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="eff", robot_idxs = env_indxs, gpu=self.using_gpu) 
        # Updating contact state for selected contact links
        self._update_contact_state(robot_name=robot_name, env_indxs=env_indxs)

    def _reset_cluster(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str]=None):
        rob_names = robot_names
        if rob_names is None:
            rob_names = self.robot_names
        for i in range(len(rob_names)):
            robot_name = rob_names[i]
            control_cluster = self.cluster_servers[robot_name]
            control_cluster.reset_controllers(idxs=env_indxs)

class LRhcGzXBotSimEnv():
    def __init__(self):
        a=1

class LRhcMJXBot2SimEnv():
    def __init__(self):
        a=1 

class LRhcGzSimEnv():
    def __init__(self):
        a=1

class LRhcMJSimEnv():
    def __init__(self):
        a=1 