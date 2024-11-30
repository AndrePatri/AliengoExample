import torch 

from typing import List
from enum import Enum

from lrhc_control.utils.urdf_limits_parser import UrdfLimitsParser
from lrhc_control.utils.jnt_imp_cfg_parser import JntImpConfigParser

import time

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from abc import ABC, abstractmethod

class FirstOrderFilter:

    # a class implementing a simple first order filter

    def __init__(self,
            dt: float, 
            filter_BW: float = 0.1, 
            rows: int = 1, 
            cols: int = 1, 
            device: torch.device = torch.device("cpu"),
            dtype = torch.double):
        
        self._torch_dtype = dtype

        self._torch_device = device

        self._dt = dt

        self._rows = rows
        self._cols = cols

        self._filter_BW = filter_BW

        import math 
        self._gain = 2 * math.pi * self._filter_BW

        self.yk = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        self.ykm1 = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        
        self.refk = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        self.refkm1 = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        
        self._kh2 = self._gain * self._dt / 2.0
        self._coeff_ref = self._kh2 * 1/ (1 + self._kh2)
        self._coeff_km1 = (1 - self._kh2) / (1 + self._kh2)

    def update(self, 
               refk: torch.Tensor = None):
        
        if refk is not None:
            self.refk[:, :] = refk
        self.yk[:, :] = torch.add(torch.mul(self.ykm1, self._coeff_km1), 
                            torch.mul(torch.add(self.refk, self.refkm1), 
                                        self._coeff_ref))
        self.refkm1[:, :] = self.refk
        self.ykm1[:, :] = self.yk
    
    def reset(self,
            idxs: torch.Tensor = None):

        if idxs is not None:
            self.yk[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.ykm1[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refk[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refkm1[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
        else:
            self.yk[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.ykm1[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refk[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refkm1[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            
    def get(self):
        return self.yk

class JntSafety:

    def __init__(self, 
            urdf_parser: UrdfLimitsParser):

        self.limits_parser = urdf_parser
        self.limit_matrix = self.limits_parser.get_limits_matrix()

    def apply(self, 
        q_cmd=None, v_cmd=None, eff_cmd=None):

        if q_cmd is not None:
            self.saturate_tensor(q_cmd, position=True)
        if v_cmd is not None:
            self.saturate_tensor(v_cmd, velocity=True)
        if eff_cmd is not None:
            self.saturate_tensor(eff_cmd, effort=True)

    def has_nan(self, 
            tensor):
        return torch.any(torch.isnan(tensor))

    def saturate_tensor(self, tensor, position=False, velocity=False, effort=False):
        if self.has_nan(tensor):
            exception = f"Found nan elements in provided tensor!!"
            Journal.log(self.__class__.__name__,
                "saturate_tensor",
                exception,
                LogType.EXCEP,
                throw_when_excep = False)
            # Replace NaN values with infinity, so that we can clamp it
            tensor[:, :] = torch.nan_to_num(tensor, nan=torch.inf)
        if position:
            tensor[:, :] = torch.clamp(tensor[:, :], min=self.limit_matrix[:, 0], max=self.limit_matrix[:, 3])
        elif velocity:
            tensor[:, :] = torch.clamp(tensor[:, :], min=self.limit_matrix[:, 1], max=self.limit_matrix[:, 4])
        elif effort:
            tensor[:, :] = torch.clamp(tensor[:, :], min=self.limit_matrix[:, 2], max=self.limit_matrix[:, 5])               
            
class JntImpCntrlBase(ABC):

    class IndxState(Enum):

        NONE = -1 
        VALID = 1
        INVALID = 0

    def __init__(self, 
            num_envs: int,
            n_jnts: int,
            jnt_names: List[str],
            default_pgain = 300.0, 
            default_vgain = 30.0, 
            device: torch.device = torch.device("cpu"), 
            filter_BW = 50.0, # [Hz]
            filter_dt = None, # should correspond to the dt between samples
            dtype = torch.float32,
            enable_safety = True,
            urdf_path: str = None,
            config_path: str = None,
            enable_profiling: bool = False,
            debug_checks: bool = False,
            override_low_lev_controller: bool = False): # [s]
        
        self._override_low_lev_controller = override_low_lev_controller

        self._torch_dtype = dtype
        self._torch_device = device

        self.enable_profiling = enable_profiling
        self._debug_checks = debug_checks
        # debug data
        self.profiling_data = {}
        self.profiling_data["time_to_update_state"] = -1.0
        self.profiling_data["time_to_set_refs"] = -1.0
        self.profiling_data["time_to_apply_cmds"] = -1.0
        self.start_time = None
        if self.enable_profiling:
            self.start_time = time.perf_counter()

        self.enable_safety = enable_safety
        self.limiter = None
        self.robot_limits = None
        self.urdf_path = urdf_path
        self.config_path = config_path
    
        self.gains_initialized = False
        self.refs_initialized = False

        self._default_pgain = default_pgain
        self._default_vgain = default_vgain
        
        self._valid_signal_types = ["pos_ref", "vel_ref", "eff_ref", # references 
            "pos", "vel", "eff", # measurements
            "pgain", "vgain"]

        self.num_envs = num_envs
        self.n_jnts = n_jnts
        if not isinstance(jnt_names, List):
            exception = "jnt_names must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(jnt_names) == n_jnts:
            exception = f"jnt_names has len {len(jnt_names)}" + \
                f" which is not {n_jnts}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        self.jnts_names = jnt_names

        self._backend = "torch"
        if self.enable_safety:
            if self.urdf_path is None:
                exception = "If enable_safety is set to True, a urdf_path should be provided too!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            self.robot_limits = UrdfLimitsParser(urdf_path=self.urdf_path, 
                                        joint_names=self.jnts_names,
                                        backend=self._backend, 
                                        device=self._torch_device)
            self.limiter = JntSafety(urdf_parser=self.robot_limits)
        
        config_parser=JntImpConfigParser(config_file=self.config_path,
            joint_names=self.jnts_names,
            default_p_gain=default_pgain,
            default_v_gain=default_vgain,
            backend=self._backend,
            device=self._torch_device,
            search_for="motor_pd")
        startup_config_parser=JntImpConfigParser(config_file=self.config_path,
            joint_names=self.jnts_names,
            default_p_gain=default_pgain,
            default_v_gain=default_vgain,
            backend=self._backend,
            device=self._torch_device,
            search_for="startup_motor_pd")
        self._default_pd_gains=config_parser.get_pd_gains()
        self._default_p_gains=(self._default_pd_gains[:, 0:1].reshape(1,-1)).expand(self.num_envs, -1)
        self._default_d_gains=(self._default_pd_gains[:, 1:2].reshape(1,-1)).expand(self.num_envs, -1)
        self._startup_pd_gains=startup_config_parser.get_pd_gains()
        self._startup_p_gains=(self._startup_pd_gains[:, 0:1].reshape(1,-1)).expand(self.num_envs, -1)
        self._startup_d_gains=(self._startup_pd_gains[:, 1:2].reshape(1,-1)).expand(self.num_envs, -1)

        self._null_aux_tensor = torch.full((self.num_envs, self.n_jnts), 
            0.0, 
            device = self._torch_device, 
            dtype=self._torch_dtype)
        
        self._pos_err = None
        self._vel_err = None
        self._pos = None
        self._vel = None
        self._eff = None
        self._imp_eff = None

        self._filter_available = False
        self._filter_BW = None
        self._filter_dt = None
        if filter_dt is not None:
            self._filter_BW = filter_BW
            self._filter_dt = filter_dt
            self._pos_ref_filter = FirstOrderFilter(dt=self._filter_dt, 
                                    filter_BW=self._filter_BW, 
                                    rows=self.num_envs, 
                                    cols=self.n_jnts, 
                                    device=self._torch_device, 
                                    dtype=self._torch_dtype)
            self._vel_ref_filter = FirstOrderFilter(dt=self._filter_dt, 
                                    filter_BW=self._filter_BW, 
                                    rows=self.num_envs, 
                                    cols=self.n_jnts, 
                                    device=self._torch_device, 
                                    dtype=self._torch_dtype)
            self._eff_ref_filter = FirstOrderFilter(dt=self._filter_dt, 
                                    filter_BW=self._filter_BW, 
                                    rows=self.num_envs, 
                                    cols=self.n_jnts, 
                                    device=self._torch_device, 
                                    dtype=self._torch_dtype)
            self._filter_available = True

        else:
            warning = f"No filter dt provided -> reference filter will not be used!"
            Journal.log(self.__class__.__name__,
                "__init__",
                warning,
                LogType.WARN,
                throw_when_excep = True)
                            
        self.reset() # initialize data

    def default_p_gains(self):
        return self._default_p_gains
    
    def default_d_gains(self):
        return self._default_d_gains
    
    def startup_p_gains(self):
        return self._startup_p_gains
    
    def startup_d_gains(self):
        return self._startup_d_gains
    
    def update_state(self, 
        pos: torch.Tensor = None, 
        vel: torch.Tensor = None, 
        eff: torch.Tensor = None,
        robot_indxs: torch.Tensor = None, 
        jnt_indxs: torch.Tensor = None):

        if self.enable_profiling:
            self.start_time = time.perf_counter()
                                      
        selector = self._gen_selector(robot_indxs=robot_indxs, 
                        jnt_indxs=jnt_indxs) # only checks and throws
        # if debug_checks
        
        if pos is not None:
            self._validate_signal(signal = pos, 
                    selector = selector,
                    name="pos") # does nothing if not debug_checks
            self._pos[selector] = pos

        if vel is not None:
            self._validate_signal(signal = vel, 
                    selector = selector,
                    name="vel") 
            self._vel[selector] = vel

        if eff is not None:
            self._validate_signal(signal = eff, 
                    selector = selector,
                    name="eff") 
            self._eff[selector] = eff

        if self.enable_profiling:
            self.profiling_data["time_to_update_state"] = \
                time.perf_counter() - self.start_time
                
    def set_gains(self, 
                pos_gains: torch.Tensor = None, 
                vel_gains: torch.Tensor = None, 
                robot_indxs: torch.Tensor = None, 
                jnt_indxs: torch.Tensor = None):
        
        selector = self._gen_selector(robot_indxs=robot_indxs, 
                           jnt_indxs=jnt_indxs) # only checks and throws
        # if debug_checks
        
        if pos_gains is not None:
            self._validate_signal(signal = pos_gains, 
                selector = selector,
                name="pos_gains") 
            self._pos_gains[selector] = pos_gains
            if not self._override_low_lev_controller:                
                self._set_gains(kps = self._pos_gains)

        if vel_gains is not None:
            self._validate_signal(signal = vel_gains, 
                selector = selector,
                name="vel_gains") 
            self._vel_gains[selector] = vel_gains
            if not self._override_low_lev_controller:
                self._set_gains(kds = self._vel_gains)
    
    def set_refs(self, 
            eff_ref: torch.Tensor = None, 
            pos_ref: torch.Tensor = None, 
            vel_ref: torch.Tensor = None, 
            robot_indxs: torch.Tensor = None, 
            jnt_indxs: torch.Tensor = None):
        
        if self.enable_profiling:
            self.start_time = time.perf_counter()

        selector = self._gen_selector(robot_indxs=robot_indxs, 
                           jnt_indxs=jnt_indxs) # only checks and throws
        # if debug_checks
        
        if eff_ref is not None:
            self._validate_signal(signal = eff_ref, 
                selector = selector,
                name="eff_ref") 
            self._eff_ref[selector] = eff_ref

        if pos_ref is not None:
            self._validate_signal(signal = pos_ref, 
                selector = selector,
                name="pos_ref") 
            self._pos_ref[selector] = pos_ref
            
        if vel_ref is not None:
            self._validate_signal(signal = vel_ref, 
                    selector = selector,
                    name="vel_ref") 
            self._vel_ref[selector] = vel_ref

        if self.enable_profiling:
            self.profiling_data["time_to_set_refs"] = time.perf_counter() - self.start_time
                
    def apply_cmds(self, 
            filter = False):

        # initialize gains and refs if not done previously 
        
        if self.enable_profiling:
            self.start_time = time.perf_counter()

        if not self.gains_initialized:
            self._apply_init_gains()
        if not self.refs_initialized:
            self._apply_init_refs()
                
        if filter and self._filter_available:
            
            self._pos_ref_filter.update(self._pos_ref)
            self._vel_ref_filter.update(self._vel_ref)
            self._eff_ref_filter.update(self._eff_ref)

            # we first filter, then apply safety
            eff_ref_filt = self._eff_ref_filter.get()
            pos_ref_filt = self._pos_ref_filter.get()
            vel_ref_filt = self._vel_ref_filter.get()

            if self.limiter is not None:
                # saturating ref cmds
                self.limiter.apply(q_cmd=pos_ref_filt,
                                v_cmd=vel_ref_filt,
                                eff_cmd=eff_ref_filt)
                
            if not self._override_low_lev_controller:
                self._check_activation() # processes cmds in case of deactivations
                self._set_pos_ref(pos_ref_filt)
                self._set_vel_ref(vel_ref_filt)
                self._set_joint_efforts(eff_ref_filt)

            else:
                # impedance torque computed explicitly
                self._pos_err  = torch.sub(self._pos_ref_filter.get(), self._pos)
                self._vel_err = torch.sub(self._vel_ref_filter.get(), self._vel)
                self._imp_eff = torch.add(self._eff_ref_filter.get(), 
                                        torch.add(
                                            torch.mul(self._pos_gains, 
                                                    self._pos_err),
                                            torch.mul(self._vel_gains,
                                                    self._vel_err)))

                # torch.cuda.synchronize()
                # we also make the resulting imp eff safe
                if self.limiter is not None:
                    self.limiter.apply(eff_cmd=eff_ref_filt)
                self._check_activation() # processes cmds in case of deactivations
                # apply only effort (comprehensive of all imp. terms)
                self._set_joint_efforts(self._imp_eff)

        else:
            
            # we first apply safety to reference joint cmds
            if self.limiter is not None:
                self.limiter.apply(q_cmd=self._pos_ref,
                                v_cmd=self._vel_ref,
                                eff_cmd=self._eff_ref)
                    
            if not self._override_low_lev_controller:
                self._pos_err  = torch.sub(self._pos_ref, self._pos)
                self._vel_err = torch.sub(self._vel_ref, self._vel)
                
                self._check_activation() # processes cmds in case of deactivations

                self._set_pos_ref(self._pos_ref)
                self._set_vel_ref(self._vel_ref)
                self._set_joint_efforts(self._eff_ref)

            else:
                # impedance torque computed explicitly
                self._pos_err  = torch.sub(self._pos_ref, self._pos)
                self._vel_err = torch.sub(self._vel_ref, self._vel)
                self._imp_eff = torch.add(self._eff_ref, 
                                        torch.add(
                                            torch.mul(self._pos_gains, 
                                                    self._pos_err),
                                            torch.mul(self._vel_gains,
                                                    self._vel_err)))

                # torch.cuda.synchronize()

                # we also make the resulting imp eff safe
                if self.limiter is not None:
                    self.limiter.apply(eff_cmd=self._imp_eff)

                # apply only effort (comprehensive of all imp. terms)
                self._check_activation() # processes cmds in case of deactivations
                self._set_joint_efforts(self._imp_eff)
        
        if self.enable_profiling:
            self.profiling_data["time_to_apply_cmds"] = \
                time.perf_counter() - self.start_time 
    
    def get_jnt_names_matching(self, 
                        name_pattern: str):

        return [jnt for jnt in self.jnts_names if name_pattern in jnt]

    def get_jnt_idxs_matching(self, 
                        name_pattern: str):

        jnts_names = self.get_jnt_names_matching(name_pattern)
        jnt_idxs = [self.jnts_names.index(jnt) for jnt in jnts_names]
        if not len(jnt_idxs) == 0:
            return torch.tensor(jnt_idxs, 
                            dtype=torch.int64,
                            device=self._torch_device)
        else:
            return None
    
    def pos_gains(self):

        return self._pos_gains
    
    def vel_gains(self):

        return self._vel_gains
    
    def eff_ref(self):

        return self._eff_ref
    
    def pos_ref(self):

        return self._pos_ref

    def vel_ref(self):

        return self._vel_ref

    def pos_err(self):

        return self._pos_err

    def vel_err(self):

        return self._vel_err

    def pos(self):

        return self._pos
    
    def vel(self):

        return self._vel

    def eff(self):

        return self._eff

    def imp_eff(self):

        return self._imp_eff
    
    def reset(self,
            robot_indxs: torch.Tensor = None):
        
        self.gains_initialized = False
        self.refs_initialized = False
        
        self._all_dofs_idxs = torch.tensor([i for i in range(0, self.n_jnts)], 
                                        dtype=torch.int64,
                                        device=self._torch_device)
        self._all_robots_idxs = torch.tensor([i for i in range(0, self.num_envs)], 
                                        dtype=torch.int64,
                                        device=self._torch_device)
        
        if robot_indxs is None: # reset all data
            # we assume diagonal joint impedance gain matrices, so we can save on memory and only store the diagonal
            self._active = torch.full((self.num_envs, 1), 
                                    True, 
                                    device = self._torch_device, 
                                    dtype=torch.bool)
            
            self._pos_gains = torch.full((self.num_envs, self.n_jnts), 
                                        self._default_pgain, 
                                        device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel_gains = torch.full((self.num_envs, self.n_jnts), 
                                        self._default_vgain,
                                        device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._pos_gains[:, :] = self._default_p_gains
            self._vel_gains[:, :] = self._default_d_gains

            self._eff_ref = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._pos_ref = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel_ref = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)                
            self._pos_err = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel_err = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._pos = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._eff = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._imp_eff = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            
            if self._filter_available:
                self._pos_ref_filter.reset()
                self._vel_ref_filter.reset()
                self._eff_ref_filter.reset()
        
        else: # only reset some robots
            
            if self._debug_checks:
                self._validate_selectors(robot_indxs=robot_indxs) # throws if checks not satisfied
            n_envs = robot_indxs.shape[0]
            # we assume diagonal joint impedance gain matrices, so we can save on memory and only store the diagonal

            self._active[robot_indxs, :] = True # reactivate inactive controller
            self._pos_gains[robot_indxs, :] =  torch.full((n_envs, self.n_jnts), 
                                        self._default_pgain, 
                                        device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel_gains[robot_indxs, :] = torch.full((n_envs, self.n_jnts), 
                                        self._default_vgain,
                                        device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._pos_gains[robot_indxs, :] = (self._default_pd_gains[:, 0:1].reshape(1,-1)).expand(n_envs, -1)
            self._vel_gains[robot_indxs, :] = (self._default_pd_gains[:, 1:2].reshape(1,-1)).expand(n_envs, -1)

            self._eff_ref[robot_indxs, :] = 0
            self._pos_ref[robot_indxs, :] = 0
            self._vel_ref[robot_indxs, :] = 0
                                                                        
            self._pos_err[robot_indxs, :] = torch.zeros((n_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel_err[robot_indxs, :] = torch.zeros((n_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            
            self._pos[robot_indxs, :] = torch.zeros((n_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._vel[robot_indxs, :] = torch.zeros((n_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            self._eff[robot_indxs, :] = torch.zeros((n_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)
            
            self._imp_eff[robot_indxs, :] = torch.zeros((n_envs, self.n_jnts), device = self._torch_device, 
                                        dtype=self._torch_dtype)

            if self._filter_available:
                self._pos_ref_filter.reset(idxs = robot_indxs)
                self._vel_ref_filter.reset(idxs = robot_indxs)
                self._eff_ref_filter.reset(idxs = robot_indxs)

            self._apply_init_gains()
            self._apply_init_refs()
    
    def deactivate(self,
            robot_indxs: torch.Tensor = None):
        
        if robot_indxs is not None:
            self._active[robot_indxs, :] = False
        else:
            self._active[:, :] = False

    def _check_activation(self):
        # inactive controllers have their imp effort set to 0 
        inactive = ~self._active.flatten()
        if not self._override_low_lev_controller:
            inactive_idxs=torch.nonzero(inactive)
            if inactive_idxs.numel()>0:
                self.set_gains(pos_gains=self._null_aux_tensor[inactive_idxs.flatten(), :],
                        vel_gains=self._null_aux_tensor[inactive_idxs.flatten(), :],
                        robot_indxs=inactive_idxs.flatten())
        self._eff_ref[inactive, :] = 0.0
        self._imp_eff[inactive, :] = 0.0

    def _set_gains(self, 
        kps: torch.Tensor = None, 
        kds: torch.Tensor = None):
        pass # to be overridden
    
    def _set_pos_ref(self, pos: torch.Tensor):
        pass # to be overridden

    def _set_vel_ref(self, vel: torch.Tensor):
        pass # to be overridden

    def _set_joint_efforts(self, effort: torch.Tensor):
        pass # to be overridden

    def _apply_init_gains(self):
        if not self.gains_initialized:
            if not self._override_low_lev_controller:
                self._set_gains(kps=self._pos_gains, 
                    kds=self._vel_gains)
            else: # gains of low lev controller are set to zero
                no_gains = torch.zeros((self.num_envs, self.n_jnts), device = self._torch_device, 
                                    dtype=self._torch_dtype)        
                self._set_gains(kps=no_gains, 
                    kds=no_gains)
            self.gains_initialized = True

    def _apply_init_refs(self):
        if not self.refs_initialized: 
            if not self._override_low_lev_controller:
                self._set_joint_efforts(self._eff_ref)
                self._set_pos_ref(self._pos_ref)
                self._set_vel_ref(self._vel_ref)
            else:
                self._set_joint_efforts(self._eff_ref)
            self.refs_initialized = True
    
    def _validate_selectors(self, 
                robot_indxs: torch.Tensor = None, 
                jnt_indxs: torch.Tensor = None):

        if robot_indxs is not None:
            robot_indxs_shape = robot_indxs.shape
            if (not (len(robot_indxs_shape) == 1 and \
                robot_indxs.dtype == torch.int64 and \
                bool(torch.min(robot_indxs) >= 0) and \
                bool(torch.max(robot_indxs) < self.num_envs)) and \
                robot_indxs.device.type == self._torch_device.type): # sanity checks 
                
                error = "Mismatch in provided selector \n" + \
                    "robot_indxs_shape -> " + f"{len(robot_indxs_shape)}" + " VS" + " expected -> " + f"{1}" + "\n" + \
                    "robot_indxs.dtype -> " + f"{robot_indxs.dtype}" + " VS" + " expected -> " + f"{torch.int64}" + "\n" + \
                    "torch.min(robot_indxs) >= 0) -> " + f"{bool(torch.min(robot_indxs) >= 0)}" + " VS" + f" {True}" + "\n" + \
                    "torch.max(robot_indxs) < self.num_envs -> " + f"{torch.max(robot_indxs)}" + " VS" + f" {self.num_envs}\n" + \
                    "robot_indxs.device -> " + f"{robot_indxs.device.type}" + " VS" + " expected -> " + f"{self._torch_device.type}" + "\n"
                Journal.log(self.__class__.__name__,
                    "_validate_selectors",
                    error,
                    LogType.EXCEP,
                    throw_when_excep = True)

        if jnt_indxs is not None:
            jnt_indxs_shape = jnt_indxs.shape
            if (not (len(jnt_indxs_shape) == 1 and \
                jnt_indxs.dtype == torch.int64 and \
                bool(torch.min(jnt_indxs) >= 0) and \
                bool(torch.max(jnt_indxs) < self.n_jnts)) and \
                jnt_indxs.device.type == self._torch_device.type): # sanity checks 

                error = "Mismatch in provided selector \n" + \
                    "jnt_indxs_shape -> " + f"{len(jnt_indxs_shape)}" + " VS" + " expected -> " + f"{1}" + "\n" + \
                    "jnt_indxs.dtype -> " + f"{jnt_indxs.dtype}" + " VS" + " expected -> " + f"{torch.int64}" + "\n" + \
                    "torch.min(jnt_indxs) >= 0) -> " + f"{bool(torch.min(jnt_indxs) >= 0)}" + " VS" + f" {True}" + "\n" + \
                    "torch.max(jnt_indxs) < self.n_jnts -> " + f"{torch.max(jnt_indxs)}" + " VS" + f" {self.num_envs}" + \
                    "robot_indxs.device -> " + f"{jnt_indxs.device.type}" + " VS" + " expected -> " + f"{self._torch_device.type}" + "\n"
                Journal.log(self.__class__.__name__,
                    "_validate_selectors",
                    error,
                    LogType.EXCEP,
                    throw_when_excep = True)
    
    def _validate_signal(self, 
                    signal: torch.Tensor, 
                    selector: torch.Tensor = None,
                    name: str = "signal"):
        
        if self._debug_checks:
            signal_shape = signal.shape
            selector_shape = selector[0].shape
            if not (signal_shape[0] == selector_shape[0] and \
                signal_shape[1] == selector_shape[1] and \
                signal.device.type == torch.device(self._torch_device).type and \
                signal.dtype == self._torch_dtype):

                big_error = f"Mismatch in provided signal [{name}" + "] and/or selector \n" + \
                    "signal rows -> " + f"{signal_shape[0]}" + " VS" + " expected rows -> " + f"{selector_shape[0]}" + "\n" + \
                    "signal cols -> " + f"{signal_shape[1]}" + " VS" + " expected cols -> " + f"{selector_shape[1]}" + "\n" + \
                    "signal dtype -> " + f"{signal.dtype}" + " VS" + " expected -> " + f"{self._torch_dtype}" + "\n" + \
                    "signal device -> " + f"{signal.device.type}" + " VS" + " expected type -> " + f"{torch.device(self._torch_device).type}"
                Journal.log(self.__class__.__name__,
                    "_validate_signal",
                    big_error,
                    LogType.EXCEP,
                    throw_when_excep = True)
    
    def _gen_selector(self, 
                robot_indxs: torch.Tensor = None, 
                jnt_indxs: torch.Tensor = None):
        
        if self._debug_checks:
            self._validate_selectors(robot_indxs=robot_indxs, 
                            jnt_indxs=jnt_indxs) # throws if not valid     
        
        if robot_indxs is None:
            robot_indxs = self._all_robots_idxs
        if jnt_indxs is None:
            jnt_indxs = self._all_dofs_idxs

        return torch.meshgrid((robot_indxs, jnt_indxs), 
                            indexing="ij")

if __name__ == "__main__":

    import tempfile
    import os
    import argparse
    
    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--urdf_path', type=str, help='')
    args = parser.parse_args()

    yaml_content = """
motor_pd:
  j_arm*_1: [500, 10]
  j_arm*_2: [500, 10]
  j_arm*_3: [500, 10]
  j_arm*_4: [500, 10]
  j_arm*_5: [100, 5]
  j_arm*_6: [100, 5]
  j_arm*_7: [100, 5]
  hip_yaw_*: [3000, 30]
  hip_pitch_*: [3000, 30]
  knee_pitch_*: [3000, 30]
  ankle_pitch_*: [1000, 10]
  ankle_yaw_*: [300, 10]
  neck_pitch: [10, 1]
  neck_yaw: [10, 1]
  torso_yaw: [1000, 30]
  j_wheel_*: [0, 30]
    """
    # Create a temporary file to store the YAML content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        temp_file.write(yaml_content.encode('utf-8'))
        temp_file_path = temp_file.name

    urdf_path=args.urdf_path

    jnt_names = ["j_arm_1", "j_arm_7", "ankle_pitch_4", "j_wheel_1", "unknown"]
    jnt_imp_controller=JntImpCntrlBase(num_envs=2,
        n_jnts=len(jnt_names),
        jnt_names=jnt_names,
        default_pgain=356,
        default_vgain=13,
        device="cuda",
        dtype=torch.float32,
        enable_safety=True,
        urdf_path=urdf_path,
        config_path=temp_file_path,
        enable_profiling=True,
        debug_checks=True,
        override_low_lev_controller=True
        )
    print(jnt_names)
    print(jnt_imp_controller.pos_gains())
    print(jnt_imp_controller.vel_gains())