from control_cluster_bridge.controllers.rhc import RHController
# from perf_sleep.pyperfsleep import PerfSleep
# from control_cluster_bridge.utilities.cpu_utils.core_utils import get_memory_usage

from lrhc_control.controllers.rhc.horizon_based.horizon_imports import *

from lrhc_control.controllers.rhc.horizon_based.hybrid_quad_rhc_refs import HybridQuadRhcRefs
from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType

import numpy as np

import os
# import shutil

import time
from abc import ABC, abstractmethod

from typing import Dict, List
import re

class HybridQuadRhc(RHController):

    def __init__(self, 
            srdf_path: str,
            urdf_path: str,
            config_path: str,
            robot_name: str, # used for shared memory namespaces
            codegen_dir: str, 
            n_nodes:float = 25,
            injection_node:int = 10,
            dt: float = 0.02,
            max_solver_iter = 1, # defaults to rt-iteration
            open_loop: bool = True,
            close_loop_all: bool = False,
            dtype = np.float32,
            verbose = False, 
            debug = False,
            refs_in_hor_frame = True,
            timeout_ms: int = 60000,
            custom_opts: Dict = {}):

        self._refs_in_hor_frame = refs_in_hor_frame

        self._injection_node = injection_node

        self._open_loop = open_loop
        self._close_loop_all = close_loop_all

        self._codegen_dir = codegen_dir
        if not os.path.exists(self._codegen_dir):
            os.makedirs(self._codegen_dir)
        # else:
        #     # Directory already exists, delete it and recreate
        #     shutil.rmtree(self._codegen_dir)
        #     os.makedirs(self._codegen_dir)

        self.step_counter = 0
        self.sol_counter = 0
    
        self.max_solver_iter = max_solver_iter
        
        self._timer_start = time.perf_counter()
        self._prb_update_time = time.perf_counter()
        self._phase_shift_time = time.perf_counter()
        self._task_ref_update_time = time.perf_counter()
        self._rti_time = time.perf_counter()

        self.robot_name = robot_name
        
        self.config_path = config_path

        self.urdf_path = urdf_path
        # read urdf and srdf files
        with open(self.urdf_path, 'r') as file:
            self.urdf = file.read()
        self._base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        self._c_timelines = dict()
        self._f_reg_timelines = dict()
        
        self._custom_opts={"replace_continuous_joints": False,
            "use_force_feedback": False,
            "fixed_flights": True,
            "lin_a_feedback": False,
            "is_open_loop": self._open_loop, # fully open (just for db)
            "fully_closed": False, # closed loop with full feedback (just for db)
            "closed_partial": False, # closed loop with partial feedback
            "adaptive_is": True, # closed loop with adaptation
            "estimate_v_root": False, # when adaptive_is or closed_partial, estimate vbase
            "alpha_from_outside": False, # alpha set ext. from shared memory
            "alpha_half": 1.0, 
            "only_vel_wheels": True # whether wheels (if present) are just vel controlled
            }
        
        self._custom_opts.update(custom_opts)
        
        self._alpha_half=self._custom_opts["alpha_half"]

        if self._open_loop:
            self._custom_opts["fully_closed"]=False
            self._custom_opts["adaptive_is"]=False
            self._custom_opts["closed_partial"]=False
        else:
            self._custom_opts["is_open_loop"]=False
            if self._custom_opts["fully_closed"]:
                self._custom_opts["adaptive_is"]=False
                self._custom_opts["closed_partial"]=False
                self._custom_opts["lin_a_feedback"]=False
            if self._custom_opts["closed_partial"]:
                self._custom_opts["adaptive_is"]=False
                self._custom_opts["fully_closed"]=False
                self._custom_opts["lin_a_feedback"]=False
            if self._custom_opts["adaptive_is"]:
                self._custom_opts["closed_partial"]=False
                self._custom_opts["fully_closed"]=False

        super().__init__(srdf_path=srdf_path,
                        n_nodes=n_nodes,
                        dt=dt,
                        namespace=self.robot_name,
                        dtype=dtype,
                        verbose=verbose, 
                        debug=debug,
                        timeout_ms=timeout_ms)

        self._rhc_fpaths.append(self.config_path)

    def _config_override(self):
        pass

    def _post_problem_init(self):

        self.rhc_costs={}
        self.rhc_constr={}

        self._fail_idx_scale=0.0
        self._expl_idx_window_size=int(1*self._n_nodes) 
        self._explosion_idx_buffer=np.zeros((1,self._expl_idx_window_size))
        self._expl_idx_counter=0
        self._expl_idx_buffer_counter=0

        self._pred_node_idx=self._n_nodes-1

        self._nq=self.nq()
        self._nq_jnts=self._nq-7# assuming floating base
        self._nv=self.nv()
        self._nv_jnts=self._nv-6

        self._alphas_q_root=np.zeros((7, 1), dtype=self._dtype)
        self._alphas_q_jnts=np.zeros((self._nq_jnts, 1), dtype=self._dtype)
        self._alphas_twist_root=np.zeros((6, 1), dtype=self._dtype)
        self._alphas_v_jnts=np.zeros((self._nv_jnts, 1), dtype=self._dtype)
        self._alphas_a=np.zeros((self._nv, 1), dtype=self._dtype)
        self._alphas_q_root[:, :]=1.0 # default to all open
        self._alphas_q_jnts[:, :]=1.0 
        self._alphas_twist_root[:, :]=1.0 
        self._alphas_v_jnts[:, :]=1.0 
        self._alphas_a[:, :]=1.0

    def _init_problem(self,
            fixed_jnt_patterns: List[str] = None,
            wheels_patterns: List[str] = None,
            foot_linkname: str = None,
            flight_duration: int = 10,
            post_flight_stance: int = 3,
            step_height: float = 0.12,
            keep_yaw_vert: bool = False,
            yaw_vertical_weight: float = 2.0,
            vertical_landing: bool = False,
            vertical_land_weight: float = 1.0,
            phase_force_reg: float = 1e-2,
            vel_bounds_weight: float = 1.0):
        
        self._fixed_jnt_patterns=fixed_jnt_patterns

        self._config_override()
        
        Journal.log(self.__class__.__name__,
            "_init_problem",
            f" Will use horizon config file at {self.config_path}",
            LogType.INFO,
            throw_when_excep=True)
        
        self._vel_bounds_weight=vel_bounds_weight
        self._phase_force_reg=phase_force_reg
        self._yaw_vertical_weight=yaw_vertical_weight
        self._vertical_land_weight=vertical_land_weight
        # overrides parent
        self._prb = Problem(self._n_intervals, 
                        receding=True, 
                        casadi_type=cs.SX)
        self._prb.setDt(self._dt)

        if "replace_continuous_joints" in self._custom_opts:
            # continous joints are parametrized in So2
            if self._custom_opts["replace_continuous_joints"]:
                self.urdf = self.urdf.replace('continuous', 'revolute')
        else:
            self.urdf = self.urdf.replace('continuous', 'revolute')
        
        self._kin_dyn = casadi_kin_dyn.CasadiKinDyn(self.urdf) # used for getting joint names 
        self._assign_controller_side_jnt_names(jnt_names=self._get_robot_jnt_names())
        
        self._init_robot_homer()

        # handle fixed joints
        fixed_joint_map={}
        if self._fixed_jnt_patterns is not None:
            for jnt_name in self._get_robot_jnt_names():
                for fixed_jnt_pattern in self._fixed_jnt_patterns:
                    if fixed_jnt_pattern in jnt_name: 
                        fixed_joint_map.update({f"{jnt_name}":
                            self._homer.get_homing_val(jnt_name=jnt_name)})
                        break # do not search for other pattern matches
        
        if not len(fixed_joint_map)==0: # we need to recreate kin dyn and homers
            Journal.log(self.__class__.__name__,
                "_init_problem",
                f"Will fix following joints: \n{str(fixed_joint_map)}",
                LogType.INFO,
                throw_when_excep=True)
            # with the fixed joint map
            self._kin_dyn = casadi_kin_dyn.CasadiKinDyn(self.urdf,fixed_joints=fixed_joint_map)
            # assign again controlled joints names
            self._assign_controller_side_jnt_names(jnt_names=self._get_robot_jnt_names())
            # updated robot homer for controlled joints
            self._init_robot_homer()

        # handle continuous joints (need to change homing and retrieve
        # cont jnts indexes) and homing
        self._continuous_joints=self._get_continuous_jnt_names()
        # reduced
        self._continuous_joints_idxs=[]
        self._continuous_joints_idxs_cos=[]
        self._continuous_joints_idxs_sin=[]
        self._continuous_joints_idxs_red=[]
        self._rev_joints_idxs=[]
        self._rev_joints_idxs_red=[]
        # qfull
        self._continuous_joints_idxs_qfull=[]
        self._continuous_joints_idxs_cos_qfull=[]
        self._continuous_joints_idxs_sin_qfull=[]
        self._continuous_joints_idxs_red_qfull=[]
        self._rev_joints_idxs_qfull=[]
        self._rev_joints_idxs_red_qfull=[]
        jnt_homing=[""]*(len(self._homer.get_homing())+len(self._continuous_joints))
        jnt_names=self._get_robot_jnt_names()
        for i in range(len(jnt_names)):
            jnt=jnt_names[i]
            index=self._get_jnt_id(jnt)# accounting for floating joint
            homing_idx=index-7 # homing is only for actuated joints
            homing_value=self._homer.get_homing_val(jnt)
            if jnt in self._continuous_joints:
                jnt_homing[homing_idx]=np.cos(homing_value).item()
                jnt_homing[homing_idx+1]=np.sin(homing_value).item()
                # just actuated joints
                self._continuous_joints_idxs.append(homing_idx) # cos
                self._continuous_joints_idxs.append(homing_idx+1) # sin
                self._continuous_joints_idxs_cos.append(homing_idx)
                self._continuous_joints_idxs_sin.append(homing_idx+1)
                self._continuous_joints_idxs_red.append(i)
                # q full
                self._continuous_joints_idxs_qfull.append(index) # cos
                self._continuous_joints_idxs_qfull.append(index+1) # sin
                self._continuous_joints_idxs_cos_qfull.append(index)
                self._continuous_joints_idxs_sin_qfull.append(index+1)
                self._continuous_joints_idxs_red_qfull.append(i+7)
            else:
                jnt_homing[homing_idx]=homing_value
                # just actuated joints
                self._rev_joints_idxs.append(homing_idx) 
                self._rev_joints_idxs_red.append(i) 
                # q full
                self._rev_joints_idxs_qfull.append(index) 
                self._rev_joints_idxs_red_qfull.append(i+7) 

        self._jnts_q_reduced=None
        if not len(self._continuous_joints)==0: 
            cont_joints=", ".join(self._continuous_joints)

            Journal.log(self.__class__.__name__,
                "_init_problem",
                f"The following continuous joints were found: \n{cont_joints}",
                LogType.INFO,
                throw_when_excep=True)
            # preallocating data 
            self._jnts_q_reduced=np.zeros((1,self.nv()-6),dtype=self._dtype)
            self._jnts_q_expanded=np.zeros((self.nq()-7,1),dtype=self._dtype)
            self._full_q_reduced=np.zeros((7+len(jnt_names), self._n_nodes),dtype=self._dtype)
            self._jnts_q_delta_expanded=np.zeros((self.nq()-7,1),dtype=self._dtype)
        else:
            self._custom_opts["replace_continuous_joints"]=True
            Journal.log(self.__class__.__name__,
                "_init_problem",
                f"No continuous joints were found.",
                LogType.INFO,
                throw_when_excep=True)

        # retrieve wheels indexes (not considering continuous joints, 
        # ok just for v, eff, etc..)
        self._wheel_patterns=wheels_patterns
        self._wheels_idxs_v=self._get_wheels_jnt_v_idxs(wheel_patterns=self._wheel_patterns)
        self._f0 = [0, 0, self._kin_dyn.mass()/4*9.81]
        
        # we can create an init for the base
        init = self._base_init.tolist() + jnt_homing

        if foot_linkname is not None:
            FK = self._kin_dyn.fk(foot_linkname) # just to get robot reference height
            ground_level = FK(q=init)['ee_pos']
            self._base_init[2] = -ground_level[2]  # override init
        
        self._model = FullModelInverseDynamics(problem=self._prb,
            kd=self._kin_dyn,
            q_init=self._homer.get_homing_map(),
            base_init=self._base_init)

        self._ti = TaskInterface(prb=self._prb, 
                            model=self._model, 
                            max_solver_iter=self.max_solver_iter,
                            debug = self._debug,
                            verbose = self._verbose, 
                            codegen_workdir = self._codegen_dir)
        self._ti.setTaskFromYaml(self.config_path)
        
        # setting initial base pos ref if exists
        base_pos = self._ti.getTask('base_height')
        if base_pos is not None:
            base_pos.setRef(np.atleast_2d(self._base_init).T)

        self._pm = pymanager.PhaseManager(self._n_nodes, debug=False) # intervals or nodes?????

        self._gm = GaitManager(self._ti, 
            self._pm, 
            self._injection_node,
            keep_yaw_vert=keep_yaw_vert,
            yaw_vertical_weight=self._yaw_vertical_weight,
            vertical_landing=vertical_landing,
            landing_vert_weight=self._vertical_land_weight,
            phase_force_reg=self._phase_force_reg,
            custom_opts=self._custom_opts,
            flight_duration=flight_duration,
            post_flight_stance=post_flight_stance,
            step_height=step_height,
            dh=0.0)
                
        self._ti.model.q.setBounds(self._ti.model.q0, self._ti.model.q0, nodes=0)
        self._ti.model.v.setBounds(self._ti.model.v0, self._ti.model.v0, nodes=0)
        self._ti.model.q.setInitialGuess(self._ti.model.q0)
        self._ti.model.v.setInitialGuess(self._ti.model.v0)
        for _, cforces in self._ti.model.cmap.items():
            n_contact_f=len(cforces)
            for c in cforces:
                c.setInitialGuess(np.array(self._f0)/n_contact_f)        

        vel_lims = self._model.kd.velocityLimits()
        import horizon.utils as utils
        self._prb.createResidual('vel_lb_barrier', self._vel_bounds_weight*utils.utils.barrier(vel_lims[7:] - self._model.v[7:]))
        self._prb.createResidual('vel_ub_barrier', self._vel_bounds_weight*utils.utils.barrier1(-1 * vel_lims[7:] - self._model.v[7:]))

        self._meas_lin_a_par=None
        # if self._custom_opts["lin_a_feedback"]:
        #     # acceleration feedback on first node
        #     self._meas_lin_a_par=self._prb.createParameter(name="lin_a_feedback",
        #         dim=3, nodes=0)   
        #     base_lin_a_prb=self._prb.getInput().getVars()[0:3] 
        #     self._prb.createConstraint('lin_acceleration_feedback', base_lin_a_prb - self._meas_lin_a_par, 
        #             nodes=[0])

        # if not self._open_loop:
        #     # we create a residual cost to be used as an attractor to the measured state on the first node
        #     # hard constraints injecting meas. states are pure EVIL!
        #     prb_state=self._prb.getState()
        #     full_state=prb_state.getVars()
        #     state_dim=prb_state.getBounds()[0].shape[0]
        #     meas_state=self._prb.createParameter(name="measured_state",
        #         dim=state_dim, nodes=0)     
        #     self._prb.createResidual('meas_state_attractor', meas_state_attractor_weight * (full_state - meas_state), 
        #                 nodes=[0])

        self._ti.finalize()
        self._ti.bootstrap()

        self._ti.init_inv_dyn_for_res() # we initialize some objects for sol. postprocessing purposes
        self._ti.load_initial_guess()

        self.n_dofs = self._get_ndofs() # after loading the URDF and creating the controller we
        # know n_dofs -> we assign it (by default = None)

        self.n_contacts = len(self._model.cmap.keys())
        
        # remove variables bounds (before they were necessary to have a good
        # quality bootstrap solution)
        self._q_inf=np.zeros((self.nq(), 1))
        self._q_inf[:, :]=np.inf
        self._v_inf=np.zeros((self.nv(), 1))
        self._v_inf[:, :]=np.inf
        self._ti.model.q.setBounds(-self._q_inf, self._q_inf, nodes=0)
        self._ti.model.v.setBounds(-self._v_inf, self._v_inf, nodes=0)

        # self.horizon_anal = analyzer.ProblemAnalyzer(self._prb)

    def get_file_paths(self):
        # can be overriden by child
        paths = super().get_file_paths()
        return paths
    
    def _get_quat_remap(self):
        # overrides parent
        return [1, 2, 3, 0] # mapping from robot quat. to Horizon's quaternion convention
    
    def _zmp(self, model):

        num = cs.SX([0, 0])
        den = cs.SX([0])
        pos_contact = dict()
        force_val = dict()

        q = cs.SX.sym('q', model.nq)
        v = cs.SX.sym('v', model.nv)
        a = cs.SX.sym('a', model.nv)

        com = model.kd.centerOfMass()(q=q, v=v, a=a)['com']

        n = cs.SX([0, 0, 1])
        for c in model.fmap.keys():
            pos_contact[c] = model.kd.fk(c)(q=q)['ee_pos']
            force_val[c] = cs.SX.sym('force_val', 3)
            num += (pos_contact[c][0:2] - com[0:2]) * cs.dot(force_val[c], n)
            den += cs.dot(force_val[c], n)

        zmp = com[0:2] + (num / den)
        input_list = []
        input_list.append(q)
        input_list.append(v)
        input_list.append(a)

        for elem in force_val.values():
            input_list.append(elem)

        f = cs.Function('zmp', input_list, [zmp])
        return f
    
    def _add_zmp(self):

        input_zmp = []

        input_zmp.append(self._model.q)
        input_zmp.append(self._model.v)
        input_zmp.append(self._model.a)

        for f_var in self._model.fmap.values():
            input_zmp.append(f_var)

        c_mean = cs.SX([0, 0, 0])
        for c_name, f_var in self._model.fmap.items():
            fk_c_pos = self._kin_dyn.fk(c_name)(q=self._model.q)['ee_pos']
            c_mean += fk_c_pos

        c_mean /= len(self._model.cmap.keys())

        zmp_nominal_weight = 10.
        zmp_fun = self._zmp(self._model)(*input_zmp)

        if 'wheel_joint_1' in self._model.kd.joint_names():
            zmp_residual = self._prb.createIntermediateResidual('zmp',  zmp_nominal_weight * (zmp_fun[0:2] - c_mean[0:2]))

    def _quaternion_multiply(self, 
                    q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])
    
    def _get_continuous_jnt_names(self):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(self.urdf)
        continuous_joints = []
        for joint in root.findall('joint'):
            joint_type = joint.get('type')
            if joint_type == 'continuous':
                joint_name = joint.get('name')
                continuous_joints.append(joint_name)
        return continuous_joints
    
    def _get_wheels_jnt_v_idxs(self, wheel_patterns: List[str]):
        jnt_names=self._get_robot_jnt_names()
        wheels_idxs=[]
        for i in range(len(jnt_names)):
            jnt_name=jnt_names[i]
            for wheel_pattern in wheel_patterns:
                if wheel_pattern in jnt_name:
                    wheels_idxs.append(i)
                    break
        return wheels_idxs
    
    def _get_jnt_id(self, jnt_name):
        return self._kin_dyn.joint_iq(jnt_name)
    
    def _init_rhc_task_cmds(self):
        
        rhc_refs = HybridQuadRhcRefs(gait_manager=self._gm,
            robot_index_shm=self.controller_index,
            robot_index_view=0, # when using optimize_mem the view if always of shape 1x...
            namespace=self.namespace,
            safe=False, 
            verbose=self._verbose,
            vlevel=VLevel.V2,
            use_force_feedback=self._custom_opts["use_force_feedback"],
            use_fixed_flights=self._custom_opts["fixed_flights"],
            optimize_mem=True)
        
        rhc_refs.run()

        rhc_refs.rob_refs.set_jnts_remapping(jnts_remapping=self._to_controller)
        rhc_refs.rob_refs.set_q_remapping(q_remapping=self._get_quat_remap())
        
        rhc_refs.set_default_refs(p_ref=np.atleast_2d(self._base_init)[:, 0:3],
            q_ref=np.atleast_2d(self._base_init)[:, 3:7])
        
        return rhc_refs
    
    def get_vertex_fnames_from_ti(self):
        tasks=self._ti.task_list
        contact_f_names=[]
        for task in tasks:
            if isinstance(task, ContactTask):
                interaction_task=task.dynamics_tasks[0]
                contact_f_names.append(interaction_task.vertex_frames[0])
        return contact_f_names
        
    def _get_contact_names(self):
        # should get contact names from vertex frames
        # list(self._ti.model.cmap.keys())
        return self.get_vertex_fnames_from_ti()
    
    def _get_robot_jnt_names(self):

        joints_names = self._kin_dyn.joint_names()
        to_be_removed = ["universe", 
                        "reference", 
                        "world", 
                        "floating", 
                        "floating_base"]
        for name in to_be_removed:
            if name in joints_names:
                joints_names.remove(name)

        return joints_names
    
    def _get_ndofs(self):
        return len(self._model.joint_names)
    
    def nq(self):
        return self._kin_dyn.nq()
    
    def nv(self):
        return self._kin_dyn.nv()
    
    def _get_robot_mass(self):

        return self._kin_dyn.mass()

    def _get_root_full_q_from_sol(self, node_idx=1):

        root_q_full=self._ti.solution['q'][0:7, node_idx].reshape(1, 7).astype(self._dtype)

        np.nan_to_num(root_q_full, nan=1e3, posinf=1e3, neginf=-1e3, copy=False)
        np.clip(a=root_q_full, a_min=-1e3, a_max=1e3, out=root_q_full)

        return root_q_full
    
    def _get_full_q_from_sol(self, node_idx=1):

        return self._ti.solution['q'][:, node_idx].reshape(1, -1).astype(self._dtype)
    
    def _get_root_twist_from_sol(self, node_idx=1):
        # provided in world frame
        twist_base_local=self._get_v_from_sol()[0:6, node_idx].reshape(1, 6)
        # if world_aligned:
        #     q_root_rhc = self._get_root_full_q_from_sol(node_idx=node_idx)[:, 0:4]
        #     r_base_rhc=Rotation.from_quat(q_root_rhc.flatten()).as_matrix()
        #     twist_base_local[:, 0:3] = r_base_rhc @ twist_base_local[0, 0:3]
        #     twist_base_local[:, 3:6] = r_base_rhc @ twist_base_local[0, 3:6]
        return twist_base_local

    def _get_root_a_from_sol(self, node_idx=0):
        # provided in world frame
        a_base_local=self._get_a_from_sol()[0:6, node_idx].reshape(1, 6)
        # if world_aligned:
        #     q_root_rhc = self._get_root_full_q_from_sol(node_idx=node_idx)[:, 0:4]
        #     r_base_rhc=Rotation.from_quat(q_root_rhc.flatten()).as_matrix()
        #     a_base_local[:, 0:3] = r_base_rhc @ a_base_local[0, 0:3]
        #     a_base_local[:, 3:6] = r_base_rhc @ v[0, 3:6]
        return a_base_local
    
    def _get_jnt_q_from_sol(self, node_idx=0, 
            reduce: bool = True,
            clamp: bool = True):
        
        full_jnts_q=self._ti.solution['q'][7:, node_idx:node_idx+1].reshape(1,-1).astype(self._dtype)
        
        np.nan_to_num(full_jnts_q, nan=1e3, posinf=1e3, neginf=-1e3, copy=False) # in place
        np.clip(a=full_jnts_q, a_max=1e3, a_min=-1e3, out=full_jnts_q) # in place

        if self._custom_opts["replace_continuous_joints"] or (not reduce):
            if clamp:
                return np.fmod(full_jnts_q, 2*np.pi)
            else:
                return full_jnts_q
        else:
            cos_sin=full_jnts_q[:,self._continuous_joints_idxs].reshape(-1,2)
            # copy rev joint vals
            self._jnts_q_reduced[:, self._rev_joints_idxs_red]=np.fmod(full_jnts_q[:, self._rev_joints_idxs], 2*np.pi).reshape(1, -1)
            # and continuous
            self._jnts_q_reduced[:, self._continuous_joints_idxs_red]=np.arctan2(cos_sin[:, 1], cos_sin[:, 0]).reshape(1,-1)
            return self._jnts_q_reduced
        
    def _get_jnt_v_from_sol(self, node_idx=1):
        
        jnt_v_sol=self._get_v_from_sol()[6:, node_idx].reshape(1,  
                    self._nv_jnts)
        np.nan_to_num(jnt_v_sol, nan=1e5, posinf=1e5, neginf=-1e5, copy=False) # in place
        # np.clip(a=jnt_v_sol, a_max=1e5, a_min=-1e5, out=jnt_v_sol) # in place

        return jnt_v_sol

    def _get_jnt_a_from_sol(self, node_idx=1):

        return self._get_a_from_sol()[6:, node_idx].reshape(1,
                    self._nv_jnts)

    def _get_jnt_eff_from_sol(self, node_idx=0):
        
        efforts_on_node = self._ti.eval_efforts_on_node(node_idx=node_idx)
        
        # if self._custom_opts["only_vel_wheels"]:

        jnt_efforts=efforts_on_node[6:, 0]
        
        if self._custom_opts["only_vel_wheels"] and self._wheels_idxs_v:
            jnt_efforts[self._wheels_idxs_v]=0.0

        return jnt_efforts.reshape(1,  
                self._nv_jnts).astype(self._dtype)
    
    def _get_rhc_cost(self):

        return self._ti.solution["opt_cost"]
    
    def _get_rhc_constr_viol(self):
        
        return self._ti.solution["residual_norm"]
    
    def _get_rhc_nodes_cost(self):

        cost = self._ti.solver_rti.getCostValOnNodes()
        return cost.reshape((1, -1))
    
    def _get_rhc_nodes_constr_viol(self):
        
        constr_viol = self._ti.solver_rti.getConstrValOnNodes()
        return constr_viol.reshape((1, -1))
    
    def _get_rhc_niter_to_sol(self):

        return self._ti.solution["n_iter2sol"]
    
    def _set_ig(self):

        shift_num = -1 # shift data by one node

        x_opt = self._ti.solution['x_opt']
        u_opt = self._ti.solution['u_opt']

        # building ig for state
        xig = np.roll(x_opt, 
                shift_num, axis=1) # rolling state sol.
        for i in range(abs(shift_num)):
            # state on last node is copied to the elements
            # which are "lost" during the shift operation
            xig[:, -1 - i] = x_opt[:, -1]
        # building ig for inputs
        uig = np.roll(u_opt, 
                shift_num, axis=1) # rolling state sol.
        for i in range(abs(shift_num)):
            # state on last node is copied to the elements
            # which are "lost" during the shift operation
            uig[:, -1 - i] = u_opt[:, -1]

        # assigning ig
        self._prb.getState().setInitialGuess(xig)
        self._prb.getInput().setInitialGuess(uig)

        return xig, uig
    
    def _update_open_loop(self):

        xig, _ = self._set_ig()

        q_state, v_state, a_state=self._set_is_open()

        # robot_state=xig[:, 0]
        # # open loop update:
        # self._prb.setInitialState(x0=robot_state) # (xig has been shifted, so node 0
        # # is node 1 in the last opt solution)

        return q_state, v_state, a_state
    
    def _update_closed_loop(self):
        
        # set initial guess for controller
        xig, _ = self._set_ig()
        # set initial state
        q_state=None
        v_state=None
        a_state=None
        if self._custom_opts["adaptive_is"]:
            # adaptive closed loop
            q_state, v_state, a_state=self._set_is_adaptive()
        elif self._custom_opts["fully_closed"]:
            q_state, v_state, a_state=self._set_is_full()
        elif self._custom_opts["closed_partial"]:
            q_state, v_state, a_state=self._set_is_partial()
        else:
            Journal.log(self.__class__.__name__,
                    "_update_closed_loop",
                    "Neither adaptive_is, fully_closed, or closed_partial.",
                    LogType.EXCEP,
                    throw_when_excep = False)
            q_state, v_state, a_state=self._set_is()

        return q_state, v_state, a_state
    
    def _set_is_open(self):
        
        # overriding states with rhc data
        q_full_root=self._get_root_full_q_from_sol(node_idx=1).reshape(-1, 1)
        q_jnts=self._get_jnt_q_from_sol(node_idx=1, reduce=False).reshape(-1, 1)
        
        twist_root=self._get_root_twist_from_sol(node_idx=1).reshape(-1, 1)
        v_jnts=self._get_jnt_v_from_sol(node_idx=1).reshape(-1, 1)

        # rhc variables to be set
        q=self._prb.getVariables("q") # .setBounds()
        root_q_full_rhc=q[0:7] # root full q
        jnts_q_rhc=q[7:] # jnts q
        vel=self._prb.getVariables("v")
        root_twist_rhc=vel[0:6] # lin v.
        jnts_v_rhc=vel[6:] # jnts v

        self.rhc_refs.set_alpha(alpha=1.0) # fully open

        # close state on known quantities
        root_q_full_rhc.setBounds(lb=q_full_root,
            ub=q_full_root, nodes=0)
        jnts_q_rhc.setBounds(lb=q_jnts, 
            ub=q_jnts, nodes=0)
        root_twist_rhc.setBounds(lb=twist_root, 
            ub=twist_root, nodes=0)
        jnts_v_rhc.setBounds(lb=v_jnts, 
            ub=v_jnts, nodes=0)
        
        # return state used for feedback
        q_state=np.concatenate((q_full_root, q_jnts),
                axis=0)
        v_state=np.concatenate((twist_root, v_jnts),
                axis=0)
        
        return (q_state, v_state, None)
    
    def _set_is_full(self):
        
        # measurements
        q_full_root = self.robot_state.root_state.get(data_type="q_full", robot_idxs=self.controller_index).reshape(-1, 1)
        v_root = self.robot_state.root_state.get(data_type="v", robot_idxs=self.controller_index).reshape(-1, 1)
        omega = self.robot_state.root_state.get(data_type="omega", robot_idxs=self.controller_index).reshape(-1, 1)
        a_root = self.robot_state.root_state.get(data_type="a_full", robot_idxs=self.controller_index).reshape(-1, 1)
        
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
        v_jnts = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index).reshape(-1, 1)
        a_jnts = self.robot_state.jnts_state.get(data_type="a", robot_idxs=self.controller_index).reshape(-1, 1)

        if (not len(self._continuous_joints)==0): # we need do expand some meas. rev jnts to So2
            self._jnts_q_expanded[self._rev_joints_idxs, :]=q_jnts[self._rev_joints_idxs_red ,:]
            self._jnts_q_expanded[self._continuous_joints_idxs_cos, :]=np.cos(q_jnts[self._continuous_joints_idxs_red, :]) # cos
            self._jnts_q_expanded[self._continuous_joints_idxs_sin, :]=np.sin(q_jnts[self._continuous_joints_idxs_red, :]) # sin
            q_jnts=self._jnts_q_expanded.reshape(-1,1)
            
        # rhc variables to be set
        q=self._prb.getVariables("q") # .setBounds()
        root_full_q_rhc=q[0:7] # root p
        jnts_q_rhc=q[7:] # jnts q
        vel=self._prb.getVariables("v")
        root_v_rhc=vel[0:3] # lin v.
        root_omega_rhc=vel[3:6] # omega
        jnts_v_rhc=vel[6:] # jnts v
        acc=self._prb.getVariables("a")
        lin_a_prb=acc[0:3] # lin acc
        
        self.rhc_refs.set_alpha(alpha=0.0) # non-adaptive

        root_full_q_rhc.setBounds(lb=q_full_root,
            ub=q_full_root, nodes=0)
        jnts_q_rhc.setBounds(lb=q_jnts, 
            ub=q_jnts, nodes=0)
        root_v_rhc.setBounds(lb=v_root, 
            ub=v_root, nodes=0) # leaving lin v of the base free (estimated from constraints)
        root_omega_rhc.setBounds(lb=omega, 
            ub=omega, nodes=0)
        jnts_v_rhc.setBounds(lb=v_jnts, 
            ub=v_jnts, nodes=0)
        if self._custom_opts["lin_a_feedback"]:
            # write base lin 13793197 from meas
            lin_a_prb.setBounds(lb=a_root[0:3, :], 
                ub=a_root[0:3, :], 
                nodes=0)

        # return state used for feedback
        q_state=np.concatenate((q_full_root, q_jnts),
                axis=0)
        v_state=np.concatenate((v_root, omega, v_jnts),
                axis=0)
        a_state=np.concatenate((a_root, a_jnts),
                axis=0)
        
        return (q_state, v_state, a_state)
    
    def _set_is_partial(self):
        
        # measurements
        p_root = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index_np).reshape(-1, 1)
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index_np).reshape(-1, 1)
        v_root = self.robot_state.root_state.get(data_type="v", robot_idxs=self.controller_index_np).reshape(-1, 1)
        omega = self.robot_state.root_state.get(data_type="omega", robot_idxs=self.controller_index_np).reshape(-1, 1)
        a_root = self.robot_state.root_state.get(data_type="a_full", robot_idxs=self.controller_index_np).reshape(-1, 1)
        
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index_np).reshape(-1, 1)
        v_jnts = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index_np).reshape(-1, 1)
        a_jnts = self.robot_state.jnts_state.get(data_type="a", robot_idxs=self.controller_index_np).reshape(-1, 1)

        if (not len(self._continuous_joints)==0): # we need do expand some meas. rev jnts to So2
            self._jnts_q_expanded[self._rev_joints_idxs, :]=q_jnts[self._rev_joints_idxs_red ,:]
            self._jnts_q_expanded[self._continuous_joints_idxs_cos, :]=np.cos(q_jnts[self._continuous_joints_idxs_red, :]) # cos
            self._jnts_q_expanded[self._continuous_joints_idxs_sin, :]=np.sin(q_jnts[self._continuous_joints_idxs_red, :]) # sin
            q_jnts=self._jnts_q_expanded.reshape(-1,1)

        # overriding states with rhc data (-> all overridden state are open looop)
        root_q_full_from_rhc=self._get_root_full_q_from_sol(node_idx=1).reshape(-1, 1)
        root_p_from_rhc=root_q_full_from_rhc[0:3, :]
        p_root[:, :]=root_p_from_rhc # position is always open loop
        if not self._custom_opts["estimate_v_root"]:
            v_root[:, :]=self._get_root_twist_from_sol(node_idx=1)[:, 0:3].reshape(-1, 1)
            # override v jnts with the ones from controller
            v_jnts[:, :]=self._get_jnt_v_from_sol(node_idx=1).reshape(-1, 1)
        # root_twist_from_rhc=self._get_root_twist_from_sol(node_idx=1)
        # root_v_from_rhc=root_twist_from_rhc[:, 0:3].reshape(-1, 1)
        # root_omega_from_rhc=root_twist_from_rhc[:, 3:6].reshape(-1, 1)
        # jnt_q_from_rhc=self._get_jnt_q_from_sol(node_idx=1,reduce=False,clamp=False).reshape(-1, 1)
        # jnt_v_from_rhc=self._get_jnt_v_from_sol(node_idx=1).reshape(-1, 1)
            
        # rhc variables to be set
        q=self._prb.getVariables("q") # .setBounds()
        root_p_rhc=q[0:3] # root p
        root_q_rhc=q[3:7] # root orientation
        jnts_q_rhc=q[7:] # jnts q
        vel=self._prb.getVariables("v")
        root_v_rhc=vel[0:3] # lin v.
        root_omega_rhc=vel[3:6] # omega
        jnts_v_rhc=vel[6:] # jnts v
        acc=self._prb.getVariables("a")
        lin_a_prb=acc[0:3] # lin acc
        
        self.rhc_refs.set_alpha(alpha=0.0) # non-adaptive

        root_p_rhc.setBounds(lb=p_root,
            ub=p_root, nodes=0)
        root_q_rhc.setBounds(lb=q_root, 
            ub=q_root, nodes=0)
        jnts_q_rhc.setBounds(lb=q_jnts, 
            ub=q_jnts, nodes=0)
        if self._custom_opts["estimate_v_root"]:
            root_v_rhc.setBounds(lb=-self._v_inf[0:3], 
                ub=self._v_inf[0:3], nodes=0) # leaving lin v of the base free (estimated from constraints)
        else: # get it from controller 
            root_v_rhc.setBounds(lb=v_root, 
                ub=v_root, nodes=0)
        root_omega_rhc.setBounds(lb=omega, 
            ub=omega, nodes=0)
        jnts_v_rhc.setBounds(lb=v_jnts, 
            ub=v_jnts, nodes=0)
        if self._custom_opts["lin_a_feedback"]:
            # write base lin 13793197 from meas
            lin_a_prb.setBounds(lb=a_root[0:3, :], 
                ub=a_root[0:3, :], 
                nodes=0)

        # return state used for feedback
        q_state=np.concatenate((p_root, q_root, q_jnts),
                axis=0)
        v_state=np.concatenate((v_root, omega, v_jnts),
                axis=0)
        a_state=np.concatenate((a_root, a_jnts),
                axis=0)
        
        return (q_state, v_state, a_state)
    
    def _set_is_adaptive(self):
        
        # measurements
        p_root = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index_np).reshape(-1, 1)
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index_np).reshape(-1, 1)
        v_root = self.robot_state.root_state.get(data_type="v", robot_idxs=self.controller_index_np).reshape(-1, 1)
        omega = self.robot_state.root_state.get(data_type="omega", robot_idxs=self.controller_index_np).reshape(-1, 1)
        a_root = self.robot_state.root_state.get(data_type="a_full", robot_idxs=self.controller_index_np).reshape(-1, 1)
        
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index_np).reshape(-1, 1)
        v_jnts = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index_np).reshape(-1, 1)
        a_jnts = self.robot_state.jnts_state.get(data_type="a", robot_idxs=self.controller_index_np).reshape(-1, 1)
            
        # rhc variables to be set
        q=self._prb.getVariables("q") # .setBounds()
        root_p_rhc=q[0:3] # root p
        root_q_rhc=q[3:7] # root orientation
        jnts_q_rhc=q[7:] # jnts q
        vel=self._prb.getVariables("v")
        root_v_rhc=vel[0:3] # lin v.
        root_omega_rhc=vel[3:6] # omega
        jnts_v_rhc=vel[6:] # jnts v
        acc=self._prb.getVariables("a")
        lin_a_prb=acc[0:3] # lin acc
        
        # getting prediction defects
        root_q_delta=self.rhc_pred_delta.root_state.get(data_type="q", robot_idxs=self.controller_index_np).reshape(-1, 1)
        jnt_q_delta=self.rhc_pred_delta.jnts_state.get(data_type="q", robot_idxs=self.controller_index_np).reshape(-1, 1)
        jnt_v_delta=self.rhc_pred_delta.jnts_state.get(data_type="v", robot_idxs=self.controller_index_np).reshape(-1, 1)
        v_root_delta = self.rhc_pred_delta.root_state.get(data_type="v", robot_idxs=self.controller_index_np).reshape(-1, 1)
        omega_root_delta = self.rhc_pred_delta.root_state.get(data_type="omega", robot_idxs=self.controller_index_np).reshape(-1, 1)
        a_root_delta = self.rhc_pred_delta.root_state.get(data_type="a_full", robot_idxs=self.controller_index_np).reshape(-1, 1)

        # close state on known quantities, estimate some (e.g. lin vel) and
        # open loop if thing start to explode
        alpha_now=1.0
        delta=0.0 
        if self._custom_opts["alpha_from_outside"]:
            alpha_now=self.rhc_refs.get_alpha()
        else: # "autotuned" alpha
            if self._custom_opts["estimate_v_root"]: # we copmute delta based on jnt v (since we use meas.)
                delta=np.max(np.abs(jnt_v_delta))
            else:
                delta=np.max(np.abs(omega_root_delta))
            # fail_idx=self._get_failure_index()
            # fail_idx=self._get_explosion_idx()/self._fail_idx_thresh
            alpha_now=(np.tanh(2*self._alpha_half*(delta-self._alpha_half))+1)/2.0

        bound_relaxation=-np.log(1-alpha_now+1e-16) # [~0, ~inf] if alpha_now [0, 1]
        self.rhc_refs.set_alpha(alpha=alpha_now) # also writes on shared mem for db
        self.rhc_refs.set_bound_relax(bound_relax=bound_relaxation) # also writes on shared mem for db

        self._alphas_q_root[:]=alpha_now # for now single alpha for everything
        self._alphas_q_jnts[:]=alpha_now
        self._alphas_twist_root[:]=alpha_now
        self._alphas_v_jnts[:]=alpha_now
        self._alphas_a[:]=alpha_now
        if not self._custom_opts["estimate_v_root"]:
            self._alphas_twist_root[0:3]=1.0 # open
            self._alphas_v_jnts[:]=1.0 # open

        # position is always open loop
        root_q_full_from_rhc=self._get_root_full_q_from_sol(node_idx=1).reshape(-1, 1)
        root_p_from_rhc=root_q_full_from_rhc[0:3, :]
        p_root[:, :]=root_p_from_rhc 
    
        # expaning meas q if continuous joints
        if (not len(self._continuous_joints)==0): # we need do expand some meas. rev jnts to So2
            self._jnts_q_expanded[self._rev_joints_idxs, :]=q_jnts[self._rev_joints_idxs_red ,:]
            self._jnts_q_expanded[self._continuous_joints_idxs_cos, :]=np.cos(q_jnts[self._continuous_joints_idxs_red, :]) # cos
            self._jnts_q_expanded[self._continuous_joints_idxs_sin, :]=np.sin(q_jnts[self._continuous_joints_idxs_red, :]) # sin
            
            # continous joints position is always open loop, but we need a delta vector of matching dimension
            q_jnts_from_rhc=self._get_jnt_q_from_sol(node_idx=1).reshape(-1, 1)

            self._jnts_q_delta_expanded[self._rev_joints_idxs, :]=jnt_q_delta[self._rev_joints_idxs_red ,:]

            self._jnts_q_delta_expanded[self._continuous_joints_idxs_cos, :]=\
                np.cos(q_jnts_from_rhc[self._continuous_joints_idxs_red, :]) - \
                    np.cos(q_jnts[self._continuous_joints_idxs_red, :])
            self._jnts_q_delta_expanded[self._continuous_joints_idxs_sin, :]=\
                np.sin(q_jnts_from_rhc[self._continuous_joints_idxs_red, :]) - \
                    np.sin(q_jnts[self._continuous_joints_idxs_red, :])

            q_jnts=self._jnts_q_expanded.reshape(-1,1) # overriting with expanded jnts
            jnt_q_delta=self._jnts_q_delta_expanded.reshape(-1, 1) # overriting with expanded jnts

            self._alphas_q_jnts[self._continuous_joints_idxs_cos, :]=1.0 # open loop
            self._alphas_q_jnts[self._continuous_joints_idxs_sin, :]=1.0 # open loop

            # self._alphas_v_jnts[self._continuous_joints_idxs_red, :]=0.0 # open loop

        root_p_rhc.setBounds(lb=p_root,
            ub=p_root, nodes=0)
        root_q_rhc.setBounds(lb=q_root+self._alphas_q_root[3:7]*root_q_delta, 
            ub=q_root+self._alphas_q_root[3:7]*root_q_delta, nodes=0)
        jnts_q_rhc.setBounds(lb=q_jnts+self._alphas_q_jnts*jnt_q_delta, 
            ub=q_jnts+self._alphas_q_jnts*jnt_q_delta, nodes=0)
        if self._custom_opts["estimate_v_root"]:
            root_v_rhc.setBounds(lb=-self._v_inf[0:3], 
                ub=self._v_inf[0:3], nodes=0)
        else:
            root_v_rhc.setBounds(lb=v_root+self._alphas_twist_root[0:3, :]*v_root_delta, 
                ub=v_root+self._alphas_twist_root[0:3, :]*v_root_delta, nodes=0)
        root_omega_rhc.setBounds(lb=omega+self._alphas_twist_root[3:6, :]*omega_root_delta, 
            ub=omega+self._alphas_twist_root[3:6, :]*omega_root_delta, nodes=0)
        jnts_v_rhc.setBounds(lb=v_jnts+self._alphas_v_jnts*jnt_v_delta, 
            ub=v_jnts+self._alphas_v_jnts*jnt_v_delta, nodes=0)
        if self._custom_opts["lin_a_feedback"]:
            lin_a_prb.setBounds(lb=a_root[0:3, :]+self._alphas_a[0:3]*a_root_delta[0:3, :], 
                ub=a_root[0:3, :]+self._alphas_a[0:3]*a_root_delta[0:3, :], 
                nodes=0)
            
        # return state used for feedback
        q_state=np.concatenate((p_root, q_root, q_jnts),
                axis=0)
        v_state=np.concatenate((v_root, omega, v_jnts),
                axis=0)
        a_state=np.concatenate((a_root, a_jnts),
                axis=0)
        
        return (q_state, v_state, a_state)

    def _solve(self):
        
        if self._debug:
            return self._db_solve()
        else:
            return self._min_solve()
        
    def _min_solve(self):
        # minimal solve version -> no debug 
        robot_qstate=None
        robot_vstate=None
        robot_astate=None
        if self._open_loop:
            robot_qstate, robot_vstate, robot_astate = self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            robot_qstate, robot_vstate, robot_astate = self._update_closed_loop() # updates the TO ig and 
            # initial conditions using robot measurements
    
        self._pm.shift() # shifts phases of one dt
        if self._refs_in_hor_frame:
            # q_base=self.robot_state.root_state.get(data_type="q", 
            #     robot_idxs=self.controller_index).reshape(-1, 1)
            # q_full=self._get_full_q_from_sol(node_idx=1).reshape(-1, 1)
            # using internal base pose from rhc. in case of closed loop, it will be the meas state
            force_norm=None
            if self._custom_opts["use_force_feedback"]:
                contact_forces=self.robot_state.contact_wrenches.get(data_type="f", 
                    robot_idxs=self.controller_index_np,
                    contact_name=None).reshape(self.n_contacts,3)
                force_norm=np.linalg.norm(contact_forces, axis=1)
            self.rhc_refs.step(qstate=robot_qstate, vstate=robot_vstate,
                force_norm=force_norm)
        else:
            self.rhc_refs.step()
            
        try:
            converged = self._ti.rti() # solves the problem
            self.sol_counter = self.sol_counter + 1
            return not self._check_rhc_failure()
        except Exception as e: # fail in case of exceptions
            return False
    
    def _db_solve(self):

        self._timer_start = time.perf_counter()

        robot_qstate=None
        robot_vstate=None
        robot_astate=None
        if self._open_loop:
            robot_qstate, robot_vstate, robot_astate = self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            robot_qstate, robot_vstate, robot_astate = self._update_closed_loop() # updates the TO ig and 
            # initial conditions using robot measurements

        self._prb_update_time = time.perf_counter() 
        self._pm.shift() # shifts phases of one dt
        self._phase_shift_time = time.perf_counter()

        if self._refs_in_hor_frame:
            # q_base=self.robot_state.root_state.get(data_type="q", 
            #     robot_idxs=self.controller_index).reshape(-1, 1)
            # q_full=self._get_full_q_from_sol(node_idx=1).reshape(-1, 1)
            # using internal base pose from rhc. in case of closed loop, it will be the meas state
            force_norm=None
            if self._custom_opts["use_force_feedback"]:
                contact_forces=self.robot_state.contact_wrenches.get(data_type="f", 
                    robot_idxs=self.controller_index_np,
                    contact_name=None).reshape(self.n_contacts,3)
                force_norm=np.linalg.norm(contact_forces, axis=1)
            self.rhc_refs.step(qstate=robot_qstate, vstate=robot_vstate,
                force_norm=force_norm)
        else:
            self.rhc_refs.step()
             
        self._task_ref_update_time = time.perf_counter() 
    
        try:
            converged = self._ti.rti() # solves the problem
            self._rti_time = time.perf_counter() 
            self.sol_counter = self.sol_counter + 1
            self._update_db_data()
            return not self._check_rhc_failure()
        except Exception as e: # fail in case of exceptions
            if self._verbose:
                exception = f"Rti() for controller {self.controller_index} failed" + \
                f" with exception{type(e).__name__}"
                Journal.log(self.__class__.__name__,
                    "solve",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = False)
            self._update_db_data()
            return False
    
    def _get_fail_idx(self):
        
        self._explosion_idx_buffer[:, self._expl_idx_buffer_counter]=self._get_explosion_idx()
        self._expl_idx_buffer_counter+=1
        self._expl_idx_counter+=1
        if self._expl_idx_counter%self._expl_idx_window_size==0:
            self._expl_idx_buffer_counter=0 # restart from 0
        
        running_avrg=np.mean(self._explosion_idx_buffer).item()
        
        return running_avrg
    
    def _get_explosion_idx(self):
        explosion_index = self._get_rhc_constr_viol() + self._get_rhc_cost()*self._fail_idx_scale
        return explosion_index

    def _update_db_data(self):

        self._profiling_data_dict["problem_update_dt"] = self._prb_update_time - self._timer_start
        self._profiling_data_dict["phases_shift_dt"] = self._phase_shift_time - self._prb_update_time
        self._profiling_data_dict["task_ref_update"] = self._task_ref_update_time - self._phase_shift_time
        self._profiling_data_dict["rti_solve_dt"] = self._rti_time - self._task_ref_update_time
        self.rhc_costs.update(self._ti.solver_rti.getCostsValues())
        self.rhc_constr.update(self._ti.solver_rti.getConstraintsValues())

    def _reset(self):
        
        # reset task interface (ig, solvers, etc..) + 
        # phase manager and sets bootstap as solution
        self._gm.reset()
        self._explosion_idx_buffer[:, :]=self._get_explosion_idx() # reset with data from reset solution
        self._expl_idx_counter=0.0
        self._expl_idx_buffer_counter=0

    def _get_cost_data(self):
        
        cost_dict = self._ti.solver_rti.getCostsValues()
        cost_names = list(cost_dict.keys())
        cost_dims = [1] * len(cost_names) # costs are always scalar
        return cost_names, cost_dims
    
    def _get_constr_data(self):
        
        constr_dict = self._ti.solver_rti.getConstraintsValues()
        constr_names = list(constr_dict.keys())
        constr_dims = [-1] * len(constr_names)
        i = 0
        for constr in constr_dict:
            constr_val = constr_dict[constr]
            constr_shape = constr_val.shape
            constr_dims[i] = constr_shape[0]
            i+=1
        return constr_names, constr_dims
    
    def _get_q_from_sol(self):
        full_q=self._ti.solution['q'].astype(self._dtype)
        if self._custom_opts["replace_continuous_joints"]:
            return full_q
        else:
            cont_jnts=full_q[self._continuous_joints_idxs_qfull, :]
            cos=cont_jnts[::2, :]
            sin=cont_jnts[1::2, :]
            # copy root
            self._full_q_reduced[0:7, :]=full_q[0:7, :]
            # copy rev joint vals
            self._full_q_reduced[self._rev_joints_idxs_red_qfull, :]=full_q[self._rev_joints_idxs_qfull, :]
            # and continuous
            angle=np.arctan2(sin, cos)
            self._full_q_reduced[self._continuous_joints_idxs_red_qfull, :]=angle
            return self._full_q_reduced

    def _get_v_from_sol(self):
        return self._ti.solution['v'].astype(self._dtype)
    
    def _get_a_from_sol(self):
        return self._ti.solution['a'].astype(self._dtype)
    
    def _get_a_dot_from_sol(self):
        return None
    
    def _get_f_from_sol(self):
        # to be overridden by child class
        contact_names =self._get_contacts() # we use controller-side names
        try: 
            data=[]
            for key in contact_names:
                contact_f=self._ti.solution["f_" + key].astype(self._dtype)
                np.nan_to_num(contact_f, nan=1e6, posinf=1e6, neginf=-1e6, copy=False)
                np.clip(a=contact_f, a_max=1e6, a_min=-1e6, out=contact_f)
                data.append(contact_f)
            return np.concatenate(data, axis=0)
        except:
            return None
            
    def _get_f_dot_from_sol(self):
        # to be overridden by child class
        return None
    
    def _get_eff_from_sol(self):
        # to be overridden by child class
        return None
    
    def _get_cost_from_sol(self,
                    cost_name: str):
        return self.rhc_costs[cost_name]
    
    def _get_constr_from_sol(self,
                    constr_name: str):
        return self.rhc_constr[constr_name]