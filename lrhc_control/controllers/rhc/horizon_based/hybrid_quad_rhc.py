from control_cluster_bridge.controllers.rhc import RHController
# from perf_sleep.pyperfsleep import PerfSleep
# from control_cluster_bridge.utilities.cpu_utils.core_utils import get_memory_usage

from lrhc_control.controllers.rhc.horizon_based.horizon_imports import * 
from lrhc_control.controllers.rhc.horizon_based.hybrid_quad_rhc_refs import HybridQuadRhcRefs
from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

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
            "lin_a_feedback": False}

        self._custom_opts.update(custom_opts)
        
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

        self._fail_idx_scale=1e-2
        self._pred_node_idx=self._n_nodes-1

        self._nq=self.nq()
        self._nq_jnts=self._nq-7# assuming floating base
        self._nv=self.nv()
        self._nv_jnts=self._nv-6

    def _init_problem(self,
            fixed_jnt_patterns: List[str] = None,
            foot_linkname: str = None,
            flight_duration: int = 10,
            step_height: float = 0.12,
            keep_yaw_vert: bool = False,
            yaw_vertical_weight: float = 2.0,
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
        else:
            self._custom_opts["replace_continuous_joints"]=True
            Journal.log(self.__class__.__name__,
                "_init_problem",
                f"No continuous joints were found.",
                LogType.INFO,
                throw_when_excep=True)

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
            phase_force_reg=self._phase_force_reg,
            custom_opts=self._custom_opts,
            flight_duration=flight_duration,
            step_height=step_height,
            dh=0.0)
                
        # self._ti.model.q.setBounds(self._ti.model.q0, self._ti.model.q0, nodes=0)
        # self._ti.model.v.setBounds(self._ti.model.v0, self._ti.model.v0, nodes=0)
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
    
    def _get_jnt_id(self, jnt_name):
        return self._kin_dyn.joint_iq(jnt_name)
    
    def _init_rhc_task_cmds(self):
        
        rhc_refs = HybridQuadRhcRefs(gait_manager=self._gm,
            robot_index=self.controller_index,
            namespace=self.namespace,
            safe=False, 
            verbose=self._verbose,
            vlevel=VLevel.V2,
            use_force_feedback=self._custom_opts["use_force_feedback"],
            use_fixed_flights=self._custom_opts["fixed_flights"])
        
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

        return self._ti.solution['q'][0:7, node_idx].reshape(1, 7)
    
    def _get_full_q_from_sol(self, node_idx=1):

        return self._ti.solution['q'][:, node_idx].reshape(1, -1)
    
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
        
        full_jnts_q=self._ti.solution['q'][7:, node_idx:node_idx+1].reshape(1,-1)

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

        return self._get_v_from_sol()[6:, node_idx].reshape(1,  
                    self._nv_jnts)

    def _get_jnt_a_from_sol(self, node_idx=1):

        return self._get_a_from_sol()[6:, node_idx].reshape(1,
                    self._nv_jnts)

    def _get_jnt_eff_from_sol(self, node_idx=0):
        
        efforts_on_node = self._ti.eval_efforts_on_node(node_idx=node_idx)
        
        return efforts_on_node[6:, 0].reshape(1,  
                self._nv_jnts)
    
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
    
    def _assemble_meas_robot_state(self,
                        x_opt = None,
                        close_all: bool=False):

        # overrides parent

        # jnts
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index).reshape(1, -1)
        v_jnts = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index).reshape(1, -1)
        a_jnts = self.robot_state.jnts_state.get(data_type="a", robot_idxs=self.controller_index).reshape(1, -1)

        # root
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index).reshape(1, -1)
        p = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index).reshape(1, -1)
        v_root = self.robot_state.root_state.get(data_type="v", robot_idxs=self.controller_index).reshape(1, -1)
        a_root = self.robot_state.root_state.get(data_type="a", robot_idxs=self.controller_index).reshape(1, -1)
        omega = self.robot_state.root_state.get(data_type="omega", robot_idxs=self.controller_index).reshape(1, -1)
        alpha_root = self.robot_state.root_state.get(data_type="alpha", robot_idxs=self.controller_index).reshape(1, -1)

        if (not len(self._continuous_joints)==0): # we need do expand some meas. rev jnts to So2
            # copy rev joints
            self._jnts_q_expanded[:, self._rev_joints_idxs]=q_jnts[:, self._rev_joints_idxs_red]
            self._jnts_q_expanded[:, self._continuous_joints_idxs_cos]=np.cos(q_jnts[:, self._continuous_joints_idxs_red]) # cos
            self._jnts_q_expanded[:, self._continuous_joints_idxs_sin]=np.sin(q_jnts[:, self._continuous_joints_idxs_red]) # sin
            q_jnts=self._jnts_q_expanded
        # meas twist is assumed to be provided in BASE link!!!
        if not close_all: # use internal MPC for the base and joints
            p[:, 0:3]=self._get_root_full_q_from_sol(node_idx=1)[:, 0:3] # base pos is open loop
            v_root[0:3,:]=self._get_root_twist_from_sol(node_idx=1)[:, 0:3] # lin vel is usually not known
            # q_jnts[:, :]=self._get_jnt_q_from_sol(node_idx=1,reduce=False,clamp=False)           
            # v_jnts[:, :]=self._get_jnt_v_from_sol(node_idx=1)
        # r_base = Rotation.from_quat(q_root.flatten()).as_matrix() # from base to world (.T the opposite)
        
        if x_opt is not None:
            # CHECKING q_root for sign consistency!
            # numerical problem: two quaternions can represent the same rotation
            # if difference between the base q in the state x on first node and the sensed q_root < 0, change sign
            state_quat_conjugate = np.copy(x_opt[3:7, 0])
            state_quat_conjugate[:3] *= -1.0
            # normalize the quaternion
            state_quat_conjugate = state_quat_conjugate / np.linalg.norm(x_opt[3:7, 0])
            diff_quat = self._quaternion_multiply(q_root.flatten(), state_quat_conjugate)
            if diff_quat[3] < 0:
                q_root[:, :] = -q_root[:, :]
        
        qv_state=np.concatenate((p, q_root, q_jnts, v_root, omega, v_jnts),
                axis=1).reshape(-1,1)
        a_state=np.concatenate((a_root, alpha_root, a_jnts),
                axis=1).reshape(-1,1)
        return (qv_state, a_state)
    
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

        qv_state, a_state=self._set_is_open()

        # robot_state=xig[:, 0]
        # # open loop update:
        # self._prb.setInitialState(x0=robot_state) # (xig has been shifted, so node 0
        # # is node 1 in the last opt solution)

        return qv_state, a_state
    
    def _update_closed_loop(self):
        
        # set initial guess for controller
        xig, _ = self._set_ig()
        # set initial state
        qv_state, a_state=self._set_is()

        return qv_state, a_state
    
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
        qv_state=np.concatenate((q_full_root, q_jnts, twist_root, v_jnts),
                axis=0)
        
        return (qv_state, None)
            
    def _set_is(self):
        
        # measurements
        p_root = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index).reshape(-1, 1)
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
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

        # overriding states with rhc data
        root_p_from_rhc=self._get_root_full_q_from_sol(node_idx=1)[:, 0:3].reshape(-1, 1)
        p_root[:, :]=root_p_from_rhc # position in open loop
        # root_v_from_rhc=self._get_root_twist_from_sol(node_idx=1)[:, 0:3].reshape(-1, 1)
        # v_root[:, :]=root_v_from_rhc
        
        # rhc variables to be set
        q=self._prb.getVariables("q") # .setBounds()
        root_p_rhc=q[0:3] # root p
        root_q_rhc=q[3:7] # root orientation
        jnts_q_rhc=q[7:] # jnts q
        vel=self._prb.getVariables("v")
        # root_v_rhc=vel[0:3] # lin v.
        root_omega_rhc=vel[3:6] # omega
        jnts_v_rhc=vel[6:] # jnts v
        acc=self._prb.getVariables("a")
        lin_a_prb=acc[0:3] # lin acc
        
        # close state on known quantities
        root_p_rhc.setBounds(lb=p_root,
            ub=p_root, nodes=0)
        root_q_rhc.setBounds(lb=q_root, 
            ub=q_root, nodes=0)
        jnts_q_rhc.setBounds(lb=q_jnts, 
            ub=q_jnts, nodes=0)
        # root_v_rhc.setBounds(lb=v_root, 
        #     ub=v_root, nodes=0)
        root_omega_rhc.setBounds(lb=omega, 
            ub=omega, nodes=0)
        jnts_v_rhc.setBounds(lb=v_jnts, 
            ub=v_jnts, nodes=0)
        if self._custom_opts["lin_a_feedback"]:
            # write base lin acceleration from meas
            lin_a_prb.setBounds(lb=a_root[0:3, :], 
                ub=a_root[0:3, :], 
                nodes=0)
        
        # return state used for feedback
        qv_state=np.concatenate((p_root, q_root, q_jnts, v_root, omega, v_jnts),
                axis=0)
        a_state=np.concatenate((a_root, a_jnts),
                axis=0)
        
        return (qv_state, a_state)

    def _solve(self):
        
        if self._debug:
            return self._db_solve()
        else:
            return self._min_solve()
        
    def _min_solve(self):
        # minimal solve version -> no debug 
        robot_state=None
        if self._open_loop:
            robot_state, _ =self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            robot_state, _ =self._update_closed_loop() # updates the TO ig and 
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
            self.rhc_refs.step(q_full=robot_state,
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

        if self._open_loop:
            self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            self._update_closed_loop() # updates the TO ig and 
            # initial conditions using robot measurements

        self._prb_update_time = time.perf_counter() 
        self._pm.shift() # shifts phases of one dt
        self._phase_shift_time = time.perf_counter()

        if self._refs_in_hor_frame:
            # q_base=self.robot_state.root_state.get(data_type="q", 
            #     robot_idxs=self.controller_index).reshape(-1, 1)
            q_full=self._get_full_q_from_sol(node_idx=1).reshape(-1, 1)
            # using internal base pose from rhc. in case of closed loop, it will be the meas state
            force_norm=None
            if self._custom_opts["use_force_feedback"]:
                contact_forces=self.robot_state.contact_wrenches.get(data_type="f", 
                    robot_idxs=self.controller_index_np,
                    contact_name=None).reshape(self.n_contacts,3)
                force_norm=np.linalg.norm(contact_forces, axis=1)
            self.rhc_refs.step(q_full=q_full,
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
        full_q=self._ti.solution['q']
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
        return self._ti.solution['v']
    
    def _get_a_from_sol(self):
        return self._ti.solution['a']
    
    def _get_a_dot_from_sol(self):
        return None
    
    def _get_f_from_sol(self):
        # to be overridden by child class
        contact_names =self._get_contacts() # we use controller-side names
        try: 
            data = [self._ti.solution["f_" + key] for key in contact_names]
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