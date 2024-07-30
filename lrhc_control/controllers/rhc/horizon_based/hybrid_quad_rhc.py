from control_cluster_bridge.controllers.rhc import RHController

from lrhc_control.controllers.rhc.horizon_based.horizon_imports import * 

from lrhc_control.controllers.rhc.horizon_based.hybrid_quad_rhc_refs import HybridQuadRhcRefs
from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

import numpy as np
from scipy.spatial.transform import Rotation

import os
# import shutil

import time
from abc import ABC, abstractmethod

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
            timeout_ms: int = 60000
            ):

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

        super().__init__(srdf_path=srdf_path,
                        n_nodes=n_nodes,
                        dt=dt,
                        namespace=self.robot_name,
                        dtype=dtype,
                        verbose=verbose, 
                        debug=debug,
                        timeout_ms=timeout_ms)

        self.rhc_costs={}
        self.rhc_constr={}

        self._fail_idx_scale=1e-2
    
    def _get_quat_remap(self):
        # overrides parent
        return [1, 2, 3, 0] # mapping from robot quat. to Horizon's quaternion convention
    
    def _quaternion_multiply(self, 
                    q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])
    
    @abstractmethod
    def _init_problem(self):
        pass
    
    @abstractmethod
    def _reset_contact_timeline(self):
        pass
    
    @abstractmethod
    def _init_contact_timelines(self):
        pass
    
    def _reset_contact_timeline(self):
        for c in self._model.cmap.keys():
            # fill timeline with stances
            stance = self._c_timelines[c].getRegisteredPhase(f'stance_{c}_short')
            while self._c_timelines[c].getEmptyNodes() > 0:
                self._c_timelines[c].addPhase(stance)
            # f reg
            f_stance = self._f_reg_timelines[c].getRegisteredPhase(f'freg_{c}_short')
            for i in range(self._n_nodes-1): # not defined on last node
                self._f_reg_timelines[c].addPhase(f_stance)

    @abstractmethod
    def _init_rhc_task_cmds(self):
        pass
    
    def _get_contact_names(self):

        return list(self._ti.model.cmap.keys())
    
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

    def _get_robot_mass(self):

        return self._kin_dyn.mass()

    def _get_cmd_jnt_q_from_sol(self):
        
        # wrapping joint q commands between 2pi and -2pi
        # (to be done for the simulator)
        return np.fmod(self._ti.solution['q'][7:, 1], 2*np.pi).reshape(1,  
                    self.n_dofs)
    
    def _get_cmd_jnt_v_from_sol(self):

        return self._ti.solution['v'][6:, 1].reshape(1,  
                    self.n_dofs)

    def _get_cmd_jnt_eff_from_sol(self):
        
        efforts_on_first_node = self._ti.eval_efforts_on_first_node()

        return efforts_on_first_node[6:, 0].reshape(1,  
                        self.n_dofs)
    
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
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
        v_jnts = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index).reshape(-1, 1)
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
        p = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index).reshape(-1, 1)
        if not close_all: # use internal MPC for the base
            p[0:3,:]=self._ti.solution['q'][0:3, 1:2] # base pos is open loop
        v_root = self.robot_state.root_state.get(data_type="v", robot_idxs=self.controller_index).reshape(-1, 1)
        omega = self.robot_state.root_state.get(data_type="omega", robot_idxs=self.controller_index).reshape(-1, 1)
        
        # we need twist in local base frame, but measured one is global
        if x_opt is not None:
            # CHECKING q_root for sign consistency!
            # numerical problem: two quaternions can represent the same rotation
            # if difference between the base q in the state x on first node and the sensed q_root < 0, change sign
            state_quat_conjugate = np.copy(x_opt[3:7, 0])
            state_quat_conjugate[:3] *= -1.0
            # normalize the quaternion
            state_quat_conjugate = state_quat_conjugate / np.linalg.norm(x_opt[3:7, 0])
            diff_quat = self._quaternion_multiply(q_root, state_quat_conjugate)
            if diff_quat[3] < 0:
                q_root[:] = -q_root

        r_base = Rotation.from_quat(q_root.flatten()).as_matrix()

        return np.concatenate((p, q_root, q_jnts, r_base.T @ v_root, r_base.T @ omega, v_jnts),
                axis=0)
    
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

        # open loop update:
        self._prb.setInitialState(x0=xig[:, 0]) # (xig has been shifted, so node 0
        # is node 1 in the last opt solution)
    
    def _update_closed_loop(self):

        self._set_ig()

        # sets state on node 0 from measurements
        robot_state = self._assemble_meas_robot_state(x_opt=self._ti.solution['x_opt'],
                                        close_all=self._close_loop_all)

        self._prb.setInitialState(x0=
                        robot_state)
    
    def _solve(self):
        
        if self._debug:
            return self._db_solve()
        else:
            return self._min_solve()
        
    def _min_solve(self):
        # minimal solve version -> no debug 
        if self._open_loop:
            self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            self._update_closed_loop() # updates the TO ig and 
            # initial conditions using robot measurements
    
        self._pm.shift() # shifts phases of one dt
        if self._refs_in_hor_frame:
            # q_base=self.robot_state.root_state.get(data_type="q", 
            #     robot_idxs=self.controller_index).reshape(-1, 1)
            q_base=self._ti.solution['q'][3:7, 0].reshape(-1, 1) # using internal 
            # base pose from rhc. in case of closed loop, it will be the meas state
            self.rhc_refs.step(q_base=q_base)
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
            q_base=self._ti.solution['q'][3:7, 0].reshape(-1, 1) # using internal 
            # base pose from rhc. in case of closed loop, it will be the meas state
            self.rhc_refs.step(q_base=q_base) # updates rhc references
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
        # phase manager
        self._gm.reset()
        # we also re-initialize contact timelines
        # self._reset_conta()
        # resets rhc references
        if self.rhc_refs is not None:
            self.rhc_refs.reset(p_ref=np.atleast_2d(self._base_init)[:, 0:3], 
                        q_ref=np.atleast_2d(self._base_init)[:, 3:7]
                        )

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
        return self._ti.solution['q']

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