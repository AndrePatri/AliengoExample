from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager
from lrhc_control.controllers.rhc.horizon_based.utils.math_utils import hor2w_frame

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from typing import Union

import numpy as np

class HybridQuadRhcRefs(RhcRefs):

    def __init__(self, 
            gait_manager: GaitManager, 
            robot_index_shm: int,
            robot_index_view: int,
            namespace: str, # namespace used for shared mem
            verbose: bool = True,
            vlevel: bool = VLevel.V2,
            safe: bool = True,
            use_force_feedback: bool = False,
            use_fixed_flights: bool = False,
            optimize_mem: bool = False):
        
        self.robot_index = robot_index_shm
        self.robot_index_view = robot_index_view
        self.robot_index_np_view = np.array(self.robot_index_view)

        self._step_idx = 0
        self._print_frequency = 100

        self._verbose = verbose

        self._use_force_feedback=use_force_feedback
        self._use_fixed_flights=use_fixed_flights

        if optimize_mem:
            super().__init__( 
                    is_server=False,
                    with_gpu_mirror=False,
                    namespace=namespace,
                    safe=safe,
                    verbose=verbose,
                    vlevel=vlevel,
                    optimize_mem=optimize_mem,
                    n_robots=1, # we just need the row corresponding to this controller
                    n_jnts=None, # got from server
                    n_contacts=None # got from server
                    )
        else:
            super().__init__( 
                is_server=False,
                with_gpu_mirror=False,
                namespace=namespace,
                safe=safe,
                verbose=verbose,
                vlevel=vlevel)
            
        if not isinstance(gait_manager, GaitManager):
            exception = f"Provided gait_manager argument should be of GaitManager type!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
               
        self.gait_manager = gait_manager

        self.timeline_names = self.gait_manager.timeline_names

        # task interfaces from horizon for setting commands to rhc
        self._get_tasks()

        self._p_ref_default=np.zeros((1, 3))
        self._q_ref_default=np.zeros((1, 4))
        self._q_ref_default[0, 0]=1

    def _get_tasks(self):
        # can be overridden by child
        # cartesian tasks are in LOCAL_WORLD_ALIGNED (frame centered at distal link, oriented as WORLD)
        self.base_lin_velxy = self.gait_manager.task_interface.getTask('base_lin_velxy')
        self.base_lin_velz = self.gait_manager.task_interface.getTask('base_lin_velz')
        self.base_omega = self.gait_manager.task_interface.getTask('base_omega')
        self.base_height = self.gait_manager.task_interface.getTask('base_height')

    def run(self):

        super().run()
        if not (self.robot_index < self.rob_refs.n_robots()):
            exception = f"Provided \(0-based\) robot index {self.robot_index} exceeds number of " + \
                " available robots {self.rob_refs.n_robots()}."
            Journal.log(self.__class__.__name__,
                "run",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        contact_names = list(self.gait_manager.task_interface.model.cmap.keys())
        if not (self.n_contacts() == len(contact_names)):
            exception = f"N of contacts within problem {len(contact_names)} does not match n of contacts {self.n_contacts()}"
            Journal.log(self.__class__.__name__,
                "run",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        # set some defaults from gait manager
        for i in range(self.n_contacts()):
            self.flight_settings.set(data=self.gait_manager.get_flight_duration(contact_name=contact_names[i]),
                data_type="len",
                robot_idxs=self.robot_index_np_view,
                contact_idx=i)
            self.flight_settings.set(data=self.gait_manager.get_step_apexdh(contact_name=contact_names[i]),
                data_type="apex_dpos",
                robot_idxs=self.robot_index_np_view,
                contact_idx=i)
            self.flight_settings.set(data=self.gait_manager.get_step_enddh(contact_name=contact_names[i]),
                data_type="end_dpos",
                robot_idxs=self.robot_index_np_view,
                contact_idx=i)
        
        self.flight_settings.synch_retry(row_index=self.robot_index,
            col_index=0,
            row_index_view=self.robot_index_view,
            n_rows=1,
            n_cols=self.flight_settings.n_cols,
            read=False)

    def step(self, qstate: np.ndarray = None,
        vstate: np.ndarray = None,
        force_norm: np.ndarray = None):

        if self.is_running():
            
            # updates robot refs from shared mem
            self.rob_refs.synch_from_shared_mem(robot_idx=self.robot_index, robot_idx_view=self.robot_index_view)
            self.phase_id.synch_all(read=True, retry=True,
                        row_index=self.robot_index,
                        row_index_view=self.robot_index_view)
            self.contact_flags.synch_all(read=True, retry=True,
                        row_index=self.robot_index,
                        row_index_view=self.robot_index_view)
            self.flight_settings.synch_all(read=True, retry=True,
                        row_index=self.robot_index,
                        row_index_view=self.robot_index_view)
            self._set_contact_phases(q_full=qstate)

            # updated internal references with latest available ones
            q_base=qstate[3:7,0:1] # quaternion
            self._apply_refs_to_tasks(q_base=q_base)
            
            # if self._use_force_feedback:
            #     self._set_force_feedback(force_norm=force_norm)

            self._step_idx +=1
        
        else:
            exception = f"{self.__class__.__name__} is not running"
            Journal.log(self.__class__.__name__,
                "step",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _set_contact_phases(self,
        q_full: np.ndarray):

        # phase_id = self.phase_id.read_retry(row_index=self.robot_index,
        #                         col_index=0)[0]
        
        contact_flags_refs = self.contact_flags.get_numpy_mirror()[self.robot_index_np_view, :]
        target_n_limbs_in_contact=np.sum(contact_flags_refs).item()
        if target_n_limbs_in_contact==0:
            target_n_limbs_in_contact=4

        is_contact = contact_flags_refs.flatten().tolist() 
        
        for i in range(len(is_contact)): # loop through contact timelines
            timeline_name = self.timeline_names[i]
            
            self.gait_manager.set_f_reg(contact_name=timeline_name,
                scale=target_n_limbs_in_contact)

            if is_contact[i]==False: # flight phase
                
                len_now=int(self.flight_settings.get(data_type="len",
                    robot_idxs=self.robot_index_np_view,
                    contact_idx=i).item())
                apex_now=self.flight_settings.get(data_type="apex_dpos",
                    robot_idxs=self.robot_index_np_view,
                    contact_idx=i).item()
                end_now=self.flight_settings.get(data_type="end_dpos",
                    robot_idxs=self.robot_index_np_view,
                    contact_idx=i).item()
                # set flight phase property depending on last value
                self.gait_manager.set_flight_duration(contact_name=timeline_name,
                    val=len_now)
                self.gait_manager.set_step_apexdh(contact_name=timeline_name,
                    val=apex_now)
                self.gait_manager.set_step_enddh(contact_name=timeline_name,
                    val=end_now)
                
                # add flight phase
                self.gait_manager.add_flight(contact_name=timeline_name,
                    robot_q=q_full)
                                
            else: # contact phase
                self.gait_manager.add_stand(contact_name=timeline_name)

            flight_info=self.gait_manager.get_flight_info(timeline_name)
            # self._flight_info=None
            if flight_info is not None:
                pos=flight_info[0]
                length=flight_info[1]
                self.flight_info.write_retry(pos, 
                    row_index=self.robot_index,
                    col_index=i, # contact i
                    row_index_view=self.robot_index_np_view,
                    )
            else:
                length=0

            self.flight_info.write_retry(length, 
                row_index=self.robot_index,
                col_index=len(is_contact)+i,
                row_index_view=self.robot_index_np_view)
                # self._flight_info[2] # n nodes
            
            self.gait_manager.check_horizon_full(timeline_name=timeline_name)
        
        self.gait_manager.update()
        
    def _handle_contact_phases_free(self):

        pz_ref=self.rob_refs.contact_pos.get(data_type = "p_z", 
                robot_idxs=self.robot_index_np_view).reshape(-1, 1)
        
        thresh=0.01
        is_contact=~(pz_ref>thresh)
        target_n_limbs_in_contact=np.sum(is_contact).item()
        if target_n_limbs_in_contact==0:
            target_n_limbs_in_contact=4

        is_contact_list = is_contact.flatten().tolist() 

        for i in range(len(is_contact)): # loop through contact timelines
            timeline_name = self._timeline_names[i]
            timeline = self.gait_manager._contact_timelines[timeline_name]
            for contact_force_ref in self._f_reg_ref[i]: # set for references depending on n of contacts and contact forces per-contact
                
                scale=self._n_forces_per_contact[i]*target_n_limbs_in_contact
                # scale=4 # just regularize
                contact_force_ref.assign(self._total_weight/scale)

            self.gait_manager.set_ref_pos(timeline_name=timeline_name,
                ref_height=pz_ref[i,:],
                threshold=thresh)
            
            # writing flight info
            self._flight_info=self.gait_manager.get_flight_info(timeline_name)
            # self._flight_info=None
            if self._flight_info is not None:
                pos=self._flight_info[0]
                length=self._flight_info[1]
                self.flight_info.write_retry(pos, 
                    row_index=self.robot_index,
                    col_index=i,
                    row_index_view=self.robot_index_np_view)
            else:
                length=0
            self.flight_info.write_retry(length, 
                row_index=self.robot_index,
                col_index=len(is_contact)+i,
                row_index_view=self.robot_index_np_view)
                # self._flight_info[2] # n nodes

            if timeline.getEmptyNodes() > 0: # if there's space, always add a stance
                self.gait_manager.add_stand(timeline_name)

            if timeline.getEmptyNodes() > 0:
                error = f"Empty nodes detected over the horizon! Make sure to fill the whole horizon with valid phases!!"
                Journal.log(self.__class__.__name__,
                    "step",
                    error,
                    LogType.EXCEP,
                    throw_when_excep = True)
      
    def _apply_refs_to_tasks(self, q_base = None):
        # overrides parent
        if q_base is not None: # rhc refs are assumed to be specified in the so called "horizontal" 
            # frame, i.e. a vertical frame, with the x axis aligned with the projection of the base x axis
            # onto the plane
            root_pose = self.rob_refs.root_state.get(data_type = "q_full", 
                                robot_idxs=self.robot_index_np_view).reshape(-1, 1) # this should also be
            # rotated into the horizontal frame (however, for now only the z componet is used, so it's ok)
            
            root_twist_ref = self.rob_refs.root_state.get(data_type="twist", 
                                robot_idxs=self.robot_index_np_view).reshape(-1, 1)

            root_twist_ref_h = root_twist_ref.copy() 

            hor2w_frame(root_twist_ref, q_base, root_twist_ref_h)
            
            if self.base_lin_velxy is not None:
                self.base_lin_velxy.setRef(root_twist_ref_h[0:2, :])
            if self.base_omega is not None:
                self.base_omega.setRef(root_twist_ref_h[3:, :])
            if self.base_lin_velz is not None:
                self.base_lin_velz.setRef(root_twist_ref_h[2:3, :])
            if self.base_height is not None:
                self.base_height.setRef(root_pose) 
        else:
            root_pose = self.rob_refs.root_state.get(data_type = "q_full", 
                                robot_idxs=self.robot_index_np_view).reshape(-1, 1)
            root_twist_ref = self.rob_refs.root_state.get(data_type="twist", 
                                robot_idxs=self.robot_index_np_view).reshape(-1, 1)

            if self.base_lin_velxy is not None:
                self.base_lin_velxy.setRef(root_twist_ref[0:2, :])
            if self.base_omega is not None:
                self.base_omega.setRef(root_twist_ref[3:, :])
            if self.base_lin_velz is not None:
                self.base_lin_velz.setRef(root_twist_ref[2:3, :])
            if self.base_height is not None:
                self.base_height.setRef(root_pose)
    
    # def _set_force_feedback(self,
    #         force_norm: np.ndarray = None):
    
    #     is_contact=force_norm>1.0

    #     for i in range(len(is_contact)):
    #         timeline_name = self._timeline_names[i]
    #         self.gait_manager.set_force_feedback(timeline_name=timeline_name,
    #             force_norm=force_norm[i])

    #         if not is_contact[i]:


    def set_default_refs(self,
        p_ref: np.ndarray,
        q_ref: np.ndarray):

        self._p_ref_default[:, :]=p_ref
        self._q_ref_default[:, :]=q_ref
    
    def set_alpha(self, alpha:float):
        # set provided value
        alpha_shared=self.alpha.get_numpy_mirror()
        alpha_shared[self.robot_index_np_view, :] = alpha
        self.alpha.synch_retry(row_index=self.robot_index, col_index=0, 
                row_index_view=self.robot_index_view,
                n_rows=1, n_cols=self.alpha.n_cols,
                read=False)
            
    def get_alpha(self):
        self.alpha.synch_retry(row_index=self.robot_index, col_index=0, 
                    row_index_view=self.robot_index_view,
                    n_rows=1, n_cols=self.alpha.n_cols,
                    read=True)
        alpha=self.alpha.get_numpy_mirror()[self.robot_index_np_view, :].item()
        return alpha

    def set_bound_relax(self, bound_relax:float):
        # set provided value
        bound_rel_shared=self.bound_rel.get_numpy_mirror()
        bound_rel_shared[self.robot_index_np_view, :] = bound_relax
        self.bound_rel.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, 
            row_index_view=self.robot_index_view,
            n_cols=self.alpha.n_cols,
            read=False)

    def reset(self):

        if self.is_running():

            # resets shared mem
            contact_flags_current = self.contact_flags.get_numpy_mirror()
            phase_id_current = self.phase_id.get_numpy_mirror()
            contact_flags_current[self.robot_index_np_view, :] = np.full((1, self.n_contacts()), dtype=np.bool_, fill_value=True)
            phase_id_current[self.robot_index_np_view, :] = -1 # defaults to custom phase id

            contact_pos_current=self.rob_refs.contact_pos.get_numpy_mirror()
            contact_pos_current[self.robot_index_np_view, :] = 0.0

            flight_info_current=self.flight_info.get_numpy_mirror()
            flight_info_current[self.robot_index_np_view, :] = 0.0

            alpha=self.alpha.get_numpy_mirror()
            alpha[self.robot_index_np_view, :] = 0.0

            self.rob_refs.root_state.set(data_type="p", data=self._p_ref_default, robot_idxs=self.robot_index_np_view)
            self.rob_refs.root_state.set(data_type="q", data=self._q_ref_default, robot_idxs=self.robot_index_np_view)
            
            self.rob_refs.root_state.set(data_type="twist", data=np.zeros((1, 6)), robot_idxs=self.robot_index_np_view)
                                           
            self.contact_flags.synch_retry(row_index=self.robot_index, col_index=0, 
                                    row_index_view=self.robot_index_view,
                                    n_rows=1, n_cols=self.contact_flags.n_cols,
                                    read=False)
            self.phase_id.synch_retry(row_index=self.robot_index, col_index=0, 
                                    row_index_view=self.robot_index_view,
                                    n_rows=1, n_cols=self.phase_id.n_cols,
                                    read=False)
            self.rob_refs.root_state.synch_retry(row_index=self.robot_index, col_index=0, 
                                    row_index_view=self.robot_index_view,
                                    n_rows=1, n_cols=self.rob_refs.root_state.n_cols,
                                    read=False)

            self.rob_refs.contact_pos.synch_retry(row_index=self.robot_index, col_index=0, 
                                    row_index_view=self.robot_index_view,
                                    n_rows=1, n_cols=self.rob_refs.contact_pos.n_cols,
                                    read=False)
            
            self.flight_info.synch_retry(row_index=self.robot_index, col_index=0, 
                                row_index_view=self.robot_index_view,
                                n_rows=1, n_cols=self.flight_info.n_cols,
                                read=False)
            
            
            
            # should also empty the timeline for stepping phases
            self._step_idx = 0

            self._flight_info=None

        else:
            exception = f"Cannot call reset() since run() was not called!"
            Journal.log(self.__class__.__name__,
                "reset",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

