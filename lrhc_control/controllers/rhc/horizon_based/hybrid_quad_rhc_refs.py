from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager
from lrhc_control.controllers.rhc.horizon_based.utils.math_utils import hor2w_frame

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from typing import Union

import numpy as np

class HybridQuadRhcRefs(RhcRefs):

    def __init__(self, 
            gait_manager: GaitManager, 
            robot_index: int,
            namespace: str, # namespace used for shared mem
            verbose: bool = True,
            vlevel: bool = VLevel.V2,
            safe: bool = True,
            use_force_feedback: bool = False,
            use_fixed_flights: bool = False):
        
        self.robot_index = robot_index
        self.robot_index_np = np.array(self.robot_index)

        self._step_idx = 0
        self._print_frequency = 100

        self._verbose = verbose

        self._use_force_feedback=use_force_feedback
        self._use_fixed_flights=use_fixed_flights

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
        self._kin_dyn = self.gait_manager.task_interface.model.kd
        self._ti=self.gait_manager.task_interface
        self._prb=self._ti.prb

        self._timelines = self.gait_manager._contact_timelines
        self._timeline_names = self.gait_manager._timeline_names

        self._total_weight = np.atleast_2d(np.array([0, 0, self._kin_dyn.mass() * 9.81])).T # the robot's weight

        # task interfaces from horizon for setting commands to rhc
        self._get_tasks()

    def _get_tasks(self):
        # can be overridden by child
        # cartesian tasks are in LOCAL_WORLD_ALIGNED (frame centered at distal link, oriented as WORLD)
        self.base_lin_velxy = self.gait_manager.task_interface.getTask('base_lin_velxy')
        self.base_lin_velz = self.gait_manager.task_interface.getTask('base_lin_velz')
        self.base_omega = self.gait_manager.task_interface.getTask('base_omega')
        self.base_height = self.gait_manager.task_interface.getTask('base_height')

        self._f_reg_ref=[None]*len(self._timeline_names)
        self._n_forces_per_contact=[1]*len(self._timeline_names)
        i=0
        for timeline in self._timeline_names:
            self._f_reg_ref[i]=[]
            j=0
            forces_on_contact=self._ti.model.cmap[timeline]
            self._n_forces_per_contact[i]=len(forces_on_contact)
            scale=4*self._n_forces_per_contact[i] # just regularize over 1/4 of the  weight
            for force in forces_on_contact:
                self._f_reg_ref[i].append(self._prb.getParameters(name=f"{timeline}_force_reg_f{j}_ref"))
                self._f_reg_ref[i][j].assign(self._total_weight/scale)
                j+=1
            i+=1

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
        contact_names = self._ti.model.cmap.keys()
        if not (self.n_contacts() == len(contact_names)):
            exception = f"N of contacts within problem {len(contact_names)} does not match n of contacts {self.n_contacts()}"
            Journal.log(self.__class__.__name__,
                "run",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
                        
    def step(self, q_base: np.ndarray = None,
        force_norm: np.ndarray = None):

        if self.is_running():
            
            # updates robot refs from shared mem
            self.rob_refs.synch_from_shared_mem()
            self.phase_id.synch_all(read=True, retry=True)
            self.contact_flags.synch_all(read=True, retry=True)
            
            if self._use_fixed_flights:
                self._handle_contact_phases_fixed()
            else:
                self._handle_contact_phases_free()

            # updated internal references with latest available ones
            self._apply_refs_to_tasks(q_base=q_base)
            
            if self._use_force_feedback:
                self._set_force_feedback(force_norm=force_norm)

            self._step_idx +=1
        
        else:
            exception = f"{self.__class__.__name__} is not running"
            Journal.log(self.__class__.__name__,
                "step",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _handle_contact_phases_fixed(self):

        phase_id = self.phase_id.read_retry(row_index=self.robot_index,
                                col_index=0)[0]
        
        contact_flags_refs = self.contact_flags.get_numpy_mirror()[self.robot_index, :]
        target_n_limbs_in_contact=np.sum(contact_flags_refs).item()
        is_contact = contact_flags_refs.flatten().tolist() 
        for i in range(len(is_contact)): # loop through contact timelines
            timeline_name = self._timeline_names[i]
            timeline = self.gait_manager._contact_timelines[timeline_name]

            # set for references depending on expected contacts
            for contact_force_ref in self._f_reg_ref[i]: 
                scale=self._n_forces_per_contact[i]*target_n_limbs_in_contact
                # scale=4 # just regularize
                contact_force_ref.assign(self._total_weight/scale)

            if is_contact[i]==False: # flight phase
                self.gait_manager.add_flight(timeline_name, 
                    ref_height=None)
            else: # contact phase
                if timeline.getEmptyNodes() > 0: # if there's space, always add a stance
                    self.gait_manager.add_stand(timeline_name)

            self._flight_info=self.gait_manager.get_flight_info(timeline_name)
            # self._flight_info=None
            if self._flight_info is not None:
                pos=self._flight_info[0]
                length=len(self._flight_info[1])
                self.flight_info.write_retry(pos, 
                    row_index=self.robot_index,
                    col_index=i)
            else:
                length=0
            self.flight_info.write_retry(length, 
                row_index=self.robot_index,
                col_index=len(is_contact)+i)
                # self._flight_info[2] # n nodes

        for timeline_name in self._timeline_names: # sanity check on the timeline to avoid nasty empty nodes
            timeline = self.gait_manager._contact_timelines[timeline_name]
            if timeline.getEmptyNodes() > 0:
                error = f"Empty nodes detected over the horizon! Make sure to fill the whole horizon with valid phases!!"
                Journal.log(self.__class__.__name__,
                    "step",
                    error,
                    LogType.EXCEP,
                    throw_when_excep = True)

    def _handle_contact_phases_free(self):

        pz_ref=self.rob_refs.contact_pos.get(data_type = "p_z", 
                robot_idxs=self.robot_index_np).reshape(-1, 1)
        
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
                length=len(self._flight_info[1])
                self.flight_info.write_retry(pos, 
                    row_index=self.robot_index,
                    col_index=i)
            else:
                length=0
            self.flight_info.write_retry(length, 
                row_index=self.robot_index,
                col_index=len(is_contact)+i)
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
                                robot_idxs=self.robot_index_np).reshape(-1, 1) # this should also be
            # rotated into the horizontal frame (however, for now only the z componet is used, so it's ok)
            
            root_twist_ref = self.rob_refs.root_state.get(data_type="twist", 
                                robot_idxs=self.robot_index_np).reshape(-1, 1)
            root_twist_ref_h = root_twist_ref.copy() 

            hor2w_frame(root_twist_ref, q_base, root_twist_ref_h)

            self.base_lin_velxy.setRef(root_twist_ref_h[0:2, :])
            self.base_omega.setRef(root_twist_ref_h[3:, :])
            if self.base_lin_velz is not None:
                self.base_lin_velz.setRef(root_twist_ref_h[2:3, :])
            if self.base_height is not None:
                self.base_height.setRef(root_pose) 
        else:
            root_pose = self.rob_refs.root_state.get(data_type = "q_full", 
                                robot_idxs=self.robot_index_np).reshape(-1, 1)
            root_twist_ref = self.rob_refs.root_state.get(data_type="twist", 
                                robot_idxs=self.robot_index_np).reshape(-1, 1)

            self.base_lin_velxy.setRef(root_twist_ref[0:2, :])
            self.base_omega.setRef(root_twist_ref[3:, :])
            if self.base_lin_velz is not None:
                self.base_lin_velz.setRef(root_twist_ref[2:3, :])
            if self.base_height is not None:
                self.base_height.setRef(root_pose)
    
    def _set_force_feedback(self,
            force_norm: np.ndarray = None):
        is_contact=force_norm>1.0

        # for i in range(len(is_contact)):
            
    def reset(self,
            p_ref: np.ndarray = None,
            q_ref: np.ndarray = None):

        if self.is_running():

            # resets shared mem
            contact_flags_current = self.contact_flags.get_numpy_mirror()
            phase_id_current = self.phase_id.get_numpy_mirror()
            contact_flags_current[self.robot_index, :] = np.full((1, self.n_contacts()), dtype=np.bool_, fill_value=True)
            phase_id_current[self.robot_index, :] = -1 # defaults to custom phase id

            contact_pos_current=self.rob_refs.contact_pos.get_numpy_mirror()
            contact_pos_current[self.robot_index, :] = 0.0

            flight_info_current=self.flight_info.get_numpy_mirror()
            flight_info_current[self.robot_index, :] = 0.0

            if p_ref is not None:
                self.rob_refs.root_state.set(data_type="p", data=p_ref, robot_idxs=self.robot_index_np)
            if q_ref is not None:
                self.rob_refs.root_state.set(data_type="q", data=q_ref, robot_idxs=self.robot_index_np)
            
            self.rob_refs.root_state.set(data_type="twist", data=np.zeros((1, 6)), robot_idxs=self.robot_index_np)
                                           
            self.contact_flags.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.contact_flags.n_cols,
                                    read=False)
            self.phase_id.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.phase_id.n_cols,
                                    read=False)
            self.rob_refs.root_state.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.rob_refs.root_state.n_cols,
                                    read=False)

            self.rob_refs.contact_pos.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.rob_refs.contact_pos.n_cols,
                                    read=False)
            
            self.flight_info.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.flight_info.n_cols,
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

