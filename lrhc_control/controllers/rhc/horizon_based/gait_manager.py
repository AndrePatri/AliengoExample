import numpy as np

from lrhc_control.controllers.rhc.horizon_based.horizon_imports import * 

from phase_manager import pymanager

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

from typing import Dict

class GaitManager:

    def __init__(self, 
            task_interface: TaskInterface, 
            phase_manager: pymanager.PhaseManager, 
            injection_node: int = None,
            keep_yaw_vert: bool = False,
            yaw_vertical_weight: float = None,
            phase_force_reg: float = None,
            flight_duration: float = 15,
            step_height: float = 0.1,
            dh: float = 0.0,
            custom_opts: Dict = {}):

        self._custom_opts=custom_opts

        self.task_interface = task_interface
        self._phase_manager = phase_manager
        self._model=self.task_interface.model
        self._q0=self._model.q0
        self._kin_dyn=self.task_interface.model.kd

        self._keep_yaw_vert=keep_yaw_vert
        self._yaw_vertical_weight=yaw_vertical_weight
        self._phase_force_reg=phase_force_reg
        self._total_weight = np.atleast_2d(np.array([0, 0, self._kin_dyn.mass() * 9.81])).T 
        
        self._flight_duration=flight_duration
        self._step_height=step_height
        self._dh=dh
        self._f_reg_ref={}

        self._contact_timelines = dict()
        self.timeline_names=[]

        self._flight_phases = {}
        self._contact_phases = {}
        self._fk_contacts = {}
        self._fkd_contacts = {}
        # reference traj
        self._tg = trajectoryGenerator.TrajectoryGenerator()
        self._traj_der= [None, 0, 0]
        self._traj_second_der=[None, 0, 0]
        self._ref_trjs = {}
        self._ref_vtrjs = {}

        if injection_node is None:
            self._injection_node = round(self.task_interface.prb.getNNodes()/2.0)
        else:
            self._injection_node = injection_node

        self.task_interfacemeline_names = []
        
        if self._custom_opts["fixed_flights"]:
            self._init_contact_timelines()  
        else:
            self._init_contact_timelines()
        
        self._reset_contact_timelines()

    def _init_contact_timelines(self):
        short_stance_duration=1
        flight_phase_short_duration=1
        for contact in self._model.cmap.keys():
            self._fk_contacts[contact]=self._kin_dyn.fk(contact)
            self._fkd_contacts[contact]=self._kin_dyn.frameVelocity(contact, self._model.kd_frame)
            self.timeline_names.append(contact)
            self._contact_timelines[contact]=self._phase_manager.createTimeline(f'{contact}_timeline')
            self.task_interfacemeline_names.append(contact)
            # stances
            self._contact_phases[contact] = self._contact_timelines[contact].createPhase(short_stance_duration, 
                                    f'stance_{contact}_short')
            
            if self.task_interface.getTask(f'{contact}') is not None:
                self._contact_phases[contact].addItem(self.task_interface.getTask(f'{contact}'))
            else:
                Journal.log(self.__class__.__name__,
                    "_init_contact_timelines",
                    f"contact task {contact} not found",
                    LogType.EXCEP,
                    throw_when_excep=True)
            i=0
            self._f_reg_ref[contact]=[]
            for force in self._model.cmap[contact]:
                f_ref=self.task_interface.prb.createParameter(name=f"{contact}_force_reg_f{i}_ref",
                    dim=3)
                force_reg=self.task_interface.prb.createResidual(f'{contact}_force_reg_f{i}', self._phase_force_reg * (force - f_ref), 
                    nodes=[])
                self._f_reg_ref[contact].append(f_ref)
                self.set_f_reg(contact_name=contact, scale=4)                
                self._contact_phases[contact].addCost(force_reg, nodes=list(range(0, short_stance_duration)))
                i+=1
            
            # flights
            self._flight_phases[contact]=self._contact_timelines[contact].createPhase(flight_phase_short_duration, 
                                    f'flight_{contact}_short')
            
            # ref pos traj
            
            self._ref_trjs[contact]=np.zeros(shape=[7, self.task_interface.prb.getNNodes()]) # allocate traj
            # of max length eual to number of nodes
            init_z_foot = self._fk_contacts[contact](q=self._q0)['ee_pos'].elements()[2]
            self._ref_trjs[contact][2, :] = np.atleast_2d(init_z_foot)
            if self.task_interface.getTask(f'z_{contact}') is not None:
                self._flight_phases[contact].addItemReference(self.task_interface.getTask(f'z_{contact}'), 
                        self._ref_trjs[contact][2, 0:1], 
                        nodes=list(range(0, flight_phase_short_duration)))
            else:
                Journal.log(self.__class__.__name__,
                    "_init_contact_timelines",
                    f"contact pos traj tracking task z_{contact} not found -> it won't be used",
                    LogType.WARN,
                    throw_when_excep=True)
            
            # ref vel traj
            self._ref_vtrjs[contact]=np.zeros(shape=[7, self.task_interface.prb.getNNodes()]) # allocate traj
            # of max length eual to number of nodes
            if self.task_interface.getTask(f'vz_{contact}') is not None:
                self._ref_vtrjs[contact][2, :] = np.atleast_2d(0)
                self._flight_phases[contact].addItemReference(self.task_interface.getTask(f'vz_{contact}'), 
                    self._ref_vtrjs[contact], 
                    nodes=list(range(0, flight_phase_short_duration)))
            else:
                Journal.log(self.__class__.__name__,
                    "_init_contact_timelines",
                    f"contact vel traj tracking task vz_{contact} not found-> it won't be used",
                    LogType.WARN,
                    throw_when_excep=True)
            
            # ee_vel=self._fkd_contacts[contact](q=self._model.q, 
            #             qdot=self._model.v)['ee_vel_linear']
            # cstr = self.task_interface.prb.createConstraint(f'{contact}_vert', ee_vel[0:2], [])
            # self._flight_phases[contact].addConstraint(cstr, nodes=[0, flight_phase_short_duration-1])
            if self._keep_yaw_vert:
                # keep ankle vertical
                c_ori = self._model.kd.fk(contact)(q=self._model.q)['ee_rot'][2, :]
                cost_ori = self.task_interface.prb.createResidual(f'{contact}_ori', self._yaw_vertical_weight * (c_ori.T - np.array([0, 0, 1])))
                # flight_phase.addCost(cost_ori, nodes=list(range(0, flight_duration+post_landing_stance)))

    def _reset_contact_timelines(self):
        for contact in self._model.cmap.keys():
            # fill timeline with stances
            contact_timeline=self._contact_timelines[contact]
            contact_timeline.clear() # remove phases
            short_stance_phase = contact_timeline.getRegisteredPhase(f'stance_{contact}_short')
            while contact_timeline.getEmptyNodes() > 0:
                contact_timeline.addPhase(short_stance_phase)

    def reset(self):
        # self.phase_manager.clear()
        self.task_interface.reset()
        self._reset_contact_timelines()

    def set_f_reg(self, 
        contact_name,
        scale: float = 4.0):
        f_refs=self._f_reg_ref[contact_name]
        for force in f_refs:
            ref=self._total_weight/(scale*len(f_refs))
            force.assign(ref)

    def add_stand(self, contact_name):
        # always add stand at the end of the horizon
        timeline = self._contact_timelines[contact_name]
        if timeline.getEmptyNodes() > 0:
            timeline.addPhase(timeline.getRegisteredPhase(f'stance_{contact_name}_short'))
    
    def add_flight(self, contact_name,
        robot_q: np.ndarray):

        if self._flight_duration>1:
            timeline = self._contact_timelines[contact_name]
        
            flights_on_horizon=self._contact_timelines[contact_name].getPhaseIdx(self._flight_phases[contact_name]) 
            
            last_flight_idx=self._injection_node-1 # default to make things work
            if not len(flights_on_horizon)==0:
                last_flight_idx=flights_on_horizon[-1]
            if last_flight_idx<self._injection_node: # do not allow overlapping
                # compute trajectory online
                starting_pos=self._fk_contacts[contact_name](q=robot_q)['ee_pos'].elements()[2]
                self._ref_trjs[contact_name][2, 0:self._flight_duration]=np.atleast_2d(self._tg.from_derivatives(self._flight_duration, 
                                                                        starting_pos, 
                                                                        starting_pos+self._dh, 
                                                                        self._step_height,
                    derivatives=self._traj_der,
                    second_der=self._traj_second_der))
                
                for i in range(self._flight_duration):
                    res, phase_token=timeline.addPhase(self._flight_phases[contact_name], 
                        pos=self._injection_node+i, 
                        absolute_position=True)
                    phase_token.setItemReference(f'z_{contact_name}',
                        self._ref_trjs[contact_name][:, i])

        else:
            Journal.log(self.__class__.__name__,
                "add_flight",
                f"Unit duration not yet supported",
                LogType.EXCEP,
                throw_when_excep=True)

        # if ref_height is not None:
        #     # set reference 
        #     self._ref_trjs[timeline_name][2, :]=ref_height
        #     flight_token_idxs=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])
        #     active_phases=self._contact_timelines[timeline_name].getActivePhases()

        #     print(flight_token_idxs)
        #     # flight_token=active_phases[last_flight_token_idxs[0]]
    
    def update(self):
        self._phase_manager.update()
        
    def get_flight_info(self, timeline_name):
        # phase indexes over timeline
        phase_idxs=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])
        # all active phases on timeline
        active_phases=self._contact_timelines[timeline_name].getActivePhases()
        if len(phase_idxs)==0:
            return None
        else:
            phase_idx=phase_idxs[0] # just get info for closest phase on the horizon
            active_nodes=active_phases[phase_idx].getActiveNodes()
            start_pos=active_phases[phase_idx].getPosition()
            n_nodes=active_phases[phase_idx].getNNodes()
            return (start_pos, active_nodes, n_nodes)
    
    def set_ref_pos(self,
        timeline_name:str,
        ref_height: np.array = None,
        threshold: float = 0.05):
        
        if ref_height is not None:
            self._ref_trjs[timeline_name][2, :]=ref_height
            if ref_height>threshold:
                self.add_flight(timeline_name=timeline_name)
                this_flight_token_idx=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])[-1]
                active_phases=self._contact_timelines[timeline_name].getActivePhases()
                active_phases[this_flight_token_idx].setItemReference(f'z_{timeline_name}',
                    self._ref_trjs[timeline_name])
            else:
                self.add_stand(timeline_name=timeline_name)

    def set_force_feedback(self,
        timeline_name: str,
        force_norm: float):
        
        flight_tokens=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])
        contact_tokens=self._contact_phases[timeline_name].getPhaseIdx(self._contact_phases[timeline_name])
        if not len(flight_tokens)==0:
            first_flight=flight_tokens[0]
            first_flight
    
    def check_horizon_full(self,
        timeline_name):
        timeline = self._contact_timelines[timeline_name]
        if timeline.getEmptyNodes() > 0:
            error = f"Empty nodes detected over the horizon for timeline {timeline}! Make sure to fill the whole horizon with valid phases!!"
            Journal.log(self.__class__.__name__,
                "check_horizon_full",
                error,
                LogType.EXCEP,
                throw_when_excep = True)