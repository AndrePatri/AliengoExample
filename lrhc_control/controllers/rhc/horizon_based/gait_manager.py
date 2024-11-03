import numpy as np

from horizon.rhc.taskInterface import TaskInterface

from phase_manager import pymanager

from typing import List

class GaitManager:

    def __init__(self, 
            task_interface: TaskInterface, 
            phase_manager: pymanager.PhaseManager, 
            contact_map,
            injection_node: int = None):

        self.task_interface = task_interface
        self.phase_manager = phase_manager

        self._contact_timelines = dict()
        self._flight_phases = {}
        self._contact_phases = {}
        self._ref_trjs = {}

        if injection_node is None:
            self._injection_node = round(self.task_interface.prb.getNNodes()/2.0)
        else:
            self._injection_node = injection_node

        self._timeline_names = []
        
        for contact_name, timeline_name in contact_map.items():
            self._contact_timelines[contact_name] = self.phase_manager.getTimelines()[timeline_name]
            self._timeline_names.append(contact_name)
            self._contact_phases[contact_name]=self._contact_timelines[contact_name].getRegisteredPhase(f'short_{contact_name}_stance')
            flight_short=self._contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}_short')
            flight=self._contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}')
            self._flight_phases[contact_name]=flight_short if flight_short is not None else flight
            self._ref_trjs[contact_name]=np.zeros(shape=[7, self._flight_phases[contact_name].getNNodes()])

    def reset(self):
        # self.phase_manager.clear()
        self.task_interface.reset()

    def add_stand(self, timeline_name):
        timeline = self._contact_timelines[timeline_name]
        timeline.addPhase(timeline.getRegisteredPhase(f'stance_{timeline_name}_short'))
    
    def add_flight(self, timeline_name,
        ref_height:np.array=None):
    
        timeline = self._contact_timelines[timeline_name]
        timeline.addPhase(self._flight_phases[timeline_name], 
            pos=self._injection_node, 
            absolute_position=True)
        
        if ref_height is not None:
            # set reference 
            self._ref_trjs[timeline_name][2, :]=ref_height
            flight_token_idxs=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])
            active_phases=self._contact_timelines[timeline_name].getActivePhases()

            print(flight_token_idxs)
            # flight_token=active_phases[last_flight_token_idxs[0]]
            
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