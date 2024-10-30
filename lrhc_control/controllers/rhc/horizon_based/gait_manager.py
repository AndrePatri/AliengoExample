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
        self._ref_trjs = {}

        if injection_node is None:
            self._injection_node = round(self.task_interface.prb.getNNodes()/2.0)
        else:
            self._injection_node = injection_node

        self._timeline_names = []
        
        for contact_name, timeline_name in contact_map.items():
            self._contact_timelines[contact_name] = self.phase_manager.getTimelines()[timeline_name]
            self._timeline_names.append(contact_name)
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
        
        if ref_height is not None:
            # set reference 
            self._ref_trjs[timeline_name][2, :]=ref_height
            self._flight_phases[timeline_name].addItemReference(self.task_interface.getTask(f'z_{timeline_name}'),
                self._ref_trjs[timeline_name], 
                nodes=list(range(0, 1)))
    
        timeline = self._contact_timelines[timeline_name]
        timeline.addPhase(self._flight_phases[timeline_name], 
            pos=self._injection_node, absolute_position=True)
    
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