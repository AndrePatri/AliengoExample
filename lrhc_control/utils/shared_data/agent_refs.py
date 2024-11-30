from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase
from control_cluster_bridge.utilities.shared_data.rhc_data import FullRobState

from EigenIPC.PyEigenIPC import VLevel

import numpy as np

from typing import List

class AgentRefs(SharedDataBase):
    
    class RobotRef(FullRobState):

        def __init__(self,
                namespace: str,
                is_server: bool,
                basename: str = "",
                n_robots: int = None,
                n_jnts: int = None,
                n_contacts: int = 1,
                jnt_names: List[str] = None,
                contact_names: List[str] = None,
                q_remapping: List[int] = None,
                with_gpu_mirror: bool = True,
                with_torch_view: bool = True,
                force_reconnection: bool = False,
                safe: bool = True,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                fill_value=np.nan, # if ref is not used
                ):

            basename = basename + "AgentRobotRef"

            super().__init__(namespace=namespace,
                basename=basename,
                is_server=is_server,
                n_robots=n_robots,
                n_jnts=n_jnts,
                n_contacts=n_contacts,
                jnt_names=jnt_names,
                contact_names=contact_names,
                q_remapping=q_remapping,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                force_reconnection=force_reconnection,
                safe=safe,
                verbose=verbose,
                vlevel=vlevel,
                fill_value=fill_value)
    
    def __init__(self,
                namespace: str,
                is_server: bool,
                n_robots: int = None,
                n_jnts: int = None,
                n_contacts: int = 1,
                jnt_names: List[str] = None,
                contact_names: List[str] = None,
                q_remapping: List[int] = None,
                with_gpu_mirror: bool = True,
                with_torch_view: bool = True,
                force_reconnection: bool = False,
                safe: bool = False,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                fill_value=np.nan):
        
        self.basename = "AgentRefs"

        self.is_server = is_server

        self.n_robots = n_robots

        self.namespace = namespace

        self.verbose = verbose

        self.vlevel = vlevel

        self.force_reconnection = force_reconnection

        self.safe = safe

        self.rob_refs = self.RobotRef(namespace=namespace,
                                basename=self.basename,
                                is_server=is_server,
                                n_robots=n_robots,
                                n_jnts=n_jnts,
                                n_contacts=n_contacts,
                                jnt_names=jnt_names,
                                contact_names=contact_names,
                                q_remapping=q_remapping,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view,
                                force_reconnection=force_reconnection,
                                safe=safe,
                                verbose=verbose,
                                vlevel=vlevel,
                                fill_value=fill_value)
        
        self._is_runnning = False

    def __del__(self):

        self.close()

    def get_shared_mem(self):

        return self.rob_refs.get_shared_mem()
    
    def is_running(self):
    
        return self._is_runnning
    
    def run(self):

        self.rob_refs.run()

        self.n_contacts = self.rob_refs.n_contacts()
        
        self.n_robots = self.rob_refs.n_robots()    

        self._is_runnning = True

    def close(self):
        
        if self.is_running():
            
            self.rob_refs.close()

            self._is_runnning = False