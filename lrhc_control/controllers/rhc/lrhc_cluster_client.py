from control_cluster_bridge.cluster_client.control_cluster_client import ControlClusterClient
from lrhc_control.utils.xrdf_gen import generate_srdf, generate_urdf
from lrhc_control.utils.hybrid_quad_xrdf_gen import get_xrdf_cmds
from lrhc_control.utils.custom_arg_parsing import extract_custom_xacro_args, merge_xacro_cmds

from EigenIPC.PyEigenIPC import Journal, LogType

from typing import List, Dict

import os 

from abc import abstractmethod

class LRhcClusterClient(ControlClusterClient):
    
    def _import_aux_libs(self):
        super()._import_aux_libs()
        
    def __init__(self, 
            namespace: str, 
            urdf_xacro_path: str,
            srdf_xacro_path: str,
            cluster_size: int,
            set_affinity: bool = False,
            use_mp_fork: bool = False,
            isolated_cores_only: bool = False,
            core_ids_override_list: List[int] = None,
            verbose: bool = False,
            debug: bool = False,
            codegen_base_dirname: str = "CodeGen",
            base_dump_dir: str = "/tmp",
            codegen_override: str = None,
            custom_opts: Dict = {}):
               
        self._base_dump_dir = base_dump_dir
    
        self._temp_path = base_dump_dir + "/" + f"{self.__class__.__name__}" + f"_{namespace}"
        
        self._codegen_base_dirname = codegen_base_dirname
        self._codegen_basedir = self._temp_path + "/" + self._codegen_base_dirname

        self._codegen_override = codegen_override # can be used to manually override
        # the default codegen dir 

        if not os.path.exists(self._temp_path):
            os.makedirs(self._temp_path)
        if not os.path.exists(self._codegen_basedir):
            os.makedirs(self._codegen_basedir)

        self._urdf_xacro_path = urdf_xacro_path
        self._srdf_xacro_path = srdf_xacro_path
        self._urdf_path=""
        self._srdf_path=""
        
        super().__init__(namespace = namespace, 
                        cluster_size=cluster_size,
                        isolated_cores_only = isolated_cores_only,
                        set_affinity = set_affinity,
                        use_mp_fork = use_mp_fork,
                        core_ids_override_list = core_ids_override_list,
                        verbose = verbose,
                        debug = debug,
                        custom_opts=custom_opts)
        self._generate_srdf(namespace=namespace)

        self._generate_urdf(namespace=namespace)

    
    def codegen_dir(self):

        return self._codegen_basedir
    
    def codegen_dir_override(self):

        return self._codegen_override
    
    def _generate_srdf(self,namespace:str):
        
        custom_xacro_args=extract_custom_xacro_args(self._custom_opts)
        cmds=merge_xacro_cmds(prev_cmds=self._xrdf_cmds(),
            new_cmds=custom_xacro_args)
    
        self._urdf_path=generate_urdf(robot_name=namespace,
            xacro_path=self._urdf_xacro_path,
            dump_path=self._temp_path,
            xrdf_cmds=cmds)
    
    def _generate_urdf(self,namespace:str):
        
        custom_xacro_args=extract_custom_xacro_args(self._custom_opts)
        cmds=merge_xacro_cmds(prev_cmds=self._xrdf_cmds(),
            new_cmds=custom_xacro_args)

        self._srdf_path=generate_srdf(robot_name=namespace,
            xacro_path=self._srdf_xacro_path,
            dump_path=self._temp_path,
            xrdf_cmds=cmds)
            
    @abstractmethod
    def _xrdf_cmds(self):
        
        # to be implemented by parent class (
        # for an example have a look at utils/centauro_xrdf_gen.py)

        pass    
