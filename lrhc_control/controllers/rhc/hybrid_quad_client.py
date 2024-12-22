from lrhc_control.controllers.rhc.lrhc_cluster_client import LRhcClusterClient

from lrhc_control.controllers.rhc.horizon_based.hybrid_quad_rhc import HybridQuadRhc
from lrhc_control.utils.hybrid_quad_xrdf_gen import get_xrdf_cmds_horizon
from lrhc_control.utils.sys_utils import PathsGetter

from typing import List, Dict

class HybridQuadrupedClusterClient(LRhcClusterClient):
    
    def _import_aux_libs(self):
        # Import Horizon and related dependencies
        import lrhc_control.controllers.rhc.horizon_based.horizon_imports
        
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
            open_loop: bool = True,
            base_dump_dir: str = "/tmp",
            timeout_ms: int = 60000,
            codegen_override: str = None,
            custom_opts: Dict = {}):
        
        self._open_loop = open_loop

        self._paths = PathsGetter()

        self._codegen_dir_name = namespace

        self._timeout_ms = timeout_ms
        
        super().__init__(namespace = namespace, 
                        urdf_xacro_path = urdf_xacro_path,
                        srdf_xacro_path = srdf_xacro_path,
                        cluster_size=cluster_size,
                        set_affinity = set_affinity,
                        use_mp_fork = use_mp_fork,
                        isolated_cores_only = isolated_cores_only,
                        core_ids_override_list = core_ids_override_list,
                        verbose = verbose,
                        debug = debug,
                        base_dump_dir=base_dump_dir,
                        codegen_override=codegen_override,
                        custom_opts=custom_opts)
    
    def _xrdf_cmds(self):
        parts = self._urdf_path.split('/')
        urdf_descr_root_path = '/'.join(parts[:-2])
        cmds = get_xrdf_cmds_horizon(urdf_descr_root_path=urdf_descr_root_path)
        return cmds

    def _process_codegen_dir(self,idx:int):

        codegen_dir = self.codegen_dir() + f"/{self._codegen_dir_name}Rhc{idx}"
        codegen_dir_ovveride = self.codegen_dir_override()
        if not (codegen_dir_ovveride=="" or \
                codegen_dir_ovveride=="none" or \
                codegen_dir_ovveride=="None" or \
                (codegen_dir_ovveride is None)): # if overrde was provided
            codegen_dir = f"{codegen_dir_ovveride}{idx}"# override
        
        return codegen_dir
        
    def _generate_controller(self,
                        idx: int):
        
        codegen_dir=self._process_codegen_dir(idx=idx)

        controller = HybridQuadRhc(
                urdf_path=self._urdf_path, 
                srdf_path=self._srdf_path,
                config_path = self._paths.CONFIGPATH,
                robot_name=self._namespace,
                codegen_dir=codegen_dir,
                n_nodes=31, 
                dt=0.03,
                max_solver_iter = 1,
                open_loop = self._open_loop,
                verbose = self._verbose, 
                debug = self._debug)
        
        return controller 