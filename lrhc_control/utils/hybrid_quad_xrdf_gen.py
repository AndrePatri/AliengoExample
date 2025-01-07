from typing import List
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType

def get_xrdf_cmds(urdf_descr_root_path: str = None):

        if "kyon" in urdf_descr_root_path:
                return get_xrdf_cmds_kyon(urdf_descr_root_path=urdf_descr_root_path)
        elif "centauro" in urdf_descr_root_path: 
                return get_xrdf_cmds_centauro(urdf_descr_root_path=urdf_descr_root_path)
        elif "b2w" in urdf_descr_root_path: 
                return get_xrdf_cmds_b2w(urdf_descr_root_path=urdf_descr_root_path)
        else:
                exception=f"xrdf cmd getter for robot {urdf_descr_root_path} not supported! Please modify this file to add your own."
                Journal.log("hybrid_quad_xrdf_gen.py",
                        "get_xrdf_cmds",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = False)
                return None
           
def get_xrdf_cmds_horizon(urdf_descr_root_path : str = None):

        parts = urdf_descr_root_path.split('/')
        urdf_descr_root_path = '/'.join(parts[:3])

        if "kyon" in urdf_descr_root_path:
                return get_xrdf_cmds_horizon_kyon(urdf_descr_root_path=urdf_descr_root_path)
        elif "centauro" in urdf_descr_root_path: 
                return get_xrdf_cmds_horizon_centauro(urdf_descr_root_path=urdf_descr_root_path)
        elif "b2w" in urdf_descr_root_path: 
                return get_xrdf_cmds_horizon_b2w(urdf_descr_root_path=urdf_descr_root_path)
        else:
                exception=f"xrdf cmd getter for robot {urdf_descr_root_path} not supported! Please modify this file to add your own."
                Journal.log("hybrid_quad_xrdf_gen.py",
                        "get_xrdf_cmds_horizon",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = False)
                return None     

def get_xrdf_cmds_centauro(urdf_descr_root_path: str = None):

        cmds_aux = []
        
        xrdf_cmd_vals = [True, True, True, False, False, False]
        legs = "true" if xrdf_cmd_vals[0] else "false"
        big_wheel = "true" if xrdf_cmd_vals[1] else "false"
        upper_body ="true" if xrdf_cmd_vals[2] else "false"
        velodyne = "true" if xrdf_cmd_vals[3] else "false"
        realsense = "true" if xrdf_cmd_vals[4] else "false"
        floating_joint = "true" if xrdf_cmd_vals[5] else "false"

        cmds_aux.append("legs:=" + legs)
        cmds_aux.append("big_wheel:=" + big_wheel)
        cmds_aux.append("upper_body:=" + upper_body)
        cmds_aux.append("velodyne:=" + velodyne)
        cmds_aux.append("realsense:=" + realsense)
        cmds_aux.append("floating_joint:=" + floating_joint)
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes             \       
        cmds_aux.append("root:=" + urdf_descr_root_path)

        return cmds_aux

def get_xrdf_cmds_horizon_centauro(urdf_descr_root_path: str = None):

        cmds = []
        
        xrdf_cmd_vals = [True, True, True, False, False, True] # horizon needs 
        # the floating base
        legs = "true" if xrdf_cmd_vals[0] else "false"
        big_wheel = "true" if xrdf_cmd_vals[1] else "false"
        upper_body ="true" if xrdf_cmd_vals[2] else "false"
        velodyne = "true" if xrdf_cmd_vals[3] else "false"
        realsense = "true" if xrdf_cmd_vals[4] else "false"
        floating_joint = "true" if xrdf_cmd_vals[5] else "false"
                
        cmds.append("legs:=" + legs)
        cmds.append("big_wheel:=" + big_wheel)
        cmds.append("upper_body:=" + upper_body)
        cmds.append("velodyne:=" + velodyne)
        cmds.append("realsense:=" + realsense)
        cmds.append("floating_joint:=" + floating_joint)
        cmds.append("use_abs_mesh_paths:=true") # use absolute paths for meshes             \       
        
        if urdf_descr_root_path is not None:
                cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_kyon(urdf_descr_root_path: str = None):

        cmds_aux = []
        
        xrdf_cmd_vals = [False, False, False, False, False, False]

        wheels = "true" if xrdf_cmd_vals[0] else "false"
        upper_body = "true" if xrdf_cmd_vals[1] else "false"
        gripper = "true" if xrdf_cmd_vals[2] else "false"
        sensors = "true" if xrdf_cmd_vals[3] else "false"
        floating_joint = "true" if xrdf_cmd_vals[4] else "false"
        payload = "true" if xrdf_cmd_vals[5] else "false"

        cmds_aux.append("wheels:=" + wheels)
        cmds_aux.append("upper_body:=" + upper_body)
        cmds_aux.append("dagana:=" + gripper)
        cmds_aux.append("sensors:=" + sensors)
        cmds_aux.append("floating_joint:=" + floating_joint)
        cmds_aux.append("payload:=" + payload)
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes
        
        cmds_aux.append("root:=" + urdf_descr_root_path)

        return cmds_aux

def get_xrdf_cmds_horizon_kyon(urdf_descr_root_path: str = None):

        cmds = []
        
        xrdf_cmd_vals = [False, False, False, False, True, False] # horizon needs 
        # the floating base

        wheels = "true" if xrdf_cmd_vals[0] else "false"
        upper_body = "true" if xrdf_cmd_vals[1] else "false"
        gripper = "true" if xrdf_cmd_vals[2] else "false"
        sensors = "true" if xrdf_cmd_vals[3] else "false"
        floating_joint = "true" if xrdf_cmd_vals[4] else "false"
        payload = "true" if xrdf_cmd_vals[5] else "false"
                
        cmds.append("wheels:=" + wheels)
        cmds.append("upper_body:=" + upper_body)
        cmds.append("dagana:=" + gripper)
        cmds.append("sensors:=" + sensors)
        cmds.append("floating_joint:=" + floating_joint)
        cmds.append("payload:=" + payload)
        
        if urdf_descr_root_path is not None:
                cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_b2w(urdf_descr_root_path: str = None):

        cmds_aux = []
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes
        cmds_aux.append("root:=" + urdf_descr_root_path)

        return cmds_aux

def get_xrdf_cmds_horizon_b2w(urdf_descr_root_path: str = None):

        cmds = []
 
        if urdf_descr_root_path is not None:
                cmds.append("root:=" + urdf_descr_root_path)

        return cmds