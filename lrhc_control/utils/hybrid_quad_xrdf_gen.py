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

        cmds = []
        
        cmds.append("legs:=true")
        cmds.append("big_wheel:=true")
        cmds.append("upper_body:=true")
        cmds.append("battery:=true")

        cmds.append("velodyne:=false")
        cmds.append("realsense:=false")
        cmds.append("floating_joint:=false")
        cmds.append("use_abs_mesh_paths:=true") # use absolute paths for meshes    
        
        cmds.append("end_effector_left:=ball")
        cmds.append("end_effector_right:=ball")
        
        cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_horizon_centauro(urdf_descr_root_path: str = None):

        cmds = []
        
        cmds.append("legs:=true")
        cmds.append("big_wheel:=true")
        cmds.append("upper_body:=true")
        cmds.append("battery:=true")

        cmds.append("velodyne:=false")
        cmds.append("realsense:=false")
        cmds.append("floating_joint:=true")
        cmds.append("use_abs_mesh_paths:=true") # use absolute paths for meshes             \       
        
        cmds.append("end_effector_left:=ball")
        cmds.append("end_effector_right:=ball")
        
        if urdf_descr_root_path is not None:
                cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_kyon(urdf_descr_root_path: str = None):

        cmds = []

        cmds.append("wheels:=false")
        cmds.append("upper_body:=false")
        cmds.append("dagana:=false")
        cmds.append("sensors:=false")
        cmds.append("floating_joint:=false")
        cmds.append("payload:=false")
        cmds.append("use_abs_mesh_paths:=true") # use absolute paths for meshes
        
        cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_horizon_kyon(urdf_descr_root_path: str = None):

        cmds = []
        
        cmds.append("wheels:=false")
        cmds.append("upper_body:=false")
        cmds.append("dagana:=false")
        cmds.append("sensors:=false")
        cmds.append("floating_joint:=true")
        cmds.append("payload:=false")
        
        if urdf_descr_root_path is not None:
                cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_b2w(urdf_descr_root_path: str = None):

        cmds = []
        cmds.append("use_abs_mesh_paths:=true") # use absolute paths for meshes
        cmds.append("root:=" + urdf_descr_root_path)

        return cmds

def get_xrdf_cmds_horizon_b2w(urdf_descr_root_path: str = None):

        cmds = []
        
        cmds.append("floating_joint:=true")
        if urdf_descr_root_path is not None:
                cmds.append("root:=" + urdf_descr_root_path)

        return cmds