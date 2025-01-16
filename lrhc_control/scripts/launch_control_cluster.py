import os
import argparse
import multiprocessing as mp
import importlib.util
import inspect

from lrhc_control.utils.custom_arg_parsing import generate_custom_arg_dict

from EigenIPC.PyEigenIPC import Journal, LogType

this_script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

# Function to dynamically import a module from a specific file path
def import_env_module(env_path):
    spec = importlib.util.spec_from_file_location("env_module", env_path)
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module

if __name__ == "__main__":  

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--urdf_path', type=str, help='Robot description package path for URDF ')
    parser.add_argument('--srdf_path', type=str, help='Robot description package path for SRDF ')
    parser.add_argument('--size', type=int, help='cluster size', default=1)

    # Replacing argparse.BooleanOptionalAction with 'store_true' and 'store_false' for compatibility with Python 3.8
    parser.add_argument('--cloop',action='store_true', help='whether to use RHC controllers in closed loop mode')
    parser.add_argument('--cluster_dt', type=float, default=0.05, help='dt used by MPC controllers for discretization')
    parser.add_argument('--n_nodes', type=int, default=31, help='n nodes used by MPC controllers')

    parser.add_argument('--verbose',action='store_true', help='run in verbose mode')

    parser.add_argument('--enable_debug',action='store_true', help='enable debug mode for cluster client and all controllers')

    parser.add_argument('--dmpdir', type=str, help='directory where data is dumped', default="/root/aux_data")

    parser.add_argument('--no_mp_fork',action='store_true', help='whether to multiprocess with forkserver context')

    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run', default="")
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--codegen_override_dir', type=str, help='Path to base dir where codegen is to be loaded', default="")

    parser.add_argument('--custom_args_names', nargs='+', default=None,
                            help='list of custom arguments names')
    parser.add_argument('--custom_args_vals', nargs='+', default=None,
                            help='list of custom arguments values')
    parser.add_argument('--custom_args_dtype', nargs='+', default=None,
                            help='list of custom arguments data types')
    
    parser.add_argument('--cluster_client_fname', type=str, 
        default="lrhc_control.controllers.rhc.hybrid_quad_client",
        help="cluster client file import pattern (without extension)")

    args = parser.parse_args()
    
    # Ensure custom_args_names and custom_args_vals have the same length
    custom_opts = generate_custom_arg_dict(args=args)
    custom_opts.update({"cloop": args.cloop, 
        "cluster_dt": args.cluster_dt, 
        "n_nodes": args.n_nodes, 
        "codegen_override_dir": args.codegen_override_dir})
    if args.no_mp_fork: # this needs to be in the main
        mp.set_start_method('spawn')
    else:
        # mp.set_start_method('forkserver')
        mp.set_start_method('fork')
        
    cluster_module=importlib.import_module(args.cluster_client_fname)
    # Get all classes defined in the module
    classes_in_module = [name for name, obj in inspect.getmembers(cluster_module, inspect.isclass) 
                        if obj.__module__ == cluster_module.__name__]
    if len(classes_in_module) == 1:
        cluster_classname=classes_in_module[0]
        ClusterClient = getattr(cluster_module, cluster_classname)
        cluster_client = ClusterClient(namespace=args.ns, 
            cluster_size=args.size,
            urdf_xacro_path=args.urdf_path,
            srdf_xacro_path=args.srdf_path,
            open_loop=not args.cloop,
            use_mp_fork = not args.no_mp_fork,
            verbose=args.verbose,
            debug=args.enable_debug,
            base_dump_dir=args.dmpdir,
            timeout_ms=args.timeout_ms,
            custom_opts=custom_opts,
            codegen_override=args.codegen_override_dir)
        cluster_client.run()
        
    else:
        class_list_str = ", ".join(classes_in_module)
        Journal.log("launch_control_cluster.py",
            "",
            f"Found more than one class in cluster client file {args.cluster_client_fname}. Found: {class_list_str}",
            LogType.EXCEP,
            throw_when_excep = False)
        exit()

    # control_cluster_client = 
    # control_cluster_client.run() # spawns the controllers on separate processes (blocking)

