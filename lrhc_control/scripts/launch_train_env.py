from lrhc_control.training_algs.ppo.ppo import PPO
from lrhc_control.training_algs.sac.sac import SAC

from control_cluster_bridge.utilities.shared_data.sim_data import SharedSimInfo

from SharsorIPCpp.PySharsorIPC import VLevel, Journal, LogType
from SharsorIPCpp.PySharsorIPC import StringTensorServer

import os, argparse

from perf_sleep.pyperfsleep import PerfSleep

import importlib.util
import torch

# Function to dynamically import a module from a specific file path
def import_env_module(env_path):
    spec = importlib.util.spec_from_file_location("env_module", env_path)
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module

if __name__ == "__main__":  

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--run_name', type=str, help='Name of training run', default="LRHCTraining")
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--drop_dir', type=str, help='Directory root where all run data will be dumped')
    parser.add_argument('--dump_checkpoints', action=argparse.BooleanOptionalAction, default=True, help='Whether to dump model checkpoints during training')
   
    parser.add_argument('--obs_norm', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable the use of running normalizer in agent')

    parser.add_argument('--use_cer', action=argparse.BooleanOptionalAction, default=False, help='')
    parser.add_argument('--sac', action=argparse.BooleanOptionalAction, default=True, help='')

    parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=False, help='Whether to perform an evaluation run')
    parser.add_argument('--n_eval_timesteps', type=int, help='Toal n. of timesteps to be evaluated', default=None)
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation',default=None)
    parser.add_argument('--mname', type=str, help='Model name',default=None)

    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--seed', type=int, help='seed', default=1)
    
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)

    parser.add_argument('--use_cpu', action=argparse.BooleanOptionalAction, default=False, help='If set, all the training (data included) will be perfomed on CPU')
    parser.add_argument('--db', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable local data logging for the algorithm (reward metrics, etc..)')
    parser.add_argument('--env_db', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable env db data logging on \
                            shared mem (e.g.reward metrics are not available for reading anymore)')
    parser.add_argument('--rmdb', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable remote debug (e.g. data logging on remote servers)')

    parser.add_argument('--override_agent_refs', action=argparse.BooleanOptionalAction, default=False, \
                help='Whether to override automatically generated agent refs (useful for debug)')
    parser.add_argument('--anomaly_detect', action=argparse.BooleanOptionalAction, default=False, \
                help='Whether to override automatically generated agent refs (useful for debug)')

    parser.add_argument('--actor_size', type=int, help='seed', default=256)
    parser.add_argument('--critic_size', type=int, help='seed', default=256)

    parser.add_argument('--env_fname', type=str, default="linvel_env_baseline", help="training env file name (without extension)")
    parser.add_argument('--env_classname', type=str, default="LinVelTrackBaseline", help='training env class name')

    args = parser.parse_args()
    args_dict = vars(args)
    
    if (not args.mpath is None) and (not args.mname is None):
        mpath_full = os.path.join(args.mpath, args.mname)
    else:
        mpath_full=None
    
    env_fname=args.env_fname
    env_classname = args.env_classname
    env_path=""
    env_module=None
    if not args.eval:
        env_path = f"lrhc_control.envs.{env_fname}"
        env_module = importlib.import_module(env_path)
    else:
        env_path=os.path.join(args.mpath, env_fname+".py")
        env_module=import_env_module(env_path)
       
    EnvClass = getattr(env_module, env_classname)
    env = EnvClass(namespace=args.ns,
            verbose=True,
            vlevel=VLevel.V2,
            use_gpu=not args.use_cpu,
            debug=args.env_db,
            override_agent_refs=args.override_agent_refs,
            timeout_ms=args.timeout_ms)
    env_type="training" if not args.eval else "evaluation"
    Journal.log("launch_train_env.py",
            "",
            f"loading {env_type} env {env_classname} from {env_path}",
            LogType.INFO,
            throw_when_excep = True)

    # getting some sim info for debugging
    sim_data = {}
    sim_info_shared = SharedSimInfo(namespace=args.ns,
                is_server=False,
                safe=False)
    sim_info_shared.run()
    sim_info_keys = sim_info_shared.param_keys
    sim_info_data = sim_info_shared.get().flatten()
    for i in range(len(sim_info_keys)):
        sim_data[sim_info_keys[i]] = sim_info_data[i]
    
    custom_args={}
    if args.sac:
        algo = SAC(env=env, 
            debug=args.db, 
            remote_db=args.rmdb,
            seed=args.seed)
        custom_args["use_combined_exp_replay"]=args.use_cer
    else:
        algo = PPO(env=env, 
               debug=args.db, 
               remote_db=args.rmdb,
               seed=args.seed)
    
    custom_args.update(args_dict)
    custom_args.update(sim_data)

    algo.setup(run_name=args.run_name, 
        ns=args.ns,
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args=custom_args,
        comment=args.comment,
        eval=args.eval,
        model_path=mpath_full,
        n_eval_timesteps=args.n_eval_timesteps,
        dump_checkpoints=args.dump_checkpoints,
        norm_obs=args.obs_norm)

    full_drop_dir=algo.drop_dir()
    shared_drop_dir = StringTensorServer(length=1, 
        basename="SharedTrainingDropDir", 
        name_space=args.ns,
        verbose=True, 
        vlevel=VLevel.V2, 
        force_reconnection=True)
    shared_drop_dir.run()
    
    while True:
        if not shared_drop_dir.write_vec([full_drop_dir], 0):
            ns=1000000000
            PerfSleep.thread_sleep(ns)
            continue
        else:
            break
    
    if not args.eval:
        try:
            while not algo.is_done():
                if not algo.learn():
                    algo.done()
        except KeyboardInterrupt:
            algo.done() # in case it's interrupted by user
    else: # eval phase
        with torch.no_grad(): # no need for grad computation
            try:
                while not algo.is_done():
                    if not algo.eval():
                        algo.done()
            except KeyboardInterrupt:
                algo.done() # in case it's interrupted by user