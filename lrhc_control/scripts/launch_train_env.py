from lrhc_control.utils.determinism import deterministic_run

from control_cluster_bridge.utilities.shared_data.sim_data import SharedEnvInfo
from control_cluster_bridge.utilities.shared_data.cluster_data import SharedClusterInfo

from EigenIPC.PyEigenIPC import VLevel, Journal, LogType
from EigenIPC.PyEigenIPC import StringTensorServer

import os, argparse

from perf_sleep.pyperfsleep import PerfSleep

import importlib.util
import torch
import signal

algo = None  # global to make it accessible by signal handler
exit_request=False

def handle_sigint(signum, frame):
    global exit_request
    Journal.log("launch_train_env.py",
        "",
        f"Received sigint. Will stop training.",
        LogType.WARN)
    exit_request=True
    
# Function to dynamically import a module from a specific file path
def import_env_module(env_path):
    spec = importlib.util.spec_from_file_location("env_module", env_path)
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module

if __name__ == "__main__":  

    signal.signal(signal.SIGINT, handle_sigint)

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")

    parser.add_argument('--run_name', type=str, default=None, help='Name of training run')
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--timeout_ms', type=int, help='Connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--drop_dir', type=str, help='Directory root where all run data will be dumped')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run', default="")
    parser.add_argument('--seed', type=int, help='Seed', default=1)
    parser.add_argument('--use_cpu',action='store_true', help='If set, all the training (data included) will be performed on CPU')

    parser.add_argument('--db',action='store_true', help='Whether to enable local data logging for the algorithm (reward metrics, etc.)')
    parser.add_argument('--env_db',action='store_true', help='Whether to enable env db data logging on shared mem (e.g. reward metrics are not available for reading anymore)')
    parser.add_argument('--full_env_db',action='store_true', help='Whether to enable detailed episodic data storage (data over single transitions)')
    parser.add_argument('--rmdb',action='store_true', help='Whether to enable remote debug (e.g. data logging on remote servers)')

    parser.add_argument('--tot_tsteps', type=int, help='Total number of timesteps to be collected', default=int(30e6))
    parser.add_argument('--action_repeat', type=int, help='Frame skipping (1-> no skip)', default=1)
    parser.add_argument('--obs_norm',action='store_true', help='Whether to enable the use of running normalizer in agent')
    parser.add_argument('--obs_rescale',action='store_true', help='Whether to rescale observation depending on their expected range')
    parser.add_argument('--add_weight_norm',action='store_true', help='Whether to add weight normalization to agent llayers')
    parser.add_argument('--act_rescale_critic',action='store_true', help='Whether to rescale actions provided to critic (if SAC) to be in range [-1, 1]')

    parser.add_argument('--sac',action='store_true', help='Use SAC, otherwise PPO, unless dummy is set')
    parser.add_argument('--dummy',action='store_true', help='Use dummy agent')

    parser.add_argument('--dump_checkpoints',action='store_true', help='Whether to dump model checkpoints during training')

    parser.add_argument('--demo_envs_perc', type=float, help='[0, 1]', default=0.0)
    parser.add_argument('--demo_stop_thresh', type=float, default=None, 
        help='Performance hreshold above which demonstration envs should be deactivated.')
    
    parser.add_argument('--expl_envs_perc', type=float, help='[0, 1]', default=0)
    
    parser.add_argument('--use_rnd',action='store_true', help='Whether to use RND for exploration')

    parser.add_argument('--eval',action='store_true', help='Whether to perform an evaluation run')
    parser.add_argument('--n_eval_timesteps', type=int, help='Total number of timesteps to be evaluated', default=int(1e6))
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation', default=None)
    parser.add_argument('--mname', type=str, help='Model name', default=None)
    parser.add_argument('--det_eval',action='store_true', help='Whether to perform a deterministic eval (only action mean is used). Only valid if --eval.')
    parser.add_argument('--allow_expl_during_eval',action='store_true', help='Whether to allow expl envs during evaluation (useful to tune exploration)')
    parser.add_argument('--override_env',action='store_true', help='Whether to override env when running evaluation')
    
    parser.add_argument('--anomaly_detect',action='store_true', help='Whether to enable anomaly detection (useful for debug)')

    parser.add_argument('--compression_ratio', type=float,
        help='If e.g. 0.8, the fist layer will be of dimension [input_features_size x (input_features_size*compression_ratio)]', default=-1.0)
    parser.add_argument('--actor_lwidth', type=int, help='Actor network layer width', default=128)
    parser.add_argument('--critic_lwidth', type=int, help='Critic network layer width', default=256)
    parser.add_argument('--actor_n_hlayers', type=int, help='Actor network size', default=3)
    parser.add_argument('--critic_n_hlayers', type=int, help='Critic network size', default=4)

    parser.add_argument('--env_fname', type=str, default="linvel_env_baseline", help='Training env file name (without extension)')
    parser.add_argument('--env_classname', type=str, default="LinVelTrackBaseline", help='Training env class name')
    parser.add_argument('--override_agent_actions',action='store_true', help='Whether to override agent actions with custom ones from shared mem (useful for db)')
    parser.add_argument('--override_agent_refs',action='store_true', help='Whether to override automatically generated agent refs (useful for debug)')
    
    args = parser.parse_args()
    args_dict = vars(args)

    deterministic_run(seed=args.seed, torch_det_algos=False)

    anomaly_detect=False
    if args.anomaly_detect:
        torch.autograd.set_detect_anomaly(True)

    if (not args.mpath is None) and (not args.mname is None):
        mpath_full = os.path.join(args.mpath, args.mname)
    else:
        mpath_full=None
    
    env_fname=args.env_fname
    env_classname = args.env_classname
    env_path=""
    env_module=None
    if (not args.eval) or (args.override_env):
        env_path = f"lrhc_control.envs.{env_fname}"
        env_module = importlib.import_module(env_path)
    else:
        if args.mpath is None:
            Journal.log("launch_train_env.py",
                "",
                f"no mpath provided! Cannot load env. Either provide a mpath or run with --override_env",
                LogType.EXCEP,
                throw_when_excep = True)
    
        env_path=os.path.join(args.mpath, env_fname+".py")
        env_module=import_env_module(env_path)
       
    EnvClass = getattr(env_module, env_classname)
    env = EnvClass(namespace=args.ns,
            verbose=True,
            vlevel=VLevel.V2,
            use_gpu=not args.use_cpu,
            debug=args.env_db,
            override_agent_refs=args.override_agent_refs,
            timeout_ms=args.timeout_ms,
            env_opts=args_dict)
    if not env.is_ready(): # something went wrong
        exit()

    env_type="training" if not args.eval else "evaluation"
    Journal.log("launch_train_env.py",
        "",
        f"loading {env_type} env {env_classname} from {env_path}",
        LogType.INFO,
        throw_when_excep = True)

    # getting some sim info for debugging
    sim_data = {}
    sim_info_shared = SharedEnvInfo(namespace=args.ns,
                is_server=False,
                safe=False)
    sim_info_shared.run()
    sim_info_keys = sim_info_shared.param_keys
    sim_info_data = sim_info_shared.get().flatten()
    for i in range(len(sim_info_keys)):
        sim_data[sim_info_keys[i]] = sim_info_data[i]
    
    # getting come cluster info for debugging
    cluster_data={}
    cluste_info_shared = SharedClusterInfo(namespace=args.ns,
                is_server=False,
                safe=False)
    cluste_info_shared.run()
    cluster_info_keys = cluste_info_shared.param_keys
    cluster_info_data = cluste_info_shared.get().flatten()
    for i in range(len(cluster_info_keys)):
        cluster_data[cluster_info_keys[i]] = cluster_info_data[i]

    custom_args={}
    custom_args["uname_host"]="user_host"
    try:
        username = os.getlogin() # add machine info to db data
        hostname = os.uname().nodename
        user_host = f"{username}@{hostname}"
        custom_args["uname_host"]=user_host
    except:
        pass
    
    algo=None
    if not args.dummy:
        if args.sac:
            from lrhc_control.training_algs.sac.sac import SAC

            algo = SAC(env=env, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)
        else:
            from lrhc_control.training_algs.ppo.ppo import PPO

            algo = PPO(env=env, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)
    else:
        from lrhc_control.training_algs.dummy.dummy import Dummy

        algo=Dummy(env=env, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)

    custom_args.update(args_dict)
    custom_args.update(cluster_data)
    custom_args.update(sim_data)

    run_name=env_classname if args.run_name is None else args.run_name
    algo.setup(run_name=run_name, 
        ns=args.ns,
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args=custom_args,
        comment=args.comment,
        eval=args.eval,
        model_path=mpath_full,
        n_eval_timesteps=args.n_eval_timesteps,
        dump_checkpoints=args.dump_checkpoints,
        norm_obs=args.obs_norm,
        rescale_obs=args.obs_rescale)

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
    
    eval=args.eval
    if args.override_agent_actions:
        eval=True
    if not eval:
        while not exit_request:
            if not algo.learn():
                break
    else: # eval phase
        with torch.no_grad(): # no need for grad computation
            while not exit_request:
                if not algo.eval():
                    break
    
    algo.done() # make sure to terminate training properly
