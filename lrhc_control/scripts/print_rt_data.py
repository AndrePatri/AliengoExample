from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import SubRewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations, SubTerminations
from lrhc_control.utils.shared_data.training_env import Truncations, SubTruncations
from lrhc_control.utils.shared_data.training_env import EpisodesCounter, TaskRandCounter, SafetyRandResetsCounter
from control_cluster_bridge.utilities.shared_data.sim_data import SharedEnvInfo

import time 
from perf_sleep.pyperfsleep import PerfSleep

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype
import torch

if __name__ == "__main__":  

    import argparse
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    
    parser.add_argument('--update_dt', type=float, help='[s]', default=0.1)
    parser.add_argument('--run_for_nominal', type=float, help='[s]]', default=10.0)
    parser.add_argument('--ns', type=str, help='', default="")
    parser.add_argument('--env_idx', type=int, help='', default=0)
    parser.add_argument('--env_range', type=int, help='', default=1)
    parser.add_argument('--dtype', type=str, help='', default="float")
    parser.add_argument('--with_counters', action="store_true", default=False, help='')
    parser.add_argument('--with_safety_counter', action="store_true", default=False, help='')
    parser.add_argument('--resolution', type=int, help='', default=2)
    parser.add_argument('--with_sub_r', action="store_true", default=True, help='')
    parser.add_argument('--with_sub_t', action="store_true", default=True, help='')
    parser.add_argument('--with_sinfo', action="store_true", default=True, help='')
    parser.add_argument('--obs_names', nargs='+', default=None,
                        help='')

    args = parser.parse_args()

    obs_selected_names=args.obs_names
    update_dt=args.update_dt
    run_for=args.run_for_nominal
    namespace=args.ns
    idx=args.env_idx
    env_range=args.env_range
    elapsed_tot_nom = 0
    with_counters = args.with_counters
    with_sf_counter=args.with_safety_counter
    n_digits=args.resolution
    dtype=eigenipc_dtype.Float
    if args.dtype == "double":
        dtype=eigenipc_dtype.Double
    
    obs = Observations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    next_obs = NextObservations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    act = Actions(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    rew = TotRewards(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    sub_rew=None
    if args.with_sub_r:
        sub_rew = SubRewards(namespace=namespace,
                        is_server=False,
                        verbose=True,
                        vlevel=VLevel.V2,
                        with_gpu_mirror=False)
    trunc = Truncations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False)
    term = Terminations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False)
    sub_trunc = None
    sub_term = None
    if args.with_sub_t:
        sub_trunc = SubTruncations(namespace=namespace,
                        is_server=False,
                        verbose=True,
                        vlevel=VLevel.V2,
                        with_gpu_mirror=False)
        sub_term = SubTerminations(namespace=namespace,
                        is_server=False,
                        verbose=True,
                        vlevel=VLevel.V2,
                        with_gpu_mirror=False)
    
    sim_data = None
    if args.with_sinfo:
        sim_data = SharedEnvInfo(namespace=namespace,
                    is_server=False,
                    safe=False,
                    verbose=True,
                    vlevel=VLevel.V2)

    obs.run()
    # next_obs.run()
    obs_names=obs.col_names()
    obs_idxs=list(range(0,len(obs_names)))
    if obs_selected_names is not None:
        obs_idxs = [obs_names.index(item) for item in obs_selected_names]
        if len(obs_idxs)==0:
            obs_selected_names=obs_names
    else:
        obs_selected_names=obs_names
    act.run()
    act_names=act.col_names()
    rew.run()
    if sub_rew is not None:
        sub_rew.run()
        sub_rew_names = sub_rew.col_names()
    trunc.run()
    term.run()
    if sub_trunc is not None:
        sub_trunc.run()
        sub_trunc_names = sub_trunc.col_names()
    if sub_term is not None:
        sub_term.run()
        sub_term_names = sub_term.col_names()
    if sim_data is not None:
        sim_data.run()
        sim_datanames = sim_data.param_keys
        simtime_idx = sim_datanames.index("cluster_time")
        rt_factor_idx = sim_datanames.index("sim_rt_factor") 

    ep_counter=None
    task_counter=None
    random_reset=None
    if with_counters:
        ep_counter=EpisodesCounter(namespace=namespace,
                    is_server=False,
                    verbose=True,
                    vlevel=VLevel.V2,
                    with_gpu_mirror=False)
        task_counter=TaskRandCounter(namespace=namespace,
                    is_server=False,
                    verbose=True,
                    vlevel=VLevel.V2,
                    with_gpu_mirror=False)
        ep_counter.run()
        task_counter.run()
        
    if with_sf_counter:
        random_reset=SafetyRandResetsCounter(namespace=namespace,
                    is_server=False,
                    verbose=True,
                    vlevel=VLevel.V2,
                    with_gpu_mirror=False)
        random_reset.run()

    torch.set_printoptions(precision=n_digits,sci_mode=False,linewidth=200)

    while True:
        try:
            start_time = time.perf_counter()

            # read data
            if sim_data is not None:
                sim_data_read=sim_data.get()
                sim_time=sim_data_read[simtime_idx].item()
                sim_rt_factor=sim_data_read[rt_factor_idx].item() 
            obs.synch_all(read=True, retry=True)
            # next_obs.synch_all(read=True, retry=True)
            act.synch_all(read=True, retry=True)
            rew.synch_all(read=True, retry=True)
            if sub_rew is not None:
                sub_rew.synch_all(read=True, retry=True)
            if sub_trunc is not None:
                sub_trunc.synch_all(read=True, retry=True)
            if sub_term is not None:
                sub_term.synch_all(read=True, retry=True)
            trunc.synch_all(read=True, retry=True)
            term.synch_all(read=True, retry=True)            

            print(f"########################")
            print(f"wall time: {round(elapsed_tot_nom, 2)} [s] -->\n")
            if sim_data is not None:
                print(f"sim time: {round(sim_time, 2)} [s]; rt factor: {round(sim_rt_factor, 2)}\n")
            print("\nobservations:")
            print(obs_selected_names, sep = ", ")
            print(obs.get_torch_mirror(gpu=False)[idx:idx+env_range, obs_idxs])
            # print("-->next observations:")
            # print(next_obs.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nactions:")
            print(act_names, sep = ", ")
            print(act.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nrewards:")
            print(rew.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if sub_rew is not None:
                print("\nsub-rewards:")
                print(*sub_rew_names, sep = ", ") 
                print(sub_rew.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nterminations:")
            print(term.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\ntruncations:")
            print(trunc.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if sub_trunc is not None:
                print("\nsub-truncations:")
                print(*sub_trunc_names, sep = ", ") 
                print(sub_trunc.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if sub_term is not None:
                print("\nsub-terminations:")
                print(*sub_term_names, sep = ", ") 
                print(sub_term.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if ep_counter is not None:
                ep_counter.counter().synch_all(read=True, retry=True)
                print("\nep. counter:")
                print(ep_counter.counter().get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if task_counter is not None:
                task_counter.counter().synch_all(read=True, retry=True)
                print("\ntask counter:")
                print(task_counter.counter().get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if random_reset is not None:
                random_reset.counter().synch_all(read=True, retry=True)
                print("\n random reset counter:")
                print(random_reset.counter().get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            
            elapsed_time = time.perf_counter() - start_time
            time_to_sleep_ns = int((update_dt - elapsed_time) * 1e+9) # [ns]
            if time_to_sleep_ns < 0:
                warning = f": Could not match desired update dt of {update_dt} s. " + \
                    f"Elapsed time to update {elapsed_time}."
                Journal.log("print_rt_data",
                    "run",
                    warning,
                    LogType.WARN,
                    throw_when_excep = True)
            else:
                PerfSleep.thread_sleep(time_to_sleep_ns) 
            elapsed_tot_nom+=update_dt
            if elapsed_tot_nom>=run_for:
                break
        except KeyboardInterrupt:
            break