import argparse
import os
import signal
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper

from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererClnt,RemoteTriggererSrvr

from SharsorIPCpp.PySharsorIPC import StringTensorClient

from perf_sleep.pyperfsleep import PerfSleep

def launch_rosbag(namespace: str, dump_path: str, timeout_sec:float):
        
    import multiprocess as mp

    retry_kill=20
    additional_secs=5.0
    term_trigger=RemoteTriggererClnt(namespace=namespace+f"SharedTerminator",
                            verbose=True,
                            vlevel=VLevel.V1)
    term_trigger.run()

    Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            "launching rosbag recording",
            LogType.INFO)
    
    command = ["./launch_rosbag.sh", "--ns", namespace, "--output_path", dump_path]
    ctx = mp.get_context('forkserver')
    proc = ctx.Process(target=os.system, args=(' '.join(command),))
    proc.start()

    timeout_ms = int(timeout_sec*1e3)
    if not term_trigger.wait(timeout_ms):
        Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            "Didn't receive any termination req within timeout! Will terminate anyway",
            LogType.WARN)
    
    Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            f"terminating rosbag recording. Dump base-path is: {dump_path}",
            LogType.INFO)

    term_trigger.close()

    # os.killpg(os.getpgid(proc.pid), signal.SIGINT)  # Send SIGINT to the whole pro
    # proc.send_signal(signal.SIGINT)  # Gracefully interrupt bag collection
    try: 
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except:
        pass

    proc.join()

    Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            f"successfully terminated rosbag recording process",
            LogType.INFO)

if __name__ == '__main__':

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--dt', type=float, default=0.01, help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, default is False')
    parser.add_argument('--verbose',action='store_true', help='Enable verbose mode, default is True')
    parser.add_argument('--ros2',action='store_true', help='Use ROS 2')
    parser.add_argument('--with_agent_refs',action='store_true', help='also forward agent refs to rhcviz')
    parser.add_argument('--rhc_refs_in_h_frame',action='store_true', help='set to true if rhc refs are \
                        specified in the horizontal frame')
    parser.add_argument('--agent_refs_in_h_frame',action='store_true', help='set to true if agent refs are \
                        specified in the horizontal frame')
    parser.add_argument('--env_idx', type=int, help='env index of which data is to be published', default=0)
    parser.add_argument('--stime_trgt', type=float, default=None, help='sim time for which this bridge runs (None -> indefinetly)')
    parser.add_argument('--srdf_path', type=str, help='path to SRDF path specifying homing configuration, to be used for missing joints', default=None)
    parser.add_argument('--abort_wallmin', type=float, default=5.0, help='abort bridge if no response wihtin this timeout')

    args = parser.parse_args()

    # Use the provided robot name and update interval
    timeout_ms=240000
    update_dt = args.dt
    debug = args.debug
    verbose = args.verbose

    bridge = None
    if not args.ros2: 
        from lrhc_control.utils.rhc_viz.rhc2viz import RhcToVizBridge

        bridge = RhcToVizBridge(namespace=args.ns, 
                        verbose=verbose,
                        rhcviz_basename="RHCViz", 
                        robot_selector=[0, None],
                        with_agent_refs=args.with_agent_refs,
                        rhc_refs_in_h_frame=args.rhc_refs_in_h_frame,
                        agent_refs_in_h_frame=args.agent_refs_in_h_frame,
                        env_idx=args.env_idx,
                        sim_time_trgt=args.stime_trgt,
                        srdf_homing_file_path=args.srdf_path,
                        abort_wallmin=args.abort_wallmin)
    else:

        from lrhc_control.utils.rhc_viz.rhc2viz2 import RhcToViz2Bridge

        bridge = RhcToViz2Bridge(namespace=args.ns, 
                        verbose=verbose,
                        rhcviz_basename="RHCViz", 
                        robot_selector=[0, None],
                        with_agent_refs=args.with_agent_refs,
                        rhc_refs_in_h_frame=args.rhc_refs_in_h_frame,
                        agent_refs_in_h_frame=args.agent_refs_in_h_frame,
                        env_idx=args.env_idx,
                        sim_time_trgt=args.stime_trgt,
                        srdf_homing_file_path=args.srdf_path,
                        abort_wallmin=args.abort_wallmin,
                        update_dt=update_dt)

    bridge.run()

    bridge.close()


