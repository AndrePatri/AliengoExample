
import signal
import time

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import dtype

class RosBagDumperv2():

    def __init__(self,
            ns:str,
            ros_bridge_dt: float,
            bag_sdt: float,
            wall_dt: float,
            remap_ns: str = None,
            debug: bool=False,
            verbose: bool=False,
            dump_path: str = "/tmp",
            is_training: bool = False,
            use_shared_drop_dir:bool=False,
            ros2:bool=True,
            env_idx:int=0,
            srdf_path:str=None,
            abort_wallmin:float=5.0,
            with_agent_refs:bool=True,
            rhc_refs_in_h_frame:bool=True,
            agent_refs_in_h_frame:bool=False,
            use_static_idx: bool = True,
            pub_stime: bool = True):

        self._closed=False
        
        self._pub_stime=pub_stime

        self._ns=ns
        self._remap_ns=remap_ns
        if self._remap_ns is None: # allow to publish with different namespace (to allow
            # support for multiple bags at once and multiple rhcviz instances)
            self._remap_ns=self._ns

        self._srdf_path=None

        self._with_agent_refs=with_agent_refs
        self._rhc_refs_in_h_frame=rhc_refs_in_h_frame
        self._agent_refs_in_h_frame=agent_refs_in_h_frame

        self._env_idx=env_idx
        self._use_static_idx=use_static_idx

        self._timeout_ms=240000
        self._dump_path=dump_path
        self._verbose=verbose
        self._debug=debug
        self._ros_bridge_dt=ros_bridge_dt
        self._bag_sdt=bag_sdt
        self._wall_dt=wall_dt

        self._ros2=ros2

        self._shared_info=None
        self._is_training=is_training
        self._use_shared_drop_dir=use_shared_drop_dir

        self._bridge = None

        # spawn a process to record bag if required
        self._bag_proc=None
        self._term_trigger=None

        self._is_done_idx=None
        
        self._abort_wallmin=abort_wallmin

        self._initialize()
        
        self._init()

    def __del__(self):
        self.close()

    def _initialize(self):
        
        self._spawn_rosbag_process() # spwaing bag process here before importing unpickable stuff

        if self._is_training:
            # for detecting when training is finished
            from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo
            
            self._shared_info=SharedRLAlgorithmInfo(is_server=False,
                        namespace=self._ns, 
                        verbose=True, 
                        vlevel=VLevel.V2)
            self._shared_info.run()
            shared_info_names=self._shared_info.dynamic_info.get()
            self._is_done_idx=shared_info_names.index("is_done")

        # bridge from rhc shared data to ROS
        if not self._ros2:
            from lrhc_control.utils.rhc_viz.rhc2viz import RhcToVizBridge
            self._bridge = RhcToVizBridge(namespace=self._ns, 
                remap_ns=self._remap_ns,
                verbose=self._verbose,
                rhcviz_basename="RHCViz", 
                robot_selector=[0, None],
                with_agent_refs=self._with_agent_refs,
                rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                agent_refs_in_h_frame=self._agent_refs_in_h_frame,
                env_idx=self._env_idx,
                sim_time_trgt=self._bag_sdt,
                srdf_homing_file_path=self._srdf_path,
                abort_wallmin=self._abort_wallmin,
                use_static_idx=self._use_static_idx,
                pub_stime=self._pub_stime)
        else:
            from lrhc_control.utils.rhc_viz.rhc2viz2 import RhcToViz2Bridge
            self._bridge = RhcToViz2Bridge(namespace=self._ns, 
                remap_ns=self._remap_ns,
                verbose=self._verbose,
                rhcviz_basename="RHCViz", 
                robot_selector=[0, None],
                with_agent_refs=self._with_agent_refs,
                rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                agent_refs_in_h_frame=self._agent_refs_in_h_frame,
                env_idx=self._env_idx,
                sim_time_trgt=self._bag_sdt,
                srdf_homing_file_path=self._srdf_path,
                abort_wallmin=self._abort_wallmin,
                use_static_idx=self._use_static_idx,
                pub_stime=self._pub_stime)

        # actual process recording bag
        from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererSrvr
        self._term_trigger=RemoteTriggererSrvr(namespace=self._remap_ns+f"SharedTerminator",
                                            verbose=self._verbose,
                                            vlevel=VLevel.V2,
                                            force_reconnection=True)
        from EigenIPC.PyEigenIPC.wrappers.shared_data_view import SharedTWrapper
        from EigenIPC.PyEigenIPC import dtype
        self._bag_req=SharedTWrapper(namespace=self._remap_ns,
            basename="RosBagRequests",
            is_server=True, 
            n_rows=1, 
            n_cols=3, # [start rosbag, stop rosbag, kill rosbag process]
            dtype = dtype.Bool,
            verbose = self._verbose, 
            vlevel = VLevel.V2,
            fill_value=False, 
            safe=False,
            force_reconnection=True,
            with_gpu_mirror=False,
            with_torch_view=False)
        self._bag_req.run()
    
    def _init(self):

        self._term_trigger.run()
        self._bridge.init(update_dt=self._ros_bridge_dt)
    
    def run(self):
        traing_done=False
        while not traing_done:
            try:
                # continue publishing state on topics
                if self._is_training: # check if training is done
                    traing_done=self.training_done()
                start_time=time.monotonic() 
                self._start_bag_recording()
                if not self._bridge.run(sim_time=self._bag_sdt):
                    break
                self._stop_bag_recording()
                elapsed_min=(time.monotonic()-start_time)*1.0/60.0
                remaining_min=self._wall_dt-elapsed_min
                if remaining_min>0.0: # wait 
                    time.sleep(remaining_min*60.0)
            except KeyboardInterrupt:
                break
        self.close()
        
    def training_done(self):
        
        return self._shared_info.get().flatten()[self._is_done_idx]>0.5

    def _spawn_rosbag_process(self):
        import multiprocess as mp
        ctx = mp.get_context('forkserver')
        bag_id=str(self._env_idx)
        self._bag_proc=ctx.Process(target=self._launch_rosbag, 
            name="rosbag_recorder_"+f"{self._ns}",
            args=(self._remap_ns,self._dump_path, self._timeout_ms, bag_id,self._use_shared_drop_dir))
        self._bag_proc.start()
    
    def _launch_rosbag(self, 
            namespace: str, dump_path: str, timeout_sec:float, bag_id: str, use_shared_drop_dir: bool = True):
        
        def ignore_keyboard_interrupt():
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        def run_bag(command):
            proc = subprocess.Popen(command, shell=shell,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid
            # preexec_fn=os.setsid # crucial -> all childs will have the same ID
            )
            # Set the process group ID to the subprocess PID
            # os.setpgid(proc.pid, proc.pid)
            Journal.log(self.__class__.__name__,
                    "launch_rosbag",
                    f"launching rosbag recording with PID {proc.pid}",
                    LogType.INFO)

            return proc
        
        ignore_keyboard_interrupt()

        import os

        Journal.log(self.__class__.__name__,
            "launch_rosbag",
            f"launch_rosbag PID is {os.getpid()}",
            LogType.INFO)

        # using a shared drop dir if enabled
        from EigenIPC.PyEigenIPC import StringTensorClient
        from perf_sleep.pyperfsleep import PerfSleep

        if use_shared_drop_dir:
            shared_drop_dir=StringTensorClient(basename="SharedTrainingDropDir", 
                            name_space=self._ns,
                            verbose=True, 
                            vlevel=VLevel.V2)
            shared_drop_dir.run()
            shared_drop_dir_val=[""]*shared_drop_dir.length()
            while not shared_drop_dir.read_vec(shared_drop_dir_val, 0):
                ns=1000000000
                PerfSleep.thread_sleep(ns)
                continue
            dump_path=shared_drop_dir_val[0] # overwrite

            shared_drop_dir.close()
        retry_kill=20
        additional_secs=5.0
        from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererClnt
        
        time.sleep(3.0) # wait a bit in case server side crashed and needs to be recrated
        term_trigger=RemoteTriggererClnt(namespace=namespace+f"SharedTerminator",
                                verbose=True,
                                vlevel=VLevel.V2)
        term_trigger.run()

        from EigenIPC.PyEigenIPC.wrappers.shared_data_view import SharedTWrapper
        from EigenIPC.PyEigenIPC import dtype

        bag_req=SharedTWrapper(namespace=namespace,
            basename="RosBagRequests",
            is_server=False, 
            dtype = dtype.Bool,
            verbose = self._verbose, 
            vlevel = VLevel.V2,
            safe=False,
            with_gpu_mirror=False,
            with_torch_view=False)
        bag_req.run()

        this_dir_path=os.path.dirname(__file__)

        shell=False
        command=None
        if not shell:
            if self._ros2:
                command = [f"{this_dir_path}/launch_ros2bag.sh", "--ns", namespace, "--id", bag_id, "--output_path", dump_path]
            else:
                command = [f"{this_dir_path}/launch_ros1bag.sh", "--ns", namespace, "--id", bag_id, "--output_path", dump_path]
        else:
            if self._ros2:
                command = f"{this_dir_path}/launch_ros2bag.sh --ns {namespace} --id {bag_id} --output_path {dump_path}"
            else:
                command = f"{this_dir_path}/launch_ros1bag.sh --ns {namespace} --id {bag_id} --output_path {dump_path}"

        import subprocess

        exit_req=False
        rosbag_started=False
        while not exit_req:
            timeout_ms = int(timeout_sec*1e3)
            if not term_trigger.wait(timeout_ms):
                Journal.log(self.__class__.__name__,
                    "launch_rosbag",
                    "Didn't receive any termination req within timeout! Will terminate anyway",
                    LogType.WARN)
                exit_req=True
                break
        
            # trigger received
            
            bag_req.synch_all(read=True,retry=True)
            req_data=bag_req.get_numpy_mirror()
            start_rosbag=req_data[:, 0].item()
            stop_rosbag=req_data[:, 1].item()
            terminate_bag_proc=req_data[:, 2].item()
            print("received request:")
            print(req_data)
            if start_rosbag and not rosbag_started:
                proc=run_bag(command)
                stop_rosbag=False
                rosbag_started=True
            if stop_rosbag and rosbag_started:
                Journal.log(self.__class__.__name__,
                    "launch_rosbag",
                    f"terminating rosbag recording. Dump base-path is: {dump_path}",
                    LogType.INFO)
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                # os.kill(proc.pid, signal.SIGINT)
                # proc.send_signal(signal.SIGINT)
                    
                try:
                    proc.wait(timeout=2.0)
                except:
                    proc.kill()

                Journal.log(self.__class__.__name__,
                        "launch_rosbag",
                        f"successfully terminated rosbag recording process",
                        LogType.INFO)
                rosbag_started=False

            exit_req=terminate_bag_proc
            term_trigger.ack()
        
        term_trigger.close()
        Journal.log(self.__class__.__name__,
            "launch_rosbag",
            f"exiting recording loop",
            LogType.INFO)
    
    def _start_bag_recording(self):
        bag_data=self._bag_req.get_numpy_mirror()
        bag_data[:, 0] = True
        bag_data[:, 1] = False
        bag_data[:, 2] = False
        self._bag_req.synch_all(read=False,retry=True)
        self._term_trigger.trigger() # triggering process termination and joining
        if not self._term_trigger.wait_ack_from(1, 
                self._timeout_ms):
            Journal.log(self.__class__.__name__,
                "_start_bag_recording",
                f"Didn't receive ack!",
                LogType.EXCEP,
                throw_when_excep = True)

    def _stop_bag_recording(self):
        bag_data=self._bag_req.get_numpy_mirror()
        bag_data[:, 0] = False
        bag_data[:, 1] = True
        bag_data[:, 2] = False
        self._bag_req.synch_all(read=False,retry=True)
        self._term_trigger.trigger() # triggering process termination and joining
        if not self._term_trigger.wait_ack_from(1, 
                self._timeout_ms):
            Journal.log(self.__class__.__name__,
                "_start_bag_recording",
                f"Didn't receive ack!",
                LogType.EXCEP,
                throw_when_excep = True)
        
        
    def _close_rosbag_proc(self):
        bag_data=self._bag_req.get_numpy_mirror()
        bag_data[:, 0] = False
        bag_data[:, 1] = True # close in case was running
        bag_data[:, 2] = True
        self._bag_req.synch_all(read=False,retry=True)
        self._term_trigger.trigger() # triggering process termination and joining
        if not self._term_trigger.wait_ack_from(1, 
                self._timeout_ms):
            Journal.log(self.__class__.__name__,
                "_start_bag_recording",
                f"Didn't receive ack!",
                LogType.EXCEP,
                throw_when_excep = True)

    def _close_rosbag(self):
        if self._bag_proc is not None:
            self._close_rosbag_proc()
            ret=self._bag_proc.join(5) # waits some time 
            if ret is not None:
                if self._bag_proc.exitcode is None: # process not terminated yet
                    Journal.log(self.__class__.__name__,
                        "close",
                        f"forcibly terminating bag process",
                        LogType.WARN)
                    self._bag_proc.terminate()

    def close(self):
        if not self._closed:
            self._close_rosbag()

            if self._term_trigger is not None:
                self._term_trigger.close()
            
            self._bridge.close()
            self._closed=True
