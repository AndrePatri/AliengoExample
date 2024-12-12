import argparse

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
    parser.add_argument('--pub_stime',action='store_true', help='whether to publish /clock')

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
                        abort_wallmin=args.abort_wallmin, 
                        pub_stime=args.pub_stime,
                        install_sighandler=True)
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
                        update_dt=update_dt, 
                        pub_stime=args.pub_stime,
                        install_sighandler=False)

    bridge.run()

    bridge.close()


