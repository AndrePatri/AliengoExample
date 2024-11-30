from control_cluster_bridge.utilities.keyboard_cmds import RefsFromKeyboard

import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    args = parser.parse_args()
    
    from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
    from EigenIPC.PyEigenIPC import VLevel

    shared_refs= RhcRefs(namespace=args.ns,
        is_server=False, 
        safe=False, 
        verbose=True,
        vlevel=VLevel.V2)

    keyb_cmds = RefsFromKeyboard(namespace=args.ns, 
                            shared_refs=shared_refs,
                            verbose=True)

    keyb_cmds.run()