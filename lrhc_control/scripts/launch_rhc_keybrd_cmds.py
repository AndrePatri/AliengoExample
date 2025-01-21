from control_cluster_bridge.utilities.keyboard_cmds import RefsFromKeyboard

import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--cmapping', type=str, help='contact mapping to, respectively, keys 7 9 1 and 3', default="0;1;2;3")
    parser.add_argument('--env_idx', type=int,default=None)
    parser.add_argument('--from_stdin', action='store_true')

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
                            verbose=True,
                            contact_mapping=args.cmapping,
                            env_idx=args.env_idx)

    keyb_cmds.run(read_from_stdin=args.from_stdin)