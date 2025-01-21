from lrhc_control.utils.keyboard_cmds import AgentRefsFromKeyboard
from lrhc_control.utils.keyboard_cmds import AgentActionsFromKeyboard

import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--actions',action='store_true', help='whether to send agent actions instead of refs')
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--agent_refs_world',action='store_true', 
        help='whether to set the agent ref in world frame (it will be internally adjucted to base frame)')
    parser.add_argument('--cmapping', type=str, help='contact mapping to, respectively, keys 7 9 1 and 3', default="0;1;2;3")
    parser.add_argument('--env_idx', type=int,default=None)
    parser.add_argument('--from_stdin', action='store_true')

    args = parser.parse_args()
    
    keyb_cmds=None
    if args.actions:
        keyb_cmds = AgentActionsFromKeyboard(namespace=args.ns, 
                            verbose=True,
                            contact_mapping=args.cmapping,
                            env_idx=args.env_idx)
    else:
        keyb_cmds = AgentRefsFromKeyboard(namespace=args.ns, 
                            verbose=True,
                            agent_refs_world=args.agent_refs_world,
                            env_idx=args.env_idx)
        
    
    keyb_cmds.run(read_from_stdin=args.from_stdin)