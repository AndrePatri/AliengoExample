from lrhc_control.utils.keyboard_cmds import AgentRefsFromKeyboard
from lrhc_control.utils.keyboard_cmds import AgentActionsFromKeyboard

import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--actions',action='store_true', help='whether to send agent actions instead of refs')
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--agent_refs_world',action='store_true', 
        help='whether to set the agent ref in world frame (it will be internally adjucted to base frame)')

    args = parser.parse_args()
    
    keyb_cmds=None
    if args.actions:
        keyb_cmds = AgentActionsFromKeyboard(namespace=args.ns, 
                                verbose=True)
    else:
        keyb_cmds = AgentRefsFromKeyboard(namespace=args.ns, 
                                verbose=True,
                                agent_refs_world=args.agent_refs_world)
        
    
    keyb_cmds.run()