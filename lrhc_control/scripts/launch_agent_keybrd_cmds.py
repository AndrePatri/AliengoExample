from lrhc_control.utils.keyboard_cmds import AgentRefsFromKeyboard
from lrhc_control.utils.keyboard_cmds import AgentActionsFromKeyboard

import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--actions',action='store_true', help='whether to send agent actions instead of refs')
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')

    args = parser.parse_args()
    
    keyb_cmds=None
    if args.actions:
        keyb_cmds = AgentActionsFromKeyboard(namespace=args.ns, 
                                verbose=True)
    else:
        keyb_cmds = AgentRefsFromKeyboard(namespace=args.ns, 
                                verbose=True)
        
    
    keyb_cmds.run()