# Ensure custom_args_names, custom_args_vals, and custom_args_dtype have the same length
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
import argparse


def extract_custom_xacro_args(custom_opt):
    xacro_commands = []

    # Iterate through the custom_args dictionary
    for key, value in custom_opt.items():
        # Check if the value contains the specific xacro command format
        if type(value)==str and (':=' in value):
            xacro_commands.append(value)
    return xacro_commands

def generate_custom_arg_dict(args: argparse.Namespace):
    custom_opt = {}
    if args.custom_args_names and args.custom_args_vals and args.custom_args_dtype:
        if not (len(args.custom_args_names) == len(args.custom_args_vals) == len(args.custom_args_dtype)):
            Journal.log("launch_control_cluster.py",
                "",
                f"custom_args_names, custom_args_vals, and custom_args_dtype lengths do not match!",
                LogType.EXCEP,
                throw_when_excep = False)
            exit()
        
        # Build custom_opt dictionary with appropriate data types
        for name, val, dtype in zip(args.custom_args_names, args.custom_args_vals, args.custom_args_dtype):
            try:
                # Convert the value to the appropriate type
                if dtype == "int":
                    custom_opt[name] = int(val)
                elif dtype == "float":
                    custom_opt[name] = float(val)
                elif dtype == "bool":
                    custom_opt[name] = val.lower() in ["true", "1", "yes"]
                elif dtype == "xacro":
                    val_str=str(val)
                    custom_opt[name] = f"{name}:="+val_str
                elif dtype == "strlist":
                    # Split the value by commas and strip whitespace from each item
                    custom_opt[name] = [item.strip() for item in val.split(",")]
                else:  # Default is string
                    custom_opt[name] = val
            except ValueError as e:
                Journal.log("custom_arg_parsing.py",
                    "generate_custom_arg_dict",
                    f"Error converting {name} to {dtype}: {e}",
                    LogType.EXCEP,
                    throw_when_excep = True)

    return custom_opt

def merge_xacro_cmds(prev_cmds, new_cmds):
    """
    Merges two lists of xacro commands.
    
    Args:
        prev_cmds (list): The original list of xacro commands in the format "arg:=val".
        new_cmds (list): The new list of xacro commands to substitute or append.
        
    Returns:
        list: A merged list of xacro commands.
    """
    
    # Create a dictionary from the previous commands
    xacro_dict = {}
    for cmd in prev_cmds:
        arg, val = cmd.split(":=")
        xacro_dict[arg.strip()] = val.strip()  # Store arguments and values

    # Substitute or append new commands
    for new_cmd in new_cmds:
        new_arg, new_val = new_cmd.split(":=")
        new_arg = new_arg.strip()
        new_val = new_val.strip()
        
        # Update or add the new command
        xacro_dict[new_arg] = new_val  # This will replace existing values if the key is found

    # Create the final list from the updated dictionary
    final_xacro_cmds = [f"{arg}:={val}" for arg, val in xacro_dict.items()]

    return final_xacro_cmds

# Example usage
if __name__ == "__main__":
    original_xacro_commands = [
        "arg1:=value1",
        "arg2:=value2",
        "arg3:=value3",
        "arg4:=value4"
    ]

    new_xacro_commands = [
        "arg2:=new_value2",  # Should substitute arg2
        "arg3:=new_value3",  # Should substitute arg3
        "arg5:=value5"       # Should be appended as arg5 does not exist
    ]

    merged_commands = merge_xacro_cmds(original_xacro_commands, new_xacro_commands)
    print(merged_commands)


