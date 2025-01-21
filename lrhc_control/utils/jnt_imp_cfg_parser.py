import yaml
import re
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

class JntImpConfigParser:
    def __init__(self, 
            config_file, 
            joint_names, 
            default_p_gain=0, 
            default_v_gain=0, 
            backend="numpy", 
            device="cpu",
            search_for="motor_pd"):
        
        self._pd_gains_parentname=search_for

        self.config_file = config_file
        self.joint_names = joint_names
        self.default_p_gain = default_p_gain
        self.default_v_gain = default_v_gain
        self.gain_matrix = None
        
        self.backend = backend
        self.device = device
        
        if self.backend == "numpy" and self.device != "cpu":
            Journal.log(self.__class__.__name__,
                "__init__",
                "When using numpy backend, only cpu device is supported!",
                LogType.EXCEP,
                throw_when_excep = True)
        
        # Attempt to load the configuration file
        self.load_config()
        # Create the gain matrix based on the configuration or default values
        self.create_gain_matrix()

    def load_config(self):
        self.config_data = None  # Set to None to indicate fail
        if self.config_file is not None:
            try:
                # Load YAML configuration
                with open(self.config_file, 'r') as file:
                    self.config_data = yaml.safe_load(file)
            except (FileNotFoundError, yaml.YAMLError) as e:
                # If the file cannot be loaded, print a warning and use default gains
                Journal.log(self.__class__.__name__,
                    "__init__",
                    "Could not load configuration file {self.config_file}. Using default gains. Error: {e}",
                    LogType.WARN)

    def create_gain_matrix(self):
        num_joints = len(self.joint_names)
        
        # Initialize the gain matrix with the appropriate backend
        if self.backend == "numpy":
            import numpy as np  # Assuming numpy is used for backend in this example
            self.gain_matrix = np.full((num_joints, 2), np.nan)
        elif self.backend == "torch":
            import torch
            self.gain_matrix = torch.full((num_joints, 2), torch.nan, device=self.device)
        else:
            raise Exception("Backend not supported")
        
        # If configuration data is not loaded, use default gains for all joints
        if self.config_data is None or self._pd_gains_parentname not in self.config_data:
            for jnt_index in range(num_joints):
                self.gain_matrix[jnt_index, 0] = self.default_p_gain  # kp
                self.gain_matrix[jnt_index, 1] = self.default_v_gain  # kd
            return
        
        # Pattern matching setup from the loaded configuration
        pattern_dict = {k: v for k, v in self.config_data.get(self._pd_gains_parentname, {}).items()}
        
        for jnt_index, joint_name in enumerate(self.joint_names):
            matched = False
            for pattern, gains in pattern_dict.items():
                if self.match_pattern(joint_name, pattern):
                    self.gain_matrix[jnt_index, 0] = gains[0]  # kp
                    self.gain_matrix[jnt_index, 1] = gains[1]  # kd
                    matched = True
                    break
            if not matched:
                # Set default values if no match is found
                self.gain_matrix[jnt_index, 0] = self.default_p_gain
                self.gain_matrix[jnt_index, 1] = self.default_v_gain

    def match_pattern(self, joint_name, pattern):
        # Convert pattern into a regular expression
        regex = pattern.replace('*', '.*')
        return re.fullmatch(regex, joint_name) is not None
    
    def get_pd_gains(self):
        return self.gain_matrix


# Test the JntImpConfigParser
if __name__ == "__main__":
    import tempfile
    import os

    # Define the temporary YAML configuration content
    yaml_content = """
motor_pd:
  "*arm*": [500, 10]
  "hip_yaw_*": [3000, 30]
  "hip_pitch_*": [3000, 30]
  "knee_pitch_*": [3000, 30]
  "ankle_pitch_*": [1000, 10]
  "ankle_yaw_*": [300, 10]
  "neck_pitch": [10, 1]
  "neck_yaw": [10, 1]
  "torso_yaw": [1000, 30]
  "j_wheel_*": [0, 30]

startup_motor_pd:
  "*arm*": [200, 5]
  "hip_yaw_*": [1000, 10]
  "hip_pitch_*": [1000, 10]
  "knee_pitch_*": [1000, 10]
  "ankle_pitch_*": [500, 10]
  "ankle_yaw_*": [150, 5]
  "neck_pitch": [10, 1]
  "neck_yaw": [10, 1]
  "torso_yaw": [500, 10]
  "j_wheel_*": [0, 30]
    """

    # Create a temporary file to store the YAML content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        temp_file.write(yaml_content.encode('utf-8'))
        temp_file_path = temp_file.name

    # Define the joint names for testing
    joint_names = [
        'j_arm_1', 'j_arm_123_5', 'j_arm_7', 'hip_yaw_1', 'ankle_pitch_4', 'neck_yaw',
        'unknown_joint', 'j_wheel_1', 'forearm_yaw'
    ]

    # Create an instance of JntImpConfigParser with the temporary YAML file
    backend = "torch"
    parser = JntImpConfigParser(
        config_file=temp_file_path,
        joint_names=joint_names,
        default_p_gain=50,  # Example default proportional gain
        default_v_gain=5,   # Example default derivative gain,
        backend=backend,
        search_for="motor_pd"
    )
    parser_startup = JntImpConfigParser(
        config_file=temp_file_path,
        joint_names=joint_names,
        default_p_gain=50,  # Example default proportional gain
        default_v_gain=5,   # Example default derivative gain,
        backend=backend,
        search_for="startup_motor_pd"
    )

    # Print the resulting gain matrix with specified formatting
    if parser.backend == "numpy":
        import numpy as np
        np.set_printoptions(precision=3, suppress=True)
    elif parser.backend == "torch":
        import torch
        torch.set_printoptions(precision=3, sci_mode=False)
    print("Gain Matrix:")
    print(joint_names)
    print(parser.get_pd_gains())
    print("Startup Gain Matrix:")
    print(joint_names)
    print(parser_startup.get_pd_gains())

    os.remove(temp_file_path)
