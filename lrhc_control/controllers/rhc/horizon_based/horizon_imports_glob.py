"""
Dynamically import all necessary Horizon and related dependencies.
This function is intended to be used within the import_child_lib method
of a class, enabling the parent process to load all required libraries.
"""
def import_horizon_global():
    # Global imports to make modules accessible in child processes
    global casadi_kin_dyn, cs, kd, Problem, FullModelInverseDynamics
    global TaskInterface, VertexContact, ContactTask, trajectoryGenerator, utils
    global pymanager, pyphase, pytimeline

    # robot modeling and automatic differentiation
    import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
    import casadi as cs

    # horizon stuff
    import horizon.utils.kin_dyn as kd
    from horizon.problem import Problem
    from horizon.rhc.model_description import FullModelInverseDynamics
    from horizon.rhc.taskInterface import TaskInterface
    from horizon.rhc.tasks.interactionTask import VertexContact
    from horizon.rhc.tasks.contactTask import ContactTask
    from horizon.utils import trajectoryGenerator, utils

    # phase managing
    import phase_manager.pymanager as pymanager
    import phase_manager.pyphase as pyphase
    import phase_manager.pytimeline as pytimeline
