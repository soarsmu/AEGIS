from .cartpole import cartpole
from .pendulum import pendulum
from .satellite import satellite
from .dcmotor import dcmotor
from .tape import tape
from .magnetic_pointer import magnetic_pointer
from .suspension import suspension
from .biology import biology
from .cooling import cooling
from .quadcopter import quadcopter
from .self_driving import self_driving
from .car_platoon_4 import car_platoon_4
from .car_platoon_8 import car_platoon_8
from .lane_keeping import lane_keeping
from .oscillator import oscillator
from .cartpole_var import cartpole_var
from .pendulum_var_1 import pendulum_var_1
from .pendulum_var_2 import pendulum_var_2
from .self_driving_var import self_driving_var

ENV_CLASSES = {"cartpole": cartpole,
               "pendulum": pendulum,
               "satellite": satellite,
               "dcmotor": dcmotor,
               "tape": tape,
               "magnetic_pointer": magnetic_pointer,
               "suspension": suspension,
               "biology": biology,
               "cooling": cooling,
               "quadcopter": quadcopter,
               "self_driving": self_driving,
               "car_platoon_4": car_platoon_4,
               "car_platoon_8": car_platoon_8,
               "lane_keeping": lane_keeping,
               "oscillator": oscillator,
               "cartpole_var": cartpole_var,
                "pendulum_var_1": pendulum_var_1,
                "pendulum_var_2": pendulum_var_2,
                "self_driving_var": self_driving_var
               }