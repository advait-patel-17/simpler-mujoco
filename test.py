import mujoco
import mujoco.viewer
from mujoco import MjData, MjModel

# Loading a variant of the model, e.g. panda without a gripper.
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

ds_filepath = "./data/episode_1.hdf5"
file = h5py.File(ds_filepath, 'r')


"""
DIMENSIONS

table length x width x height (m): 1.21 x 0.76 x 0.75


"""