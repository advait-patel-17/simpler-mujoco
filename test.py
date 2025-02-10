import mujoco
import mujoco.viewer
from mujoco import MjData, MjModel
# Loading a specific model description as an imported module.
from robot_descriptions import aloha_mj_description
# model = mujoco.MjModel.from_xml_path(aloha_mj_description.MJCF_PATH)

# # Directly loading an instance of MjModel.
from robot_descriptions.loaders.mujoco import load_robot_description
# model = load_robot_description("panda_mj_description")

# Loading a variant of the model, e.g. panda without a gripper.
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

ds_filepath = "./data/episode_1.hdf5"
file = h5py.File(ds_filepath, 'r')

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

"""
DIMENSIONS

table length x width x height (m): 1.21 x 0.76 x 0.75


"""