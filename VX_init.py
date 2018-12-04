
from VxSim import *
import zmq
import numpy as np
from x2q2cyl import X2Q2Cyl
from math import *
import time
from scipy import io

SCENE_DIR = "C:\CM Labs\Vortex Studio Samples 2018b\Scenario\Excavator Scene\Excavator_basic.vxscene"
PORT = "5555"

X_INIT = 4.0
Z_INIT = 0.0
PHI_INIT = -pi/2

BOOM_L = 3.0632
ARM_L = 3.7442
BUCKET_L = 1.7906

BOOM_INIT = 2.7070
ARM_INIT = 2.5891
BUCKET_INIT = 2.0081

# Load vortex scene file
def loader():
    application = VxApplication()

    fileManager = application.getSimulationFileManager()
    imp_file = SCENE_DIR
    scene = SceneInterface(fileManager.loadObject(imp_file))

    # Add the graphic module that uses OSG
    graphicsModule = VxExtensionFactory.create(GraphicsModuleICD.kModuleFactoryKey)
    application.insertModule(graphicsModule)

    display = VxExtensionFactory.create(DisplayICD.kExtensionFactoryKey)
    display.getInput(DisplayICD.kPlacementMode).setValue("Windowed")
    display.getInput(DisplayICD.kPlacement).setValue(VxVector4(50, 50, 1280, 720))
#    Setup.mViewport()
    application.add(display)


    #  Add a dynamics module (physics engine)
    dynamicsModule = VxExtensionFactory.create(VxDynamicsModuleICD.kFactoryKey)
    application.insertModule(dynamicsModule)

    # Add the DynamicsVisualizer (physics objects visualization)
    dynamicsVisualizer = VxExtensionFactory.create(DynamicsVisualizerICD.kExtensionFactoryKey)
    application.add(dynamicsVisualizer)

    application.update()

    return application, scene

# Access to Vortex studio parameters
def parameter(scene):
    excavator_mechanism = [m for m in scene.getMechanisms() if m.getName() == 'Excavator'][0]
    excavator_assembly = [a for a in excavator_mechanism.getAssemblies() if a.getName() == 'Excavator'][0]
    excavator_constraint = excavator_assembly.getConstraints()
    #excavator_parts = excavator_assembly.getParts()
    print("Excavator connected...")

    boom_act = ConstraintInterface(excavator_constraint[17])
    boom_container = boom_act.getInputCoordinate(0)
    boom_pos = boom_container[5][0]
    print("Boom connected...")

    arm_act = ConstraintInterface(excavator_constraint[28])
    arm_container = arm_act.getInputCoordinate(0)
    arm_pos = arm_container[5][0]
    print("Arm connected...")

    bucket_act = ConstraintInterface(excavator_constraint[29])
    bucket_container = bucket_act.getInputCoordinate(0)
    bucket_pos = bucket_container[5][0]
    print("Bucket connected..")

    excavator_earthwork = scene.findExtensionByName("Dynamics Bucket")
    earthwork_container = excavator_earthwork.getOutputContainer()
    payload = earthwork_container[1]
    print("Bucket detected..")

    height_dynamics = scene.findExtensionByName("HeightDynamics")
    height_output_cont = height_dynamics.getOutputContainer()
    height_output = height_output_cont[3][4]

    height_graphics = scene.findExtensionByName("HeightGraphics")
    height_input_cont = height_graphics.getInputContainer()
    height_input = height_input_cont[8][4]

    application.update()

    return boom_pos, arm_pos, bucket_pos, payload, height_output, height_input


def zmq_socket():

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % PORT)
    print("Vortex studio socket open..")

    return socket

def reset(x2q2c, boom_act, arm_act, bucket_act, i):

    q1,q2,q3 = x2q2c.x2q(X_INIT, Z_INIT, PHI_INIT)
    L1, L2, L3 = x2q2c.q2cyl(q1,q2,q3)

    boom_act.value = L1
    arm_act.value = L2
    bucket_act.value = L3

    print (L1, L2, L3)
    application.update()

def step(boom_act, arm_act, bucket_act, payload, action):

    q1, q2, q3 = x2q2c.x2q(X_INIT+action[0], Z_INIT+action[1], PHI_INIT+action[2])
    L1, L2, L3 = x2q2c.q2cyl(q1, q2, q3)

    boom_act.value = L1
    arm_act.value = L2
    bucket_act.value = L3

    application.update()

    reward = payload
    return reward

if __name__ == '__main__':
    start_time=time.time()

    application, scene = loader()
    boom_act, arm_act, bucket_act, payload, height_output, height_input= parameter(scene)
    x2q2c=X2Q2Cyl()

    for i in range(100):
        boom_act.value = -(BOOM_INIT - BOOM_L)
        arm_act.value = -(ARM_INIT - ARM_L)
        bucket_act.value = (BUCKET_INIT - BUCKET_L)
        application.update()

    trjExp = io.loadmat("D:\Doosan_simul\Excavator Data\Doosan_ExpertData\ExpTrjL.mat")['trjExpL']
    L1_Exp = trjExp[0][0][0]
    L2_Exp = trjExp[0][0][1]
    L3_Exp = trjExp[0][0][2]
    application.update()

    # Add a camera manipulator



    for j in range(len(L1_Exp)):
        boom_act.value = -float(L1_Exp[j][0] - BOOM_L)
        arm_act.value = -float(L2_Exp[j][0] - ARM_L)
        bucket_act.value = float(L3_Exp[j][0] - BUCKET_L)

        application.update()
        print(height_output, height_input)
        print(len(height_output))
