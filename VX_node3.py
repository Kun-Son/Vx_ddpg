
from VxSim import *
import zmq
import numpy as np
from x2q2cyl import X2Q2Cyl
from math import *
import time
import datetime
from scipy import io
from scipy.signal import butter
import pyvx

SCENE_DIR = "C:\CM Labs\Vortex Studio Samples 2018b\Scenario\Excavator Scene\Excavator_basic.vxscene"
PORT = "5555"

BOOM_INIT = 2.7070
ARM_INIT = 2.5891
BUCKET_INIT = 2.0081

BOOM_L = 3.0632
ARM_L = 3.7442
BUCKET_L = 1.7906

#BOOM_MAX = 0.746
#BOOM_MIN = -0.5
#ARM_MAX=1.36
ARM_MIN=-0.108
BUCKET_MAX=1.164
BUCKET_MIN=0.0

BOOM_MAX = 0.3542
ARM_MAX = 1.1519
BUCKET_MAX = 0.4737
BOUND_OFF = 0.1

CABBIN_THRESHOLD = 0.0002

MAX_STEPS_EPISODE = 349

# Load vortex scene file
def loader():
    application = VxApplication()
    graphicsModule = VxExtensionFactory.create(GraphicsModuleICD.kModuleFactoryKey)
    application.insertModule(graphicsModule)

    fileManager = application.getSimulationFileManager()
    imp_file = SCENE_DIR
    scene = SceneInterface(fileManager.loadObject(imp_file))

    return application, scene, fileManager

def reloader(file):
    file.clean()
    imp_file = SCENE_DIR
    scene = SceneInterface(file.loadObject(imp_file))

    return scene

# Access to Vortex studio parameters
def parameter(scene):
    excavator_mechanism = [m for m in scene.getMechanisms() if m.getName() == 'Excavator'][0]
    excavator_assembly = [a for a in excavator_mechanism.getAssemblies() if a.getName() == 'Excavator'][0]
    excavator_constraint = excavator_assembly.getConstraints()
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

    turret_part = scene.findExtensionByName("Turret")
    turret_container = turret_part.getOutputContainer()
    turret_pos = turret_container[0]

    return boom_pos, arm_pos, bucket_pos, payload, turret_pos

def render(application):
    # Create a display window
    display = VxExtensionFactory.create(DisplayICD.kExtensionFactoryKey)
    display.getInput(DisplayICD.kPlacementMode).setValue("Windowed")
    display.getInput(DisplayICD.kPlacement).setValue(VxVector4(50, 50, 360, 240))
    application.add(display)

    dynamicsModule = VxExtensionFactory.create(VxDynamicsModuleICD.kFactoryKey)
    application.insertModule(dynamicsModule)

    # Add the DynamicsVisualizer (physics objects visualization)
    dynamicsVisualizer = VxExtensionFactory.create(DynamicsVisualizerICD.kExtensionFactoryKey)
    application.add(dynamicsVisualizer)

def zmq_socket():

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % PORT)
    print("Vortex studio socket open..")

    return socket


def reset(boom_act, arm_act, bucket_act):

    for i in range(100):

        boom_act.value = -(BOOM_INIT - BOOM_L)
        arm_act.value = -(ARM_INIT - ARM_L)
        bucket_act.value = (BUCKET_INIT-BUCKET_L)
        application.update()

def step(boom_act, arm_act, bucket_act, payload, action):

    boom_act.value = -(BOOM_INIT + action[0] - BOOM_L)
    arm_act.value = -(ARM_INIT + action[1] - ARM_L)
    bucket_act.value = (BUCKET_INIT + action[2] -BUCKET_L)

    application.update()

    terminal = 0

    reward1 = payload.value

    return reward1, terminal

def lpfilter(inputSignal, outputSignal, timeConstant):
    deltaTime = pyvx.frame.timestep
    value = ((deltaTime * inputSignal) + (timeConstant * outputSignal)) / (deltaTime + timeConstant)

    return value

    #filtered_value = (cutoff*u + timeconst*y)/(timeconst+cutoff)
    #filtered_value = ((cutoff * input) + (timeconstant * output)) / (cutoff + timeconstant)


if __name__ == '__main__':

    cycle_time=time.time()
    application, scene, file = loader()
    boom_act, arm_act, bucket_act, payload, turret = parameter(scene)
    socket = zmq_socket()
    x2q2c=X2Q2Cyl()
    prep_time = time.time()

    action_input = [.0,.0,.0]
    turret_prev = turret.value[1][2]

    trjExp = io.loadmat("D:\Doosan_simul\Excavator Data\Doosan_ExpertData\ExpTrjL.mat")['trjExpL']
    L1_Exp = trjExp[0][0][0]
    L2_Exp = trjExp[0][0][1]
    L3_Exp = trjExp[0][0][2]

    ActionExp = io.loadmat("D:\Doosan_simul\Excavator Data\Doosan_ExpertData\AExpL.mat")['aExpL']
    a1_Exp = ActionExp[0][0][0]
    a2_Exp = ActionExp[0][0][1]
    a3_Exp = ActionExp[0][0][2]


    keyFramesMgr = application.getContext().getKeyFrameManager()
    keyFrameList = keyFramesMgr.createKeyFrameList("MyList", False)

    keyFrameId = keyFrameList.saveKeyFrame()

    while True:
        action = socket.recv()

        if action.split()[0] == 'start':

            print("Start signal received")
            socket.send("Successfully connected")
            render(application)
            reset(boom_act, arm_act, bucket_act)
            keyFrameId = keyFrameList.saveKeyFrame()

        elif action.split()[0] == 'reset':
            print("Cycle time : %f" % (time.time() - cycle_time))
            cycle_time = time.time()
            print ("Scene reset")
            keyFrameList.restore(KeyFrame(keyFrameId))
            socket.send("%f %f %f %f %f " %(L1_Exp[0][0], L2_Exp[0][0], L3_Exp[0][0], 0, 0))
            action_input = [0,0,0]
            f = open("D:\Doosan_simul\\20181117\Raw\{}.txt".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), 'w')
            f.write("%s\n\n" %action.split()[1])
            reward5 = 0.
            prev_z = 0.
            count = 0

        elif action.split()[0] == 'step':

            new_input = np.array(action.split()[1:4]).astype(np.float)
            step_num = int(np.array(action.split()[4]).astype(np.float))

            action_input[0] = action_input[0] + new_input[0] * 0.004
            action_input[1] = action_input[1] + new_input[1] * 0.006
            action_input[2] = action_input[2] + new_input[2] * 0.006

            if action_input[0] > BOOM_MAX + BOUND_OFF:
                action_input[0] = BOOM_MAX + BOUND_OFF
            if action_input[1] > ARM_MAX + BOUND_OFF:
                action_input[1] = ARM_MAX + BOUND_OFF
            if action_input[2] > BUCKET_MAX + BOUND_OFF:
                action_input[2] = BUCKET_MAX + BOUND_OFF

            reward1, terminal = step(boom_act, arm_act, bucket_act, payload, action_input)


            reward2 = float(3 * (BOOM_INIT + action_input[0] - float(L1_Exp[step_num][0])) ** 2 \
                            + 1.0 * (ARM_INIT + action_input[1] - float(L2_Exp[step_num][0])) ** 2 \
                            + 1.75 * (BUCKET_INIT + action_input[2] - float(L3_Exp[step_num][0])) ** 2)

            reward3 = float((new_input[0] * 0.004 - float(a1_Exp[step_num][0])) ** 2 \
                            + (new_input[1] * 0.006 - float(a2_Exp[step_num][0])) ** 2 \
                            + (new_input[2] * 0.006 - float(a3_Exp[step_num][0])) ** 2)




            '''
            reward2 = exp(- 10*float(3 * (BOOM_INIT + action_input[0] - float(L1_Exp[step_num][0])) ** 2 \
                                     + 1.0 * (ARM_INIT + action_input[1] - float(L2_Exp[step_num][0])) ** 2 \
                                     + 1.75 * (BUCKET_INIT + action_input[2] - float(L3_Exp[step_num][0])) ** 2))

            reward3 = exp(- 20000*float((new_input[0] * 0.004 - float(a1_Exp[step_num][0])) ** 2 \
                                       + (new_input[1] * 0.006 - float(a2_Exp[step_num][0])) ** 2 \
                                       + (new_input[2] * 0.006 - float(a3_Exp[step_num][0])) ** 2))
            '''

            #reward = - 3.0 * reward2 - 5000 * reward3
            reward = - 50000 * reward3
            '''
            if step_num>300:
                reward = reward1 / 4000.0 - 5.0 * reward2 - 20000 * reward3
            else:
                reward = reward2 + reward3
            '''

            if step_num % 10 == 0 :
               print ("%f %0.1f %0.3f %0.3f %0.6f" %(float(step_num)/349.0, reward, 0.00005, 3*reward2, 50000*reward3))
            socket.send("%f %f %f %f %f %f %f %r" % (
            boom_act.value, arm_act.value, bucket_act.value, payload.value, float(step_num)/349.0, reward, reward1, terminal))


            f.write("%d %f %f %f %f %f %f %f\n" %(step_num, boom_act.value, arm_act.value,  bucket_act.value, reward, reward1, reward2, reward3))









