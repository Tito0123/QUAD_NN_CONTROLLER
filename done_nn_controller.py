"""
Author: Daniel Ingram (daniel-s-ingram)
"""
import sys
from math import cos, sin
import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from torch._C import JITException
import csv
show_animation = False
 
class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1):
        super(TwoLayerNet, self).__init__()
        # self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, 1)
        self.control1 = torch.nn.Linear(D_in,H1)
        self.dropout = torch.nn.Dropout(0.2)
        self.control2 = torch.nn.Linear(H1,4)
 
    def forward(self,x):
        # h1 = torch.nn.functional.relu(self.linear1(x))
        # # h2 = torch.nn.functional.relu(self.linear2(h1))
        # y = self.linear2(h1)
 
        h2 = torch.relu(self.control1(x))
        h2 = self.dropout(h2)
        u = self.control2(h2)
        return u
 
 
"""
Class for plotting a quadrotor
 
"""
 
 
class Quadrotor():
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T
 
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.show_animation = show_animation
 
        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
 
            self.ax = fig.add_subplot(111, projection='3d')
 
        self.update_pose(x, y, z, roll, pitch, yaw)
 
    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
 
        if self.show_animation:
            self.plot()
 
    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])
 
    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()
 
        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)
 
        plt.cla()
 
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')
 
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')
 
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')
 
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        self.ax.set_zlim(0, 5)
 
        plt.pause(0.001)
 
"""
Generates a quintic polynomial trajectory.
"""
 
class TrajectoryGenerator():
    def __init__(self, start_pos, des_pos, T, start_vel=[0,0,0], des_vel=[0,0,0], start_acc=[0,0,0], des_acc=[0,0,0]):
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.start_z = start_pos[2]
 
        self.des_x = des_pos[0]
        self.des_y = des_pos[1]
        self.des_z = des_pos[2]
 
        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]
 
        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]
 
        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]
 
        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]
 
        self.T = T
 
    def solve(self):
        A = np.array(
            [[0, 0, 0, 0, 0, 1],
             [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1],
             [0, 0, 0, 0, 1, 0],
             [5*self.T**4, 4*self.T**3, 3*self.T**2, 2*self.T, 1, 0],
             [0, 0, 0, 2, 0, 0],
             [20*self.T**3, 12*self.T**2, 6*self.T, 2, 0, 0]
            ])
 
        b_x = np.array(
            [[self.start_x],
             [self.des_x],
             [self.start_x_vel],
             [self.des_x_vel],
             [self.start_x_acc],
             [self.des_x_acc]
            ])
 
        b_y = np.array(
            [[self.start_y],
             [self.des_y],
             [self.start_y_vel],
             [self.des_y_vel],
             [self.start_y_acc],
             [self.des_y_acc]
            ])
 
        b_z = np.array(
            [[self.start_z],
             [self.des_z],
             [self.start_z_vel],
             [self.des_z_vel],
             [self.start_z_acc],
             [self.des_z_acc]
            ])
 
        self.x_c = np.linalg.solve(A, b_x)
        self.y_c = np.linalg.solve(A, b_y)
        self.z_c = np.linalg.solve(A, b_z)
 
 
 
 
"""
Simulate a quadrotor following a 3D trajectory
"""
waypoints = [[1, 0, 1], [0, 5, 0], [5, 5, 5], [1, 0, 1], [2,5,7]]
 
# Simulation parameters
g = 9.81
m = 0.2
Ixx = 1
Iyy = 1
Izz = 1
T = 15
dt = 0.1
 
# Proportional coefficients
Kp_x = 1
Kp_y = 1
Kp_z = 1
Kp_roll = 25
Kp_pitch = 25
Kp_yaw = 25
 
# Derivative coefficients
Kd_x = 10
Kd_y = 10
Kd_z = 1
 
epoch_list = []
loss_list = []  
 
desired_x_pos = []
desired_y_pos = []
desired_z_pos = []
 
desired_x_acc = []
desired_y_acc = []
desired_z_acc = []
 
actual_x_acc = []
actual_y_acc = []
actual_z_acc = []
 
actual_x_pos = []
actual_y_pos = []
actual_z_pos = []
 
actual_x_vel = []
actual_y_vel = []
actual_z_vel = []
 
actual_roll_vel = []
actual_pitch_vel = []
actual_yaw_vel = []
 
actual_roll = []
actual_pitch = []
actual_yaw = []
 
thrust_array = []
roll_torque_array = []
pitch_torque_array = []
 
yaw_torque_array = []
 
model = TwoLayerNet(3,100)
 
item_matrix = np.array([])
item_matrix = np.append(item_matrix, [m, g, Ixx, Iyy, Izz, dt])
 
 
def quad_sim(x_c, y_c, z_c):
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c.
    """
    x_pos = waypoints[0][0]
    y_pos = waypoints[0][1]
    z_pos = waypoints[0][2]
 
    x_vel = 0
    y_vel = 0
    z_vel = 0
    x_acc = 0
    y_acc = 0
    z_acc = 0
    roll  = 0
    pitch = 0
    yaw = 0
    roll_vel = 0
    pitch_vel = 0
    yaw_vel = 0
    des_yaw = 0
 
    t = 0
 
    q = Quadrotor(x=x_pos, y=y_pos, z=z_pos, roll=roll, pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)
 
    i = 0
    n_run = len(waypoints) - 1
    irun = 0
 
    while True:
        while t <= T:
           
            des_x_pos = calculate_position(x_c[i], t)
            des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)
           
            des_x_vel = calculate_velocity(x_c[i], t)
            des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)
            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)
           
            thrust = m * (g + des_z_acc + Kp_z * (des_z_pos - z_pos) + Kd_z * (des_z_vel - z_vel))
            roll_torque = Kp_roll * (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll)
            pitch_torque = Kp_pitch * (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch)
            yaw_torque = Kp_yaw * (des_yaw - yaw)
 
            roll_vel = roll_vel + roll_torque * dt / Ixx
            pitch_vel = pitch_vel + pitch_torque * dt / Iyy
            yaw_vel = yaw_vel + yaw_torque * dt / Izz
             
            actual_roll_vel.append(roll_vel)
            actual_pitch_vel.append(pitch_vel)
            actual_yaw_vel.append(yaw_vel)
 
            roll = roll + roll_vel * dt
            pitch = pitch + pitch_vel * dt
            yaw = yaw + yaw_vel * dt
 
            actual_roll.append(roll)
            actual_pitch.append(pitch)
            actual_yaw.append(yaw)
 
            R = rotation_matrix(roll, pitch, yaw)
            acc = (np.matmul(R, np.array([0, 0, thrust.item()]).T) - np.array([0, 0, m * g]).T) / m
            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = acc[2]
 
            x_vel = x_vel + x_acc * dt
            y_vel = y_vel + y_acc * dt
            z_vel = z_vel + z_acc * dt
 
            x_pos = x_pos + x_vel * dt
            y_pos = y_pos + y_vel * dt
            z_pos = z_pos + z_vel * dt
           
            q.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)
 
            desired_x_pos.append(des_x_pos)
            desired_y_pos.append(des_y_pos)
            desired_z_pos.append(des_z_pos)
 
            desired_x_acc.append(des_x_acc)
            desired_y_acc.append(des_y_acc)
            desired_z_acc.append(des_z_acc)
 
            actual_x_pos.append(x_pos)
            actual_y_pos.append(y_pos)
            actual_z_pos.append(z_pos)
 
            actual_x_acc.append(x_acc)
            actual_y_acc.append(y_acc)
            actual_z_acc.append(z_acc)
 
            actual_x_vel.append(x_vel)
            actual_y_vel.append(y_vel)
            actual_z_vel.append(z_vel)
 
            thrust_array.append(thrust)
            roll_torque_array.append(roll_torque)
            pitch_torque_array.append(pitch_torque)
            yaw_torque_array.append(yaw_torque)
 
            t += dt
         
        t = 0
        i = (i + 1)
       
        irun += 1
        if irun >= n_run:
            break
    print("Done")
   
    np.set_printoptions(threshold = sys.maxsize )

    x_pos_d = np.array([])
    y_pos_d = np.array([])
    z_pos_d = np.array([])

    x_acc_d = np.array([])  
    y_acc_d = np.array([])
    z_acc_d = np.array([])

    x_pos_act = np.array([])
    y_pos_act = np.array([])
    z_pos_act = np.array([])

    x_vel_act = np.array([])  
    y_vel_act = np.array([])
    z_vel_act = np.array([])

    roll_vel_act = np.array([])  
    pitch_vel_act = np.array([])
    yaw_vel_act = np.array([])

    roll_act = np.array([])  
    pitch_act = np.array([])
    yaw_act = np.array([])




   
    #training
    for i in range(len(actual_x_vel)):
        roll_act = np.append(roll_act, actual_roll[i])
        pitch_act = np.append(pitch_act, actual_pitch[i])
        yaw_act = np.append(yaw_act, actual_yaw[i])

        roll_vel_act = np.append(roll_vel_act, actual_roll_vel[i])  
        pitch_vel_act = np.append(pitch_vel_act, actual_pitch_vel[i])  
        yaw_vel_act = np.append(yaw_vel_act, actual_yaw_vel[i])  

        x_vel_act = np.append(x_vel_act, actual_x_vel[i])  
        y_vel_act = np.append(y_vel_act, actual_y_vel[i])
        z_vel_act = np.append(z_vel_act, actual_z_vel[i])

        x_pos_act = np.append(x_pos_act, actual_x_pos[i])
        y_pos_act = np.append(y_pos_act, actual_y_pos[i])
        z_pos_act = np.append(z_pos_act, actual_z_pos[i])

        x_pos_d = np.append(x_pos_d, desired_x_pos[i])
        y_pos_d = np.append(y_pos_d, desired_y_pos[i])
        z_pos_d = np.append(z_pos_d, desired_x_pos[i])

    item_mat_tensor = torch.FloatTensor(item_matrix)
 
    Ixx_tensor = item_mat_tensor[2]
    Iyy_tensor = item_mat_tensor[3]
    Izz_tensor = item_mat_tensor[4]
 
    m_tensor = item_mat_tensor[0]
    g_tensor = item_mat_tensor[1]
 
    dt_tensor = item_mat_tensor[5]

    x_pos_d = x_pos_d .T

    x_pos_tensor = torch.FloatTensor(x_pos_act)
    y_pos_tensor = torch.FloatTensor(y_pos_act)
    z_pos_tensor = torch.FloatTensor(z_pos_act)
 
    x_vel_tensor = torch.FloatTensor(x_vel_act)
    y_vel_tensor = torch.FloatTensor(y_vel_act)
    z_vel_tensor = torch.FloatTensor(z_vel_act)
 
    roll_tensor = torch.FloatTensor(roll_act)
    pitch_tensor = torch.FloatTensor(pitch_act)
    yaw_tensor = torch.FloatTensor(yaw_act)
 
    roll_vel_tensor = torch.FloatTensor(roll_vel_act)
    pitch_vel_tensor = torch.FloatTensor(pitch_vel_act)
    yaw_vel_tensor = torch.FloatTensor(yaw_vel_act)



    print(x_pos_d)
    print(np.shape(x_pos_d))
    print(np.shape(z_pos_tensor))

    print(np.shape(x_vel_tensor))
    print(np.shape(y_vel_tensor))
    print(np.shape(z_vel_tensor))

    print(np.shape(roll_tensor))
    print(np.shape(pitch_tensor))
    print(np.shape(yaw_tensor))

    print(np.shape(roll_vel_tensor))
    print(np.shape(pitch_vel_tensor))
    print(np.shape(yaw_vel_tensor))

    #roll_vel_tensor = torch.randn(604,1)
    #pitch_vel_tensor = torch.randn(604, 1)
    #yaw_vel_tensor = torch.randn(604, 1)

    #roll_tensor = torch.randn(604,1)
    #pitch_tensor = torch.randn(604, 1)
    #yaw_tensor = torch.randn(604, 1)

    #x_pos_tensor = torch.randn(604,1)
    #y_pos_tensor = torch.randn(604,1)
    #z_pos_tensor = torch.randn(604,1)
 
    #x_vel_tensor = torch.randn(604,1)
    #y_vel_tensor = torch.randn(604,1)
    #z_vel_tensor = torch.randn(604,1)


    des_x_tensor = torch.FloatTensor(x_pos_d)
    des_y_tensor = torch.FloatTensor(y_pos_d)
    des_z_tensor = torch.FloatTensor(z_pos_d)
 
    input_x_data = x_pos_d - x_pos_act
    input_x_data = input_x_data.T
    input_y_data = y_pos_d - y_pos_act
    input_y_data = input_y_data.T
    input_z_data = y_pos_d - y_pos_act
    input_z_data = input_z_data.T
    input_data   = [input_x_data, input_y_data, input_z_data]
    input_data_tensor = torch.FloatTensor(input_data.T)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)
   
    for j in range(10000):
   
     control_tensor = model(input_data_tensor)
     thrust_nn = torch.clamp(control_tensor[:,0], 0,2)
     roll_torque_nn = torch.clamp(control_tensor[:,1], -1, 1)
     pitch_torque_nn = torch.clamp(control_tensor[:,2], -1, 1)
     yaw_torque_nn = torch.clamp(control_tensor[:,3], -1, 1)

     for i in range(len(thrust_nn)):
      roll_vel_tensor = roll_vel_tensor + roll_torque_nn[i] * dt_tensor / Ixx_tensor
      pitch_vel_tensor = pitch_vel_tensor + pitch_torque_nn[i] * dt_tensor / Iyy_tensor
      yaw_vel_tensor = yaw_vel_tensor + yaw_torque_nn[i] * dt_tensor / Izz_tensor
   
      roll_tensor = roll_tensor + roll_vel_tensor * dt_tensor
      pitch_tensor = pitch_tensor + pitch_vel_tensor * dt_tensor
      yaw_tensor = yaw_tensor + yaw_vel_tensor * dt_tensor
 
      x_acc_tensor = torch.mul(torch.mul( torch.sin(yaw_tensor) , torch.sin(roll_tensor)) + torch.mul(torch.mul(torch.cos(yaw_tensor) , torch.sin(pitch_tensor)) , torch.cos(roll_tensor)), thrust_nn[i])/m_tensor
      y_acc_tensor = torch.mul(torch.mul(-torch.cos(yaw_tensor) , torch.sin(roll_tensor)) + torch.mul(torch.mul(torch.sin(yaw_tensor) , torch.sin(pitch_tensor)) , torch.cos(roll_tensor)), thrust_nn[i])/m_tensor
      z_acc_tensor = torch.subtract(torch.mul(torch.mul(torch.cos(pitch_tensor) , torch.cos(yaw_tensor)) , thrust_nn[i]) , m*g)/m_tensor
 
      x_vel_tensor = x_vel_tensor + x_acc_tensor * dt_tensor
      y_vel_tensor = y_vel_tensor + y_acc_tensor * dt_tensor
      z_vel_tensor = z_vel_tensor + z_acc_tensor * dt_tensor
 
      x_pos_tensor = x_pos_tensor + x_vel_tensor * dt_tensor
      y_pos_tensor = y_pos_tensor + y_vel_tensor * dt_tensor
      z_pos_tensor = z_pos_tensor + z_vel_tensor * dt_tensor

      new_x_pos = np.append(new_x_pos, x_pos_tensor)
      new_y_pos = np.append(new_y_pos, y_pos_tensor)
      new_z_pos = np.append(new_z_pos, z_pos_tensor)
     
     new_x_pos_tensor = torch.FloatTensor(new_x_pos)

     errorx = 0.5 * (des_x_tensor - new_x_pos_tensor)**2
     errory = 0.5 * (des_y_tensor - new_y_pos_tensor)**2
     errorz = 0.5 * (des_z_tensor - new_z_pos_tensor)**2
     
     error =  errorx + errory + errorz
     
     loss = error.mean()
     if j%10== 0:
       print(j,loss.item(),errorx.mean().item(),errory.mean().item(),errorz.mean().item())
 
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     scheduler.step()
     epoch_list.append(j)
     loss_list.append(loss.item())  
 
    #f = open('data.csv', 'w')
    #writer = csv.writer(f)
    #writer.writerows(input_data)
    #f.close()
 
    plt.plot(epoch_list, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.show()
 
desired1_x_pos = []
desired1_y_pos = []
desired1_z_pos = []
 
def data (x_c, y_c, z_c, waypoint):
    t = 0
    i = 0
    n_run = len(waypoint) - 1
    irun = 0
   
    while True:
        while t <= T:
           
            des_x_pos = calculate_position(x_c[i], t)
            des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)
            des_x_vel = calculate_velocity(x_c[i], t)
            des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)
            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)
           
            t += dt
           
           
            desired1_x_pos.append(des_x_pos)
            desired1_y_pos.append(des_y_pos)
            desired1_z_pos.append(des_z_pos)
         
        t = 0
        i = (i + 1)
        irun += 1
        if irun >= n_run:
            break
    print("Done")
    return(desired1_x_pos,desired1_y_pos,desired1_z_pos )
   
 
 
def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.
 
    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the position
 
    Returns
        Position
    """
    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]
 
 
def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.
 
    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the velocity
 
    Returns
        Velocity
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]
 
 
def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.
 
    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the acceleration
 
    Returns
        Acceleration
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]
 
 
 
def rotation_matrix(roll, pitch, yaw):
    """
    Calculates the ZYX rotation matrix.
    Args
        Roll: Angular position about the x-axis in radians.
        Pitch: Angular position about the y-axis in radians.
        Yaw: Angular position about the z-axis in radians.
    Returns
        3x3 rotation matrix as NumPy array
    """
    return np.array(
        [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
         [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) *
          sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
         [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
         ])
 
 
def main():
    """
    Calculates the x, y, z coefficients for the four segments
    of the trajectory
    """
    x_coeffs = [[], [], [], []]
    y_coeffs = [[], [], [], []]
    z_coeffs = [[], [], [], []]
 
    x_coeffs1 = [[], []]
    y_coeffs1 = [[], []]
    z_coeffs1 = [[], []]
   
 
    for i in range(len(waypoints) - 1):
        traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1)], T)
        traj.solve()
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c
   
    quad_sim(x_coeffs, y_coeffs, z_coeffs)
 
    a = [[0, 1, 0], [2,5,-5], [4,3,2]]
    for j in range(len(a) - 1):
        traj2 = TrajectoryGenerator(a[j], a[(j + 1)], T)
        traj2.solve()
        x_coeffs1[j] = traj2.x_c
        y_coeffs1[j] = traj2.y_c
        z_coeffs1[j] = traj2.z_c
 
    x1,y1,z1 = data(x_coeffs1, y_coeffs1, z_coeffs1, a)
    x_l2a = np.array([])
    y_l2a = np.array([])
    z_l2a = np.array([])
 
    for i in range(len(x1)):
        x_l2a = np.append(x_l2a, x1[i])
        y_l2a = np.append(y_l2a, y1[i])
        z_l2a = np.append(z_l2a, z1[i])
   
    x_pos = 0
    y_pos = 1
    z_pos = 0
 
    x_vel = 0
    y_vel = 0
    z_vel = 0
 
    x_acc = 0
    y_acc = 0
    z_acc = 0
 
    roll  = 0
    pitch = 0
    yaw = 0
 
    roll_vel = 0
    pitch_vel = 0
    yaw_vel = 0
 
    des_yaw = 0
 
    x = []
    y = []
    z = []
 
    for p in range(len(x_l2a)):
     error_x =  (x_l2a[p] - x_pos)**2
     error_y =  (y_l2a[p] - y_pos)**2
     error_z =  (z_l2a[p] - z_pos)**2
     
     
 
     input  = torch.FloatTensor([error_x,error_y,error_z])
     u = model(input.T)
     thrust = u[0]
     roll_torque = u[1]
     pitch_torque = u[2]
     yaw_torque = u[3]
     
     thrust = thrust.item()
     roll_torque = roll_torque.item()
     pitch_torque = pitch_torque.item()
     yaw_torque  = yaw_torque.item()
     
     roll_vel = roll_vel + roll_torque * dt / Ixx
     pitch_vel = pitch_vel + pitch_torque * dt / Iyy
     yaw_vel = yaw_vel + yaw_torque * dt / Izz
             
     roll = roll + roll_vel * dt
     pitch = pitch + pitch_vel * dt
     yaw = yaw + yaw_vel * dt
 
     R = rotation_matrix(roll, pitch, yaw)
     acc = (np.matmul(R, np.array([0, 0, thrust]).T) - np.array([0, 0, m * g]).T) / m
     x_acc = acc[0]
     y_acc = acc[1]
     z_acc = acc[2]
 
     x_vel = x_vel + x_acc * dt
     y_vel = y_vel + y_acc * dt
     z_vel = z_vel + z_acc * dt
 
     x_pos = x_pos + x_vel * dt
     y_pos = y_pos + y_vel * dt
     z_pos = z_pos + z_vel * dt
 
     x.append(x_pos)
     y.append(y_pos)
     z.append(z_pos)    
     error_pos = ((x_l2a[p] - x_pos)**2 + ( y_l2a[p] - y_pos)**2 + (z_l2a[p] - z_pos)**2)
   
     print(p,x_pos, y_pos, z_pos, error_x,error_y,error_z,thrust,roll_torque,  pitch_torque , yaw_torque)
 
   
 
 
 
if __name__ == "__main__":
    main()
 

