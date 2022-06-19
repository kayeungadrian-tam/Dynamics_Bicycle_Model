from ursina import *                    # Import the ursina engine
import numpy as np
from dynamic_bicycle_model import *

from ursina.prefabs.video_recorder import VideoRecorder

import os
import zmq
import struct

print("Connecting to server…")
context = zmq.Context()
socket = context.socket(zmq.REQ)
# socket.connect("tcp://0.0.0.0:5555")
socket.connect(os.environ["server_address"])


def write_csv():
        wb = Workbook()
        ws = wb.active
        ws.title = 'log_data'
        alphabet = list(string.ascii_uppercase)
        columns_label = [
            ('time','[s]'),
            ('x','[m]'),
            ('y','[m]'),
            ('v_x','[km/h]'),
            ('v_y','[km/h]'),
            ('yaw','[rad]'),
            ('beta','[rad]'),
            ('delta','[rad]'),
            ('cte','[m]'),
        ]

        for idx, (k,value) in enumerate(columns_label):
            ws[f'{alphabet[idx]}1'] = k
            ws[f'{alphabet[idx]}2'] = value

        for step in range(len(t)):
            ws[f'A{step+3}'] = f'{t[step]/60:.2f}'
            ws[f'B{step+3}'] = f'{X[step]:.2f}'
            ws[f'C{step+3}'] = f'{Y[step]:.2f}'
            ws[f'D{step+3}'] = f'{v*3.6:.2f}'
            ws[f'E{step+3}'] = f'{v_y[step]*3.6:.2f}'        
            ws[f'F{step+3}'] = f'{yaw[step]:.2f}'
            ws[f'G{step+3}'] = f'{beta[step]:.2f}'
            ws[f'H{step+3}'] = f'{delta[step]:.2f}'
            ws[f'I{step+3}'] = f'{CTE[step]:.2f}'
        
        wb.save(f'{out_path}/00_log-{datetime.date.today()}.xlsx')
        excel = pandas.read_excel(f'{out_path}/00_log-{datetime.date.today()}.xlsx')
        excel.to_csv(f'{out_path}/test.csv')

bike = Vehicle(
    x = 0,
    y = 10,
    v = 30,
    yaw = 0,
    yaw_rate = 0.,
    beta = 0.0
)

pid = PID(P=1.7)

app = Ursina()    
     
from ursina.vec2 import Vec2

window.fullscreen_size = Vec2(800,600)
window.windowed_size = Vec2(800,600)
window.title = 'Dynmaic Bicycle Model'                # The window title
window.borderless = False               # Show a border
window.fullscreen = False               # Do not go Fullscreen
window.exit_button.visible = False      # Do not show the in-game red X that loses the window
window.fps_counter.enabled = True 
window.color = color.rgb(85,79,72)
# window.vsync = False  
window.render_mode = 'wireframe'

for k in range(len(bike.full_track)):
    if k % 1000==0:
        cone = Entity(model=Cone(8, direction=(0,1,0)), color=color.violet , scale=(1,1,1),position=(bike.full_track[k][1],-0.15,bike.full_track[k][0]), texture='brick')

grid_1 = Entity(model=Grid(60,60), scale=600,rotation_x=90, y=-0.15, color=color.color(0.4,0.4,.4), position=(250,-0.15,0), texture='brick')

Text.default_resolution = 1080 * Text.size
direction_sphere = Entity(model='sphere',  scale=(0.5,0.5,0.5), position=(0,0,0), texture='white_cube')
x_pole = Entity(model=Cylinder(30, start=0, radius=0.15, height = 5, direction=(0,0,1)),color=color.red)
y_pole = Entity(model=Cylinder(30, start=0, radius=0.15, height = 5, direction=(1,0,0)),color=color.green)
z_pole = Entity(model=Cylinder(30, start=0, radius=0.15, height = 5, direction=(0,1,0)),color=color.blue)

text_x = Text(parent=x_pole,text="<scale:40><red>X-axis", position=(0,1,3.75), background=True, rotation_y=90)
text_y = Text(parent=y_pole,text="<scale:40><green>Y-axis", position=(1.5,1,0), background=True)
text_z = Text(parent=z_pole,text="<scale:40><blue>Z-axis", position=(-1.5,5.2,0), background=True)

from ursina.prefabs.trail_renderer import TrailRenderer
car = Entity(model='sphere',  scale=(0.1,0.1,0.1), position=(10,0,-0.075), texture='white_cube')
Trail = TrailRenderer(target=car)


text_ori = Text(parent=direction_sphere,text="<scale:80><orange>Origin (0,0,0)", position=(-5,15,0), background=False)
text_car = Text(parent=car,text="<scale:1000><red>Car", position=(-25,40,0), background=False)

front = Entity(parent=car, model=Cylinder(30, start=-0.1, radius=0.5, height = 0.2),  color=color.black, scale=(2,2,2), position=(0,0,1.1), texture="wheel_3.jpg")
front.rotation_z = 90

back = Entity(parent=car, model=Cylinder(30, start=-0.1, radius=0.5, height = 0.2), color=color.black, scale=(2,2,2), position=(0,0,-1.57), texture="shore")
back.rotation_z = 90

axle = Entity(parent=car, model=Cylinder(30, start=-0.1, radius=0.05, height = 2.67/2), color=color.red, scale=(2,2,2), position=(0,0,-1.57), texture="white_cube")
axle.rotation_x = 90

floating = Entity(parent=car, model='sphere',  scale=(0.5,0.5,0.5), position=(0,4,0), texture='white_cube')
point_x = Entity(parent=floating, model=Cone(8, direction=(1,0,0)),  scale=(1,1,1), color=color.red, position=(0,0,0))
point_y = Entity(parent=floating, model=Cone(8, direction=(0,1,0)),  scale=(1,1,1), color=color.green, position=(0,0,0))
point_z = Entity(parent=floating, model=Cone(8, direction=(0,0,1)),  scale=(1,1,1), color=color.blue, position=(0,0,0))

beta_list = [0]
yaw_rate_list = [0]

feedback = [0.]
X, Y, yaw, beta, CTE, delta, v_y, t = [], [], [], [],[], [], [] ,[]

def update(): 
    bike.update(delta=feedback[-1])

    message = struct.pack('ffff', bike.x, bike.y, bike.yaw, bike.delta)
    socket.send(message)
    r_message = socket.recv()



    front.rotation_y = np.degrees(bike.delta)
    front.rotation_y = np.clip(front.rotation_y, -45,45)   

    point_x.rotation_z = np.degrees(bike.yaw)
    point_z.rotation_y = np.degrees(bike.yaw)

    car.z += (np.cos(bike.yaw)*bike.x_dot - np.sin(bike.yaw)*bike.y_dot)*dt
    car.x += (np.sin(bike.yaw)*bike.x_dot + np.cos(bike.yaw)*bike.y_dot)*dt

    car.rotation_y = np.degrees(bike.yaw)
    
    target_value = bike.predict()
    
    feedback.append(feedback[-1] + pid.update(feedback[-1],target_value))

    t.append(bike.count)
    X.append(bike.x)
    Y.append(bike.y)
    v_y.append(bike.v*(bike.beta+bike.yaw))
    yaw.append(bike.yaw)
    beta.append(bike.beta)
    CTE.append(bike.cte)
    delta.append(bike.delta)


    back.rotation_x += 2    
    front.rotation_x += 2    

    if held_keys['left arrow']:                              
        camera.x -=  time.dt*10           
    if held_keys['right arrow']:                               
        camera.x +=  time.dt*10
    if held_keys['up arrow']:                              
        camera.y +=  time.dt*10        
    if held_keys['down arrow']:                               
        camera.y -=  time.dt*10


    if held_keys['w']:                              
        camera.rotation_x -= time.dt*30   
    if held_keys['s']:                               
        camera.rotation_x += time.dt*30   

    if held_keys['a']:                              
        camera.rotation_y -=  time.dt*30
    if held_keys['d']:                               
        camera.rotation_y +=  time.dt*30

def input(key):
    if key == 'c':
        print('Saving log data...')
        write_csv()
        print('log data saved.')
        app.quit
if __name__ == '__main__':
  
    camera.orthographic = False
    camera.fov = 80
    camera.parent = car
    camera.y = 15
    camera.rotation_x = 15
    camera.clip_plane_near
    # EditorCamera()
    video_name = 'Test2'
    # VideoRecorder(recording=True, video_name=video_name, duration = 20, frame_skip=2)
    app.run()

