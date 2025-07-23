fish
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib.patches import Polygon

parameters = [40 - 15j, 25 + 20j, 20 - 8j, 15 + 12j, 0 + 0j]

def fourier(t, C):
    f = np.zeros(t.shape)
    for k in range(len(C)):
        f += C.real[k] * np.cos(k * t) + C.imag[k] * np.sin(k * t)
    return f
def fish(t, p):
    npar = 6
    Cx = np.zeros((npar,), dtype='complex')
    Cy = np.zeros((npar,), dtype='complex')

    Cy[1] = p[0].real * 1j
    Cx[1] = p[3].imag + p[0].imag * 1j
    Cy[2] = p[1].real * 1j
    Cx[2] = p[1].imag * 1j
    Cy[3] = p[2].real
    Cx[3] = p[2].imag * 1j
    Cy[5] = p[3].real

    x = fourier(t, Cx)
    y = fourier(t, Cy)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    return x, -y

def init_plot():
    t = np.linspace(0, 2 * np.pi, 100)  
    x, y = fish(t, parameters)
  
    fish_body.set_xy(np.column_stack((x, y)))

    eye_x = np.mean(x) + 10
    eye_y = np.mean(y)
    eye.set_data([eye_x], [eye_y])

    return fish_body, eye

def move_tail(i):
    t = np.linspace(0, 2 * np.pi, 100)
    x, y = fish(t, parameters)

    tail_start = 70  
    tail_end = 99   

    tail_amplitude = 15 * np.sin(i * 0.1) 

    for j in range(tail_start, tail_end):
    
        dist_from_base = (j - tail_start) / (tail_end - tail_start)
    
        y[j] += tail_amplitude * dist_from_base * np.sin(2 * dist_from_base)


    x[-1] = x[0]
    y[-1] = y[0]


    fish_body.set_xy(np.column_stack((x, y)))


    eye_x = np.mean(x) + 10
    eye_y = np.mean(y)
    eye.set_data([eye_x], [eye_y])

    return fish_body, eye


fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)


ax.set_facecolor('#e0f7fa')


fish_body = Polygon(np.zeros((100, 2)), closed=True,
                   facecolor='darkblue', edgecolor='navy', linewidth=2, zorder=10)
ax.add_patch(fish_body)


eye, = ax.plot([], [], 'ko', markersize=8, zorder=11)


ax.set_xlim([-70, 70])
ax.set_ylim([-70, 70])
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('封闭图形的游鱼动画', fontsize=18, fontweight='bold', color='navy')


ax.grid(True, linestyle='--', alpha=0.3, color='gray')


ani = animation.FuncAnimation(
    fig=fig,
    func=move_tail,
    frames=200,
    init_func=init_plot,
    interval=30,
    blit=True,
    repeat=True
)


plt.tight_layout()
plt.show()

dog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib.patches import Polygon, Circle

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
ax.set_facecolor('#fff8e1')
ax.set_xlim([-60, 60])
ax.set_ylim([-50, 50])
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('会抖耳朵的小狗', fontsize=20, fontweight='bold', color='saddlebrown')


def get_dog_shape():

    head_x = np.array([-30, -35, -40, -35, -20, 0, 20, 35, 40, 35, 30, 0])
    head_y = np.array([0, 15, 5, -10, -20, -25, -20, -10, 5, 15, 0, 5])

    left_ear_x = np.array([-35, -45, -50, -45, -35])
    left_ear_y = np.array([10, 20, 15, 10, 10])

    right_ear_x = np.array([35, 45, 50, 45, 35])
    right_ear_y = np.array([10, 10, 15, 20, 10])

    left_eye = [-15, 5]
    right_eye = [15, 5]

    nose = [0, 1]

    return head_x, head_y, left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_eye, right_eye, nose

head_x, head_y, left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_eye, right_eye, nose = get_dog_shape()

head = Polygon(np.column_stack((head_x, head_y)), closed=True,
              facecolor='sienna', edgecolor='saddlebrown', linewidth=2)
ax.add_patch(head)


left_ear = Polygon(np.column_stack((left_ear_x, left_ear_y)), closed=True,
                  facecolor='sienna', edgecolor='saddlebrown', linewidth=2)
ax.add_patch(left_ear)


right_ear = Polygon(np.column_stack((right_ear_x, right_ear_y)), closed=True,
                   facecolor='sienna', edgecolor='saddlebrown', linewidth=2)
ax.add_patch(right_ear)


nose_circle = Circle((nose[0], nose[1]), radius=1,
                    facecolor='black', edgecolor='saddlebrown', linewidth=1)
ax.add_patch(nose_circle)


def init_plot():
  
    _, _, l_ear_x, l_ear_y, r_ear_x, r_ear_y, _, _, _ = get_dog_shape()


    left_ear.set_xy(np.column_stack((l_ear_x, l_ear_y)))
    right_ear.set_xy(np.column_stack((r_ear_x, r_ear_y)))

    return head, left_ear, right_ear, nose_circle, left_eye_circle, right_eye_circle


def move_ears(i):
   
    _, _, base_l_ear_x, base_l_ear_y, base_r_ear_x, base_r_ear_y, _, _, _ = get_dog_shape()

 
    ear_amplitude = 5 * np.sin(i * 0.2) 

   
    l_ear_x = base_l_ear_x.copy()
    l_ear_y = base_l_ear_y.copy() + ear_amplitude

   
    r_ear_x = base_r_ear_x.copy()
    r_ear_y = base_r_ear_y.copy() + ear_amplitude * np.cos(i * 0.15)

  
    left_ear.set_xy(np.column_stack((l_ear_x, l_ear_y)))
    right_ear.set_xy(np.column_stack((r_ear_x, r_ear_y)))


    head_rotation = 0.5 * np.sin(i * 0.1)
    head.set_xy(np.column_stack((
        head_x * np.cos(head_rotation) - head_y * np.sin(head_rotation),
        head_x * np.sin(head_rotation) + head_y * np.cos(head_rotation)
    )))

    return head, left_ear, right_ear, nose_circle, left_eye_circle, right_eye_circle


ani = animation.FuncAnimation(
    fig=fig,
    func=move_ears,
    frames=200,
    init_func=init_plot,
    interval=50,
    blit=True,
    repeat=True
)


plt.tight_layout()
plt.show()


swan
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation
from numpy import append, cos, linspace, pi, sin, zeros
import matplotlib.pyplot as plt
from IPython.display import HTML


parameters = [1 - 2j, 9 + 9j, 1 - 2j, 9 + 9j, 0 + 0j]

# Fourier coordinate expansion
def fourier(t, C):
    f = zeros(t.shape)
    for k in range(len(C)):
        f += C.real[k] * cos(k * t) + C.imag[k] * sin(k * t)
    return f


def swan(t, p):
    npar = 6
    Cx = zeros((npar,), dtype='complex')
    Cy = zeros((npar,), dtype='complex')

    Cy[1] = p[0].real * 1j
    Cx[1] = p[3].imag + p[0].imag * 1j
    Cy[2] = p[1].real * 1j
    Cx[2] = p[1].imag * 1j
    Cy[3] = p[2].real
    Cx[3] = p[2].imag * 1j
    Cy[5] = p[3].real

    x = fourier(t, Cx)
    y = fourier(t, Cy)
    return x, -y


def init_plot():
 
    x, y = swan(linspace(0, 2 * pi, 1000), parameters)
    body.set_data(x, y)

   
    wing_x, wing_y = get_wing_position(0)
    wing.set_data(wing_x, wing_y)

 
    eye_x, eye_y = get_eye_position()
    eye.set_data(eye_x, eye_y)

    return body, wing, eye, 


def get_wing_position(i):
  
    t_values = linspace(0, 2 * pi, 1000)
    x, y = swan(t_values, parameters)

   
    start_idx = 300 
    end_idx = 400    
    wing_indices = range(start_idx, end_idx)

    wing_x = []
    wing_y = []
    for idx in wing_indices:
        
        flap = sin(i * 0.2) * 1.5  
        wing_x.append(x[idx])
        wing_y.append(y[idx] + flap)

    return wing_x, wing_y

def get_eye_position():
  
    eye_x = 7.0  # x坐标
    eye_y = 3.5  # y坐标
    return [eye_x], [eye_y]


def move_wing(i):
    wing_x, wing_y = get_wing_position(i)
    wing.set_data(wing_x, wing_y)

    
    eye_x, eye_y = get_eye_position()
    eye.set_data(eye_x, eye_y)

    return wing, eye,  


fig, ax = plt.subplots(figsize=(10, 6))


t_values = linspace(0, 2 * pi, 1000)
x, y = swan(t_values, parameters)
body, = ax.plot(x, y, 'b-', linewidth=2)  

wing, = ax.plot([], [], 'b-', linewidth=2)  

eye, = ax.plot([], [], 'ko', markersize=8)  

plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.axis('off') 
plt.title('Philipp\'s Flying Swan with Eye', fontsize=14)

ani = animation.FuncAnimation(
    fig=fig,
    func=move_wing,     
    frames=100,         
    init_func=init_plot, 
    interval=50,         
    blit=False,          
    repeat=True          
)

HTML(ani.to_jshtml())

plt.tight_layout()
plt.show()

snake
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML


snake_length = 1000  
amplitude = 0.5     
tail_length = 200    
tail_amplitude = 1.5 


def create_closed_snake():
   
    t = np.linspace(0, 2 * np.pi, snake_length)
   
    x = 5 * np.cos(t)
    y = 3 * np.sin(t)
    
   
    x += 0.5 * np.cos(3 * t)
    y += 0.5 * np.sin(4 * t)
    

    head_x = x[0] + 0.1 * np.cos(t[0])
    head_y = y[0] + 0.1 * np.sin(t[0])
    
    return x, y, head_x, head_y


def init_snake():
 
    x, y, head_x, head_y = create_closed_snake()
    
   
    snake_body.set_data(x, y)
    snake_tail.set_data(x[:tail_length], y[:tail_length])
    snake_head.set_data([head_x], [head_y])
    
    return snake_body, snake_tail, snake_head


def update_snake(frame):
   
    x, y, head_x, head_y = create_closed_snake()
    

    tail_start = snake_length - tail_length
    for i in range(tail_start, snake_length):
   
        phase_shift = 0.1 * frame + 0.05 * (i - tail_start)
        

        x[i] += tail_amplitude * 0.3 * np.sin(phase_shift) * (i - tail_start) / tail_length
        y[i] += tail_amplitude * np.cos(phase_shift) * (i - tail_start) / tail_length
    
 
    head_x = x[0] + 0.1 * np.cos(0.1 * frame)
    head_y = y[0] + 0.1 * np.sin(0.1 * frame)
    

    snake_body.set_data(x, y)
    snake_tail.set_data(x[tail_start:], y[tail_start:])
    snake_head.set_data([head_x], [head_y])
    
    return snake_body, snake_tail, snake_head


fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('#f0f8ff') 
fig.patch.set_facecolor('#f0f8ff')
plt.grid(False)
plt.axis('equal')
plt.xlim(-6, 6)
plt.ylim(-5, 5)
plt.title('抖尾巴的蛇（封闭图形）', fontsize=16, fontweight='bold', color='darkblue')


snake_body, = ax.plot([], [], 'b-', linewidth=3, alpha=0.9)
snake_tail, = ax.plot([], [], 'b-', linewidth=3, alpha=0.7)
snake_head, = ax.plot([], [], 'bo', markersize=12, alpha=0.9)


plt.text(-5.8, 4.2, "封闭环形蛇身", fontsize=12, color='darkblue')
plt.text(-5.8, 3.7, "尾巴部分有自然抖动效果", fontsize=12, color='darkblue')


ani = animation.FuncAnimation(
    fig=fig,
    func=update_snake,
    frames=200,
    init_func=init_snake,
    interval=40,
    blit=True
)

plt.tight_layout()
plt.show()


# HTML(ani.to_jshtml())
