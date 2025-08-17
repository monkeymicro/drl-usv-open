
import pygame
import numpy as np
from dynamics_simulation import motionSim
import random
from typing import Dict, List, Tuple
from algo_struct.algo import rad_limit
import os
import pickle

class DrawPolygon():
    def __init__(self):
        # initial position
        self.points = [np.array([-10, -20]), np.array([10, -20]), np.array([10, 20]), np.array([0, 30]), np.array([-10, 20])]
        self.polygon = []
        # trasition
        self.rotation = 0
        self.x = 0
        self.y = 0
        self.scale=1.0

    def reset(self):
        self.points = [np.array([-10, -20]), np.array([10, -20]), np.array([10, 20]), np.array([0, 30]), np.array([-10, 20])]
        self.polygon = []
        # trasition
        self.rotation = 0
        self.x = 0
        self.y = 0
        self.scale=1.0

    def transition(self, move_x, move_y, rot_angle, scale=1.0):
        self.x += move_x
        self.y += move_y

        if self.x > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH
        elif self.x < 0:
            self.x = 0
        if self.y > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT
        elif self.y < 0:
            self.y = 0
        # print(f'self x, y is : {self.x}, {self.y}')
        self.rotation += rot_angle
        self.scale = scale
        self.polygon.clear()
        for point in self.points:
            # rotation
            _x = np.cos(self.rotation) * point[0] - np.sin(self.rotation) * point[1]
            _y = np.sin(self.rotation) * point[0] + np.cos(self.rotation) * point[1]
            # translation
            _x += self.x
            _y += self.y
            _x = self.scale * _x
            _y = self.scale * _y
            self.polygon.append([_x, _y])

    def detect_collision(self, collision: list):
        """
        Check for collision with convex polygon
        :param collision: list of (x, y) - collision object convex list
        :return: True, False
        """
        for point in self.polygon:  # check convex points
            if is_point_inside_polygon(point, collision):
                return True

        return False
    
    def detect_overlap(self, target: list):
        for point in self.polygon: 
            if is_point_inside_polygon(point, target) == False:
                return False
        return True
    
def is_point_inside_polygon(point, polygon):
    x, y = point
    count = 0
    n = len(polygon)
    
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        
        if (y1 <= y < y2 or y2 <= y < y1) and x <= max(x1, x2):
            xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-10) + x1
            if abs(x - xinters) < 1e-10:
                return True
            if x < xinters:
                count += 1
    
    return count % 2 == 1

def get_reward(observation: dict):
    score = -( observation['dpsi'] ** 2
            + 100 * observation['dspeed']**2)

    return score

def draw_text(surface, text, pos, font_size=24, color=(0, 0, 0)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def draw_info_panel(surface, observation, rspl, rspr, exp_speed, exp_angle, reward):
    panel_x = 10
    panel_y = 10
    line_height = 25
    
    panel_width = 300
    panel_height = 300
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(200)
    panel_surface.fill((255, 255, 255))
    surface.blit(panel_surface, (panel_x, panel_y))
    
    info_texts = [
        f"Boat State:",
        f"Position X: {observation.get('x', 0):.2f} m",
        f"Position Y: {observation.get('y', 0):.2f} m", 
        f"Heading angle: {observation.get('psi', 0):.1f}°",
        f"Heading angluar: {observation.get('dpsi', 0):.2f}°/s",
        f"Forward Speed: {observation.get('u', 0):.2f} m/s",
        f"Cross Speed: {observation.get('v', 0):.2f} m/s",
        f"Speed error: {observation.get('dspeed', 0):.2f} m/s",
        f"Longtitude Acc: {observation.get('acc_x', 0):.2f} m/s²",
        f"Yaw acc: {observation.get('acc_n', 0):.2f} m/s²",
        f"Thruster rspl: {rspl:.1f} rpm",
        f"Thruster rspr: {rspr:.0f} rpm",
        f"Expect speed: {exp_speed:.1f} m/s",
        f"Expect heading angle: {exp_angle:.0f}°",
        f"Reward: {reward:.2f}"
    ]
    
    for i, text in enumerate(info_texts):
        color = (0, 0, 0) if i > 0 else (0, 0, 255)  # blue
        font_size = 20 if i > 0 else 24
        draw_text(surface, text, (panel_x + 5, panel_y + 5 + i * line_height), font_size, color)

def draw_controls_help(surface):
    help_x = SCREEN_WIDTH - 250
    help_y = 10
    
    # 背景面板
    panel_surface = pygame.Surface((240, 120))
    panel_surface.set_alpha(200)
    panel_surface.fill((200, 255, 200))
    surface.blit(panel_surface, (help_x, help_y))
    
    help_texts = [
        "Description:",
        "W/S: Add/Decrease RSPL",
        "E/D: Add/Decrease RSPR",
        "SPACE: PAUSE/CONTINUE",
        "R: RESET"
    ]
    
    for i, text in enumerate(help_texts):
        color = (0, 100, 0) if i > 0 else (0, 0, 100)
        font_size = 18 if i > 0 else 20
        draw_text(surface, text, (help_x + 5, help_y + 5 + i * 20), font_size, color)

# pygame setup
pygame.init()
pygame.font.init()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("USV Simulation")
clock = pygame.time.Clock()
running = True
dt = 0

# Predefined some colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

RECT_WIDTH = 50
RECT_HEIGHT = 100

enemy_pos = DrawPolygon()
enemy_pos.transition(SCREEN_WIDTH * 2/5, SCREEN_HEIGHT * 2/5, np.pi/2)

player_pos = DrawPolygon()
player_pos.transition(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, np.pi)

goal_pos = DrawPolygon()
goal_pos.transition(SCREEN_WIDTH * 2/5, SCREEN_HEIGHT * 3/5, np.pi/2, scale=1.5)

delta_h = 0.1

my_mmg = motionSim.BoatMotionSim(delta_h=delta_h, name='usv')
x = np.array([0.0, 0.0, 0.0])
u = np.array([0.0, 0.0, 0.0])
nu = np.array([0.0, 0.0, 0.0])
acc = np.array([0.0, 0.0, 0.0])

rspl = 0
rspr = 0

observation=dict()
memory_demo = list()
count =0
exp_speed = random.choice([-2, -1, 1, 2, 3])
exp_angle = random.randint(-90, 90)
epoch = 1

paused = False
show_trail = True
trail_points = []

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                # reset simulation
                x = np.array([0.0, 0.0, 0.0])
                u = np.array([0.0, 0.0, 0.0])
                nu = np.array([0.0, 0.0, 0.0])
                acc = np.array([0.0, 0.0, 0.0])
                delta = 0
                rsp = 0
                player_pos.reset()
                player_pos.transition(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, np.pi)
                trail_points.clear()
                exp_speed = random.choice([-2, -1, 1, 2, 3])
                exp_angle = random.randint(-90, 90)
            elif event.key == pygame.K_t:
                show_trail = not show_trail
                
    if not paused:
        # Key inputs for manual control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            rspl += 100
        if keys[pygame.K_s]:
            rspl -= 100
        if keys[pygame.K_e]:
            rspr += 100
        if keys[pygame.K_d]:
            rspr -= 100

        # Clip the RPM values to a reasonable range
        rspl = np.clip(rspl, -5000, 5000)
        rspr = np.clip(rspr, -5000, 5000)

        def rsp_to_force(rsp):
            a = 2.30549953e-04
            b = 3.88991253e-01
            c = -2.87920929e+02
            return a*rsp*abs(rsp) + b*rsp + c

        # rsp to force
        f_left = rsp_to_force(rspl)
        f_right = rsp_to_force(rspr)
        x, u, nu, acc = my_mmg.run_MMG_double_thrust(x=x, u=u, f_left=f_left, f_right=f_right)
        
        x[2] = rad_limit(x[2])
        
        observation = {
            'x': x[0],
            'y': x[1],
            'psi': x[2] * 180.0/np.pi, # rad to deg
            'u': u[0],
            'v': u[1],
            'dpsi': u[2] * 180.0/np.pi - exp_angle, # rad/s to deg/s
            'dspeed': exp_speed - u[0],
            'acc_x': acc[0],
            'acc_y': acc[1],
            'acc_n': acc[2] * 180.0/np.pi,
            'rspl': rspl,
            'rspr': rspr,
        }
        
        reward = get_reward(observation)
        
        player_pos.transition(nu[1]*dt, -nu[0]* dt, nu[2]*dt)
        if len(trail_points) == 0 or np.linalg.norm([player_pos.x - trail_points[-1][0], player_pos.y - trail_points[-1][1]]) > 5:
            trail_points.append((player_pos.x, player_pos.y))
            if len(trail_points) > 500:
                trail_points.pop(0)

    screen.fill(WHITE)
    
    if show_trail and len(trail_points) > 1:
        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)  
            color = (int(255 * (1-alpha)), int(255 * alpha), 0) 
            pygame.draw.line(screen, color, trail_points[i-1], trail_points[i], 2)
    
    pygame.draw.polygon(screen, RED, player_pos.polygon)
    
    pygame.draw.circle(screen, BLUE, (int(player_pos.x), int(player_pos.y)), 3)
    
    heading_length = 40
    heading_end_x = player_pos.x + heading_length * np.cos(player_pos.rotation)
    heading_end_y = player_pos.y + heading_length * np.sin(player_pos.rotation)
    pygame.draw.line(screen, BLUE, (player_pos.x, player_pos.y), (heading_end_x, heading_end_y), 3)
    
    if 'u' in observation and 'v' in observation:
        speed_scale = 20 
        vel_end_x = player_pos.x + observation['u'] * speed_scale
        vel_end_y = player_pos.y - observation['v'] * speed_scale 
        pygame.draw.line(screen, GREEN, (player_pos.x, player_pos.y), (vel_end_x, vel_end_y), 2)
        if abs(observation['u']) > 0.1 or abs(observation['v']) > 0.1:
            pygame.draw.circle(screen, GREEN, (int(vel_end_x), int(vel_end_y)), 4)

    exp_heading_length = 60
    exp_heading_rad = np.radians(exp_angle)
    exp_end_x = player_pos.x + exp_heading_length * np.cos(exp_heading_rad)
    exp_end_y = player_pos.y + exp_heading_length * np.sin(exp_heading_rad)
    pygame.draw.line(screen, YELLOW, (player_pos.x, player_pos.y), (exp_end_x, exp_end_y), 2)
    
    if 'x' in observation:
        draw_info_panel(screen, observation, rspl, rspr, exp_speed, exp_angle, reward)
    
    draw_controls_help(screen)
    
    if paused:
        pause_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        pause_surface.set_alpha(100)
        pause_surface.fill((0, 0, 0))
        screen.blit(pause_surface, (0, 0))
        draw_text(screen, "PAUSED - Press SPACE to continue", 
                 (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2), 32, (255, 255, 255))
    
    status_y = SCREEN_HEIGHT - 30
    status_texts = [
        f"Time: {count*dt:.1f}s",
        f"Epoch: {epoch}",
        f"Trail: {'ON' if show_trail else 'OFF'} (T to toggle)"
    ]
    
    for i, text in enumerate(status_texts):
        draw_text(screen, text, (10 + i * 150, status_y), 18, (50, 50, 50))
    
    collision_detected = False
    
    goal_reached = False
    
    if collision_detected:
        draw_text(screen, "COLLISION!", (SCREEN_WIDTH//2 - 50, 50), 32, (255, 0, 0))
    
    if goal_reached:
        draw_text(screen, "GOAL REACHED!", (SCREEN_WIDTH//2 - 70, 80), 32, (0, 255, 0))
    
    pygame.display.flip()

    dt = 0.1
    clock.tick(10)  # 60 FPS
    count += 1
    
    if count % 1000 == 0:
        exp_speed = random.choice([-2, -1, 1, 2, 3])
        exp_angle = random.randint(-180, 180)
        epoch += 1
    
pygame.quit()
