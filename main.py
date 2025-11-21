import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -----------------------------
# Paramètres de l'environnement
# -----------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
FPS = 60
STATE_SIZE = 4   # CartPole a 4 états
ACTION_SIZE = 2  # Gauche ou droite
EPISODES = 500
MAX_STEPS = 200

# -----------------------------
# Hyperparamètres DQN
# -----------------------------
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# -----------------------------
# Définition du DQN
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Mémoire de Replay
# -----------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -----------------------------
# Simple environnement CartPole
# -----------------------------
class CartPoleEnv:
    def __init__(self):
        self.gravity = 0.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # moitié de la longueur du poteau
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.reset()

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (9.8 * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0 if not done else 0.0
        return self.state, reward, done

# -----------------------------
# Initialisation DQN
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

# -----------------------------
# Pygame Interface
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("DQN CartPole")
font = pygame.font.SysFont("Arial", 20)

def draw_cartpole(state):
    screen.fill((255, 255, 255))
    x = SCREEN_WIDTH//2 + int(state[0]*100)
    y = SCREEN_HEIGHT//2
    pole_len = 100
    theta = state[2]
    x_pole = x + int(pole_len*np.sin(theta))
    y_pole = y - int(pole_len*np.cos(theta))
    pygame.draw.rect(screen, (0,0,255), (x-20, y-10, 40, 20))  # cart
    pygame.draw.line(screen, (255,0,0), (x,y), (x_pole,y_pole), 5)  # pole
    pygame.display.flip()

# -----------------------------
# Sélection d'action
# -----------------------------
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state_t).argmax().item()

# -----------------------------
# Entraînement DQN
# -----------------------------
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
    state_batch = torch.FloatTensor(batch_state).to(device)
    action_batch = torch.LongTensor(batch_action).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(batch_next_state).to(device)
    done_batch = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
    expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------
# Boucle principale
# -----------------------------
env = CartPoleEnv()
epsilon = EPSILON_START
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for t in range(MAX_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        draw_cartpole(state)
        optimize_model()
        if done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Affichage récompense
    screen.fill((255,255,255))
    text = font.render(f"Episode: {episode}  Reward: {total_reward}  Epsilon: {epsilon:.2f}", True, (0,0,0))
    screen.blit(text, (20, 20))
    pygame.display.flip()

pygame.quit()
