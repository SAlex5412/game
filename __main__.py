import math
import os
from random import *

import pygame_gui
from PIL import Image


class dMap:
    def __init__(self):
        self.roomList = []
        self.cList = []

    def makeMap(self, xsize, ysize, fail, b1, mrooms, mrsize):
        self.size_x = xsize
        self.size_y = ysize
        self.mapArr = []
        self.mrsize = mrsize
        for y in range(ysize):
            tmp = []
            for x in range(xsize):
                tmp.append(1)
            self.mapArr.append(tmp)

        w, l, t = self.makeRoom()
        while len(self.roomList) == 0:
            y = randrange(ysize - 1 - l) + 1
            x = randrange(xsize - 1 - w) + 1
            p = self.placeRoom(l, w, x, y, xsize, ysize, 6, 0)
        failed = 0
        while failed < fail:  # The lower the value that failed< , the smaller the dungeon
            chooseRoom = randrange(len(self.roomList))
            ex, ey, ex2, ey2, et = self.makeExit(chooseRoom)
            feature = randrange(100)
            if feature < b1:  # Begin feature choosing (more features to be added here)
                w, l, t = self.makeCorridor()
            else:
                w, l, t = self.makeRoom()
            roomDone = self.placeRoom(l, w, ex2, ey2, xsize, ysize, t, et)
            if roomDone == 0:  # If placement failed increase possibility map is full
                failed += 1
            elif roomDone == 2:  # Possiblilty of linking rooms
                if self.mapArr[ey2][ex2] == 0:
                    if randrange(100) < 7:
                        self.makePortal(ex, ey)
                    failed += 1
            else:  # Otherwise, link up the 2 rooms
                self.makePortal(ex, ey)
                failed = 0
                if t < 5:
                    tc = [len(self.roomList) - 1, ex2, ey2, t]
                    self.cList.append(tc)
                    self.joinCorridor(len(self.roomList) - 1, ex2, ey2, t, 50)
            if len(self.roomList) == mrooms:
                failed = fail
        self.finalJoins()

    def makeRoom(self):
        rtype = 5
        rwide = randrange(8) + self.mrsize
        rlong = randrange(8) + self.mrsize
        return rwide, rlong, rtype

    def makeCorridor(self):
        clength = randrange(18) + 3
        heading = randrange(4)
        if heading == 0:
            wd = 1
            lg = -clength
        elif heading == 1:
            wd = clength
            lg = 1
        elif heading == 2:
            wd = 1
            lg = clength
        elif heading == 3:
            wd = -clength
            lg = 1
        return wd, lg, heading

    def placeRoom(self, ll, ww, xposs, yposs, xsize, ysize, rty, ext):
        xpos = xposs
        ypos = yposs
        if ll < 0:
            ypos += ll + 1
            ll = abs(ll)
        if ww < 0:
            xpos += ww + 1
            ww = abs(ww)

        if rty == 5:
            if ext == 0 or ext == 2:
                offset = randrange(ww)
                xpos -= offset
            else:
                offset = randrange(ll)
                ypos -= offset

        canPlace = 1
        if ww + xpos + 1 > xsize - 1 or ll + ypos + 1 > ysize:
            canPlace = 0
            return canPlace
        elif xpos < 1 or ypos < 1:
            canPlace = 0
            return canPlace
        else:
            for j in range(ll):
                for k in range(ww):
                    if self.mapArr[(ypos) + j][(xpos) + k] != 1:
                        canPlace = 2

        if canPlace == 1:
            temp = [ll, ww, xpos, ypos]
            self.roomList.append(temp)
            for j in range(ll + 2):
                for k in range(ww + 2):
                    self.mapArr[(ypos - 1) + j][(xpos - 1) + k] = 2
            for j in range(ll):
                for k in range(ww):
                    self.mapArr[ypos + j][xpos + k] = 0
        return canPlace

    def makeExit(self, rn):
        room = self.roomList[rn]
        while True:
            rw = randrange(4)
            if rw == 0:
                rx = randrange(room[1]) + room[2]
                ry = room[3] - 1
                rx2 = rx
                ry2 = ry - 1
            elif rw == 1:
                ry = randrange(room[0]) + room[3]
                rx = room[2] + room[1]
                rx2 = rx + 1
                ry2 = ry
            elif rw == 2:
                rx = randrange(room[1]) + room[2]
                ry = room[3] + room[0]
                rx2 = rx
                ry2 = ry + 1
            elif rw == 3:
                ry = randrange(room[0]) + room[3]
                rx = room[2] - 1
                rx2 = rx - 1
                ry2 = ry
            if self.mapArr[ry][rx] == 2:
                break
        return rx, ry, rx2, ry2, rw

    def makePortal(self, px, py):
        ptype = randrange(100)
        if ptype > 90:
            self.mapArr[py][px] = 0
            return
        elif ptype > 75:
            self.mapArr[py][px] = 0
            return
        elif ptype > 40:
            self.mapArr[py][px] = 0
            return
        else:
            self.mapArr[py][px] = 0

    def joinCorridor(self, cno, xp, yp, ed, psb):
        cArea = self.roomList[cno]
        if xp != cArea[2] or yp != cArea[3]:
            endx = xp - (cArea[1] - 1)
            endy = yp - (cArea[0] - 1)
        else:
            endx = xp + (cArea[1] - 1)
            endy = yp + (cArea[0] - 1)
        checkExit = []
        if ed == 0:
            if endx > 1:
                coords = [endx - 2, endy, endx - 1, endy]
                checkExit.append(coords)
            if endy > 1:
                coords = [endx, endy - 2, endx, endy - 1]
                checkExit.append(coords)
            if endx < self.size_x - 2:
                coords = [endx + 2, endy, endx + 1, endy]
                checkExit.append(coords)
        elif ed == 1:
            if endy > 1:
                coords = [endx, endy - 2, endx, endy - 1]
                checkExit.append(coords)
            if endx < self.size_x - 2:
                coords = [endx + 2, endy, endx + 1, endy]
                checkExit.append(coords)
            if endy < self.size_y - 2:
                coords = [endx, endy + 2, endx, endy + 1]
                checkExit.append(coords)
        elif ed == 2:
            if endx < self.size_x - 2:
                coords = [endx + 2, endy, endx + 1, endy]
                checkExit.append(coords)
            if endy < self.size_y - 2:
                coords = [endx, endy + 2, endx, endy + 1]
                checkExit.append(coords)
            if endx > 1:
                coords = [endx - 2, endy, endx - 1, endy]
                checkExit.append(coords)
        elif ed == 3:
            if endx > 1:
                coords = [endx - 2, endy, endx - 1, endy]
                checkExit.append(coords)
            if endy > 1:
                coords = [endx, endy - 2, endx, endy - 1]
                checkExit.append(coords)
            if endy < self.size_y - 2:
                coords = [endx, endy + 2, endx, endy + 1]
                checkExit.append(coords)
        for xxx, yyy, xxx1, yyy1 in checkExit:
            if self.mapArr[yyy][xxx] == 0:
                if randrange(100) < psb:
                    self.makePortal(xxx1, yyy1)

    def finalJoins(self):
        for x in self.cList:
            self.joinCorridor(x[0], x[1], x[2], x[3], 10)


startx = 150 // 3
starty = 150 // 3
themap = dMap()
themap.makeMap(startx, starty, 200, 20, 20, mrsize=7)
themap = dMap()
themap.makeMap(startx, starty, 200, 20, 20, mrsize=7)
for y in range(starty):
    line = ""
    for x in range(startx):
        if themap.mapArr[y][x] == 0:
            line += "."
        if themap.mapArr[y][x] == 1:
            line += " "
        if themap.mapArr[y][x] == 2:
            line += "#"
        if themap.mapArr[y][x] == 3 or themap.mapArr[y][x] == 4 or themap.mapArr[y][x] == 5:
            line += "="

import pygame

# Initialize Pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
BLUE = (0, 0, 255)  # Define BLUE color for the player

# Tile size and map dimensions
tile_size = 10
map_width = themap.size_x * tile_size
map_height = themap.size_y * tile_size

# Width of the UI area
ui_width = 100

# Create the Pygame window
screen_width = map_width + ui_width
screen = pygame.display.set_mode((screen_width, map_height), pygame.SRCALPHA)
pygame.display.set_caption('Randomly Generated Dungeon')

# Initialize the Pygame GUI manager
gui_manager = pygame_gui.UIManager((screen_width, map_height))

# Add a scrollable list
list_rect = pygame.Rect(map_width - 30, 0, ui_width+ 30, map_height)
list_container = pygame_gui.elements.UIPanel(
    relative_rect=list_rect,
    manager=gui_manager,
)

list_height = 20  # Height of each list item
scrollable_list = pygame_gui.elements.UIScrollingContainer(
    relative_rect=pygame.Rect((0, 0), (ui_width+ 30, map_height)),
    manager=gui_manager,
    container=list_container,
)
# Player settings
player_color = BLUE
player_size = tile_size
player_x, player_y = 0, 0
player_x += ui_width // tile_size

# Find a random starting position for the player on a floor tile
while True:
    player_x = randint(0, themap.size_x - 1)
    player_y = randint(0, themap.size_y - 1)
    if themap.mapArr[player_y][player_x] == 0:  # 0 is a floor tile
        break


# Function to draw the map
def draw_map():
    for y in range(themap.size_y):
        for x in range(themap.size_x):
            rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
            if themap.mapArr[y][x] == 0:  # Floor
                pygame.draw.rect(screen, WHITE, rect)
            elif themap.mapArr[y][x] == 1:  # Wall
                pygame.draw.rect(screen, GRAY, rect)
            elif themap.mapArr[y][x] in [3, 4, 5]:  # Door
                pygame.draw.rect(screen, RED, rect)


class Player:
    def __init__(self, color, size, mapArr):
        self.color = color
        self.size = size
        self.mapArr = mapArr
        self.x, self.y = self.find_start_position()
        self.collision_circles = []  # List to store collision circles (position, transparency)
        self.health = 100  # Set initial health value
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            print("Game Over")  # You can replace this with your game over logic

    def find_start_position(self):
        while True:
            x = randint(0, len(self.mapArr[0]) - 1)
            y = randint(0, len(self.mapArr) - 1)
            if self.mapArr[y][x] == 0:
                return x, y

    def draw(self, screen):
        rect = pygame.Rect(self.x * self.size, self.y * self.size, self.size, self.size)
        pygame.draw.rect(screen, self.color, rect)

    def move(self, dx, dy, enemies):
        new_x = self.x + dx
        new_y = self.y + dy

        # Check for collisions with enemies
        for enemy in enemies:
            if new_x == enemy.x and new_y == enemy.y:
                enemy.take_damage(randint(40, 100))
                if enemy.killed == True:
                    continue
                return  # Do not move if collision with an enemy

        # Check if the new position is a valid floor tile
        if 0 <= new_x < len(self.mapArr[0]) and 0 <= new_y < len(self.mapArr) and self.mapArr[new_y][new_x] == 0:
            self.x = new_x
            self.y = new_y
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def draw_rays(self, screen, tile_size, mapArr):
        mapArr = self.mapArr
        ray_length = 3 * tile_size
        ray_width = 2
        ray_collision_radius = 5  # Adjust the radius of the collision circle
        rays = 16

        for angle in range(0, 360, int(360 / rays)):
            radians = math.radians(angle)
            dx = int(math.cos(radians) * ray_length)
            dy = int(math.sin(radians) * ray_length)

            start_pos = (self.x * tile_size + tile_size // 2, self.y * tile_size + tile_size // 2)
            end_pos = (start_pos[0] + dx, start_pos[1] + dy)

            # Perform ray-casting to detect collisions
            collision_pos = self.cast_ray(start_pos, end_pos, mapArr, tile_size)

            # Draw the ray up to the collision point
            # pygame.draw.line(screen, BLUE, start_pos, collision_pos, ray_width)

            # Store the collision circle if it collided with a wall
            if collision_pos != end_pos:
                self.collision_circles.append((collision_pos, 255))  # Initial transparency is 255 (opaque)

    def decrease_transparency(self, decrement=5):
        new_circles = []

        for circle_pos, transparency in self.collision_circles:
            # Decrease transparency (adjust the rate as needed)
            new_transparency = max(0, transparency - decrement)  # Adjust the decrement value as needed
            new_circles.append((circle_pos, new_transparency))

        # Update the list with new transparency values
        self.collision_circles = new_circles

    def draw_collision_circles(self, screen, radius):
        for circle_pos, transparency in self.collision_circles:
            # Create a surface with per-pixel alpha
            circle_surface = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)

            # Draw the circle on the surface
            pygame.draw.circle(circle_surface, (255, 255, 0, transparency), (radius, radius), radius)

            # Blit the circle surface onto the main surface
            screen.blit(circle_surface, (int(circle_pos[0]) - radius, int(circle_pos[1]) - radius))

    def cast_ray(self, start, end, mapArr, tile_size):
        step_size = 5  # Adjust step size based on your preference for accuracy

        # Calculate the direction vector
        direction = (end[0] - start[0], end[1] - start[1])
        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        direction = (direction[0] / length, direction[1] / length)

        # Initialize variables for ray-casting
        current_pos = start
        steps = 0

        while steps < length:
            steps += step_size
            current_pos = (start[0] + direction[0] * steps, start[1] + direction[1] * steps)

            # Check for collisions with walls (type 2)
            tile_x = int(current_pos[0] // tile_size)
            tile_y = int(current_pos[1] // tile_size)

            if 0 <= tile_x < len(mapArr[0]) and 0 <= tile_y < len(mapArr):
                if mapArr[tile_y][tile_x] == 2:
                    return current_pos

        # If no collision, return the endpoint of the ray
        return end


player = Player(BLUE, tile_size, themap.mapArr)


def check_player_enemy_collisions(player, enemies):
    for enemy in enemies:
        if (
                abs(player.x - enemy.x) <= 1
                and abs(player.y - enemy.y) <= 1
                and player.health > 0
        ):
            player.take_damage(enemy.damage)


class Enemy:
    def __init__(self, color, size, mapArr):
        self.color = color
        self.size = size
        self.mapArr = mapArr
        self.x, self.y = self.find_start_position()
        self.last_player_position = None
        self.target_position = None
        self.preferred_direction = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.steps_in_preferred_direction = 0
        self.max_steps_in_preferred_direction = randint(5, 15)
        self.damage = randint(1, 100)
        self.killed = False
        self.hp = randint(75, 100)

    def take_damage(self, damage):
        self.hp -= damage
        if self.hp <= 0:
            self.killed = True

    def find_start_position(self):
        while True:
            x = randint(0, len(self.mapArr[0]) - 1)
            y = randint(0, len(self.mapArr) - 1)
            if self.mapArr[y][x] == 0:
                return x, y

    def distance_to_player(self, player_x, player_y):
        return math.sqrt((self.x - player_x) ** 2 + (self.y - player_y) ** 2)

    def draw(self, screen, player_x, player_y):
        if self.killed is False:
            distance = self.distance_to_player(player_x, player_y)
            transparency = max(0, min(255, int((5 - distance) * 5.1 * 50 / 5)))  # Adjust 50 as needed

            # Create a surface with per-pixel alpha
            enemy_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)

            # Draw the enemy with transparency on the surface
            pygame.draw.rect(enemy_surface, (255, 0, 0, transparency), (0, 0, self.size, self.size))

            # Blit the enemy surface onto the main surface
            screen.blit(enemy_surface, (self.x * self.size, self.y * self.size))

    def move_randomly(self):
        if self.steps_in_preferred_direction < self.max_steps_in_preferred_direction:
            self.steps_in_preferred_direction += 1
        else:
            # Change preferred direction and reset step count
            self.preferred_direction = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            self.steps_in_preferred_direction = 0

        possible_moves = self.get_neighbors((self.x, self.y))
        if possible_moves:
            # With a certain probability, deviate from the preferred direction
            if random() < 0.2:
                self.preferred_direction = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

            dx, dy = self.preferred_direction
            new_x = self.x + dx
            new_y = self.y + dy

            if self.mapArr[new_y][new_x] == 0:
                self.x = new_x
                self.y = new_y
            else:
                # If the preferred direction is blocked, choose another random move
                self.steps_in_preferred_direction = self.max_steps_in_preferred_direction
                self.move_randomly()

    def move_towards_player(self, player_x, player_y):
        path = self.a_star_pathfinding((self.x, self.y), (player_x, player_y))

        if path:
            next_x, next_y = path[0]
            if (next_x, next_y) == (player_x, player_y):
                return

            if self.mapArr[next_y][next_x] == 0:
                self.x = next_x
                self.y = next_y

    def update(self, player_x, player_y):
        if self.can_see_player(player_x, player_y):
            self.target_position = (player_x, player_y)
        elif self.last_player_position is not None:
            self.target_position = self.last_player_position

        if self.target_position is not None:
            self.move_towards_player(*self.target_position)
            if (self.x, self.y) == self.target_position:
                # Player reached, reset target position
                self.target_position = None
        else:
            # Move randomly if no target position
            self.move_randomly()

    def can_see_player(self, player_x, player_y):
        # Implement simple line-of-sight check
        return abs(player_x - self.x) <= 5 and abs(player_y - self.y) <= 5

    def a_star_pathfinding(self, start, goal):
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda pos: f_score[pos])

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            open_set.remove(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

    def heuristic(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [neighbor for neighbor in neighbors if
                0 <= neighbor[0] < len(self.mapArr[0]) and 0 <= neighbor[1] < len(self.mapArr) and
                self.mapArr[neighbor[1]][neighbor[0]] == 0]


class Effect:
    def __init__(self, spritesheet_path, columns, scale=1, speed=0.1):
        self.spritesheet = pygame.image.load(spritesheet_path).convert_alpha()
        self.columns = columns
        self.scale = scale
        self.speed = speed
        self.frames = self.load_frames()
        self.current_frame = 0
        self.position = (0, 0)
        self.active = False

    def load_frames(self):
        frame_width = self.spritesheet.get_width() // self.columns
        frame_height = self.spritesheet.get_height()
        frames = []
        for i in range(self.columns):
            frame = pygame.Surface((frame_width, frame_height), pygame.SRCALPHA)
            frame.blit(self.spritesheet, (0, 0), (i * frame_width, 0, frame_width, frame_height))
            frame = pygame.transform.scale(frame, (int(frame_width * self.scale), int(frame_height * self.scale)))
            frames.append(frame)
        return frames

    def activate(self, position):
        self.position = position
        self.position[0] += tile_size // 2
        self.position[1] += tile_size // 2
        self.current_frame = 0
        self.active = True

    def deactivate(self):
        self.active = False

    def update(self):
        if self.active:
            self.current_frame += self.speed
            if int(self.current_frame) >= len(self.frames):
                self.deactivate()

    def draw(self, screen, color=(245, 222, 179)):
        if self.active:
            frame_index = int(self.current_frame) % len(self.frames)
            frame = self.frames[frame_index]

            # Create a copy of the frame with the desired color
            colored_frame = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
            colored_frame.fill((*color, 0))
            frame.blit(colored_frame, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

            rect = frame.get_rect(center=self.position)
            screen.blit(frame, rect.topleft)


num_enemies = 5
enemies = [Enemy(RED, tile_size, themap.mapArr) for _ in range(num_enemies)]
im = Image.open("eea.png")
effects = [Effect("eea.png", columns=9, scale=1 / im.size[1] * tile_size, speed=2) for i in range(num_enemies)]


# Particle class for the particles generated by the player's square
class Particle:
    def __init__(self, position, color, size):
        self.position = position
        self.color = color
        self.size = size
        self.transparency = 255  # Initial transparency is 255 (opaque)

    def decrease_transparency(self, decrement=5):
        self.transparency = max(0, self.transparency - decrement)

    def draw(self, screen):
        # Create a surface with per-pixel alpha
        particle_surface = pygame.Surface((2 * self.size, 2 * self.size), pygame.SRCALPHA)

        # Draw the particle on the surface
        pygame.draw.circle(particle_surface, (self.color[0], self.color[1], self.color[2], self.transparency),
                           (self.size, self.size), self.size)

        # Blit the particle surface onto the main surface
        screen.blit(particle_surface, (int(self.position[0]) - self.size, int(self.position[1]) - self.size))


class PlayerSquare:
    def __init__(self, color, size):
        self.color = color
        self.size = size
        self.position = (0, 0)
        self.active = False
        self.particles = []

    def activate(self, position):
        self.position = position
        self.active = True

    def deactivate(self, enemies):
        if self.active:
            # Emit particles for enemies underneath the square
            particle_color = (0, 255, 0)  # Particle color for square deactivation
            particle_size = 2
            particle_count = 10

            for enemy in enemies:
                if self.check_collision(enemy):
                    for _ in range(particle_count):
                        particle = Particle(
                            (enemy.x * tile_size + tile_size // 2, enemy.y * tile_size + tile_size // 2),
                            particle_color, particle_size)
                        self.particles.append(particle)

            # Deactivate the square after emitting particles
            self.active = False

    def update(self, enemies):
        if self.active:
            # Check for collisions with enemies while the square is active
            for enemy in enemies:
                if self.check_collision(enemy):
                    # Do something when the square collides with an enemy
                    pass

    def check_collision(self, enemy):
        # Check if the square collides with the enemy
        x, y = self.position
        enemy_x, enemy_y = enemy.x * tile_size, enemy.y * tile_size
        return abs(x - enemy_x) < self.size and abs(y - enemy_y) < self.size

    def draw(self, screen):
        if self.active:
            # Create a surface with per-pixel alpha
            square_surface = pygame.Surface((2 * self.size, 2 * self.size), pygame.SRCALPHA)

            # Draw the square on the surface
            pygame.draw.rect(square_surface, (self.color[0], self.color[1], self.color[2], 100),
                             (0, 0, 2 * self.size, 2 * self.size))

            # Blit the square surface onto the main surface
            screen.blit(square_surface, (int(self.position[0]) - self.size, int(self.position[1]) - self.size))

        # Draw particles
        for particle in self.particles:
            particle.decrease_transparency(5)  # Adjust decrement value as needed
            particle.draw(screen)

            # Remove particles with zero transparency
            if particle.transparency == 0:
                self.particles.remove(particle)


player_square = PlayerSquare((0, 0, 255), 200)

effect_spritesheet_path = os.path.join("eea.png")
effect = Effect(effect_spritesheet_path, columns=9, scale=10, speed=1)


class DungeonObject:
    def __init__(self, color, size):
        self.color = color
        self.size = size
        self.position = (0, 0)
        self.visible = False
        self.image = None  # Surface to store the object's image
        self.original_image = None  # Surface to store the original image without transparency
        self.transparency = 255  # 255 is fully opaque, 0 is fully transparent
        self.rect = pygame.Rect(self.position[0], self.position[1], self.size, self.size)

    def place_randomly(self, dungeon_map, tile_size):
        while True:
            x = randint(0, len(dungeon_map[0]) - 1)
            y = randint(0, len(dungeon_map) - 1)
            if dungeon_map[y][x] == 0:
                self.position = (x * tile_size, y * tile_size)
                self.rect = pygame.Rect(self.position[0], self.position[1], self.size, self.size)
                break

    def update_visibility(self, player_position, visibility_range):
        distance_to_player = math.hypot(player_position[0] - self.position[0], player_position[1] - self.position[1])
        self.visible = distance_to_player <= visibility_range

        # Adjust transparency based on distance
        max_transparency_distance = visibility_range  # Tweak this value based on your game's scale
        transparency_ratio = min(1, distance_to_player / max_transparency_distance)
        self.transparency = int((1 - transparency_ratio) * 255)

    def draw(self, screen):
        if self.visible:
            if self.image is None:
                # Create the object's image if it's not created yet
                self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
                self.original_image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
                pygame.draw.rect(self.original_image, self.color, (0, 0, self.size, self.size))
                self.image.blit(self.original_image, (0, 0))
            # Set transparency based on the calculated transparency value
            self.image.set_alpha(self.transparency)
            screen.blit(self.image, self.position)


dungeon_object = DungeonObject((255, 165, 0), tile_size)
dungeon_object.place_randomly(themap.mapArr, tile_size)

sound = pygame.mixer.Sound("servernaya-gudenie-ventilyatorov-37229.wav")
move_sound = pygame.mixer.Sound("move sound.mp3")
move_sound.set_volume(0.125)


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


sound.play(loops=-1)


def control_volume():
    distance = float("inf")
    for i in enemies:
        distance = min(distance, calculate_distance((player.x, player.y), (i.x, i.y)))

    # Define a maximum distance (adjust according to your needs)
    max_distance = 25

    # Exponential volume increase with adjustable exponent
    exponent = 15  # Adjust the exponent to control the rate of increase
    volume = max(0, min(1, math.exp(-exponent * distance / max_distance)))

    # Set the volume
    sound.set_volume(volume * 0.125)


class PowerUp(DungeonObject):
    def __init__(self, color, size):
        super().__init__(color, size)

    def activate(self):
        # Implement the power-up effect or ability here
        print("Power-up activated: Increased player health by 10")


power_up = PowerUp((255, 165, 165), tile_size)
power_up.place_randomly(themap.mapArr, tile_size)

# Set the maximum distance for transparency
visibility_range = 500
# Game loop
clock = pygame.time.Clock()
running = True
player_moved = False
sonara = False
reseted = False

score = 0
def reset_duungeon():
    themap = dMap()
    themap.makeMap(startx, starty, 200, 20, 20, mrsize=7)
    player.mapArr = themap.mapArr
    player.x, player.y = player.find_start_position()
    for el in enemies:
        el.mapArr = themap.mapArr
        el.x, el.y = el.find_start_position()
        el.killed = False

    dungeon_object.place_randomly(themap.mapArr, tile_size)
    power_up.place_randomly(themap.mapArr, tile_size)



background = pygame.image.load("load screen.png")
ss = True
while ss:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            ss = False
        if event.type == pygame.QUIT:
            running = False
            ss = False

    screen.fill((35, 35, 35))
    screen.blit(background, (0, 0))
    pygame.display.flip()

powerups_d = {
    "heal": 0,
    "explosion": 0,
    "radar": 0
}
powerups_da = {
    "heal": False,
    "explosion": False,
    "radar": False
}
explosion_square = PlayerSquare((0, 0, 255), 200)
while running:
    time_delta = clock.tick(60) / 1000.0  # Track the time passed

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                player.move(0, -1, enemies)  # Up
                player_moved = True
            elif event.key == pygame.K_s:
                player.move(0, 1, enemies)  # Down
                player_moved = True
            elif event.key == pygame.K_a:
                player.move(-1, 0, enemies)  # Left
                player_moved = True
            elif event.key == pygame.K_d:
                player.move(1, 0, enemies)  # Right
                player_moved = True
            elif event.key == pygame.K_SPACE:
                # Activate the player square at the player's position
                if powerups_da["radar"] == True:
                    player_square.activate(pygame.mouse.get_pos())
                    for i in range(len(effects)):
                        if player_square.check_collision(enemies[i]) and enemies[i].killed is False:
                             effects[i].activate([enemies[i].x*tile_size, enemies[i].y*tile_size])
                    powerups_da["radar"] = False
                elif powerups_d["radar"] > 0:
                    powerups_da["radar"] = True
            elif event.key == pygame.K_q:
                if powerups_da["explosion"] == True:
                    explosion_square.activate(pygame.mouse.get_pos())
                    for i in range(len(effects)):
                        if explosion_square.check_collision(enemies[i]) and enemies[i].killed is False:
                            enemies[i].take_damage(25)
                    print("'")
                    powerups_da["explosion"] = False

                elif powerups_d["explosion"] > 0:
                    powerups_da["explosion"] = True
            elif event.key == pygame.K_r:
                if powerups_d["heal"] > 0:
                    player.health += 20
                    if player.health > 100:
                        player.health = 100


        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                # Deactivate the player square when space key is released

                # player_square.deactivate(enemies)
                sonara = True

        # Process GUI events
        gui_manager.process_events(event)

    # Update the GUI manager
    # Update the GUI manager
    gui_manager.update(time_delta)

    player.decrease_transparency(1)  # Adjust decrement value as needed

    # Update player square position based on cursor position
    cursor_position = pygame.mouse.get_pos()
    player_square.activate(cursor_position)
    explosion_square.activate(cursor_position)

    player_square.update(enemies)  # Update the player square
    # Update enemies
    if player_moved:
        move_sound.play()
        print("!")
        for enemy in enemies:
            enemy.update(player.x, player.y)

    dungeon_object.update_visibility((player.x * tile_size, player.y * tile_size), visibility_range)

    power_up.update_visibility((player.x * tile_size, player.y * tile_size), visibility_range)

    # Draw the map, player, player square, enemies, and particles
    screen.fill(BLACK)  # Fill the background
    # draw_map()
    player.draw(screen)
    dungeon_object.draw(screen)
    power_up.draw(screen)
    if powerups_da["radar"] == True:
        player_square.draw(screen)
    if powerups_da["explosion"] == True:
        explosion_square.draw(screen)
    if player_moved:
        player.draw_rays(screen, tile_size, themap.mapArr)  # Draw rays with collisions
    player.draw_collision_circles(screen, 2)  # Draw collision circles
    for i in range(len(enemies)):
        enemies[i].draw(screen, player.x, player.y)  # Draw enemies

        effects[i].draw(screen)
        effects[i].update()
    sonara = False

    # Draw particles emitted by the player square
    for particle in player_square.particles:
        particle.decrease_transparency(5)  # Adjust decrement value as needed
        particle.draw(screen)

        # Remove particles with zero transparency
        if particle.transparency == 0:
            player_square.particles.remove(particle)
    # effect.update()
    # effect.draw(screen)
    # Draw the GUI
    # Populate the scrollable list with some items\
    c=0
    for i, j in powerups_d.items():
        list_item = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((0, c * list_height), (ui_width+30, list_height)),
            text=f'{i} {j}',
            manager=gui_manager,
            container=scrollable_list,
        )
        c+=1
    gui_manager.draw_ui(screen)
    dungeon_object.draw(screen)
    if player_moved:
        check_player_enemy_collisions(player, enemies)
    if player.health <= 0:
        reset_duungeon()
        score = 0
        player.health = 100
    if dungeon_object.position == (player.x * tile_size, player.y * tile_size) and reseted is False:
        reset_duungeon()
        reseted = True
        score +=1
    if power_up.position == (player.x * tile_size, player.y * tile_size):
        r = randint(1, 3)
        if r == 1:
            powerups_d["radar"] += 1
        elif r == 2:
            powerups_d["explosion"] += 1
        elif r == 3:
            powerups_d["heal"] += 1
        power_up.position = (-100, -100)

    # Display player's health
    health_font = pygame.font.Font(None, 36)
    health_text = health_font.render(f"Health: {player.health}", True, BLUE)
    screen.blit(health_text, (50, screen.get_height() - 50 / 2))
    score_font = pygame.font.Font(None, 36)
    score_text = score_font.render(f"Score: {score}", True, BLUE)
    screen.blit(score_text, (300, screen.get_height()-25))

    control_volume()
    # Update the display
    pygame.display.flip()
    player_moved = False
    reseted = False

# Quit Pygame
pygame.quit()
