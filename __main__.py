import math
import sys
from copy import deepcopy
from random import *

import pygame
import pygame_gui
from PIL import Image
from pygame.locals import SRCALPHA

try:
    with open('save.txt') as f:
        lines = f.readlines()
except Exception:
    lines = 0
bestscore = lines[0]


class DungeonMap:
    def __init__(self):
        self.mrsize = None
        self.mapArr = None
        self.size_y = None
        self.size_x = None
        self.roomList = []
        self.cList = []

    def make_map(self, xsize, ysize, fail, b1, mrooms, mrsize):
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
            self.placeRoom(l, w, x, y, xsize, ysize, 6, 0)
        failed = 0
        while failed < fail:
            chooseRoom = randrange(len(self.roomList))
            ex, ey, ex2, ey2, et = self.makeExit(chooseRoom)
            feature = randrange(100)
            if feature < b1:
                w, l, t = self.makeCorridor()
            else:
                w, l, t = self.makeRoom()
            roomDone = self.placeRoom(l, w, ex2, ey2, xsize, ysize, t, et)
            if roomDone == 0:
                failed += 1
            elif roomDone == 2:
                if self.mapArr[ey2][ex2] == 0:
                    if randrange(100) < 7:
                        self.makePortal(ex, ey)
                    failed += 1
            else:
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

    @staticmethod
    def makeCorridor():
        clength = randrange(18) + 3
        heading = randrange(4)
        wd = 0
        lg = 0
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
                    if self.mapArr[ypos + j][xpos + k] != 1:
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
themap = DungeonMap()
themap.make_map(startx, starty, 700, 15, 10, mrsize=7)

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

tile_size = 10
map_width = themap.size_x * tile_size
map_height = themap.size_y * tile_size

ui_width = 300

screen_width = map_width + ui_width
screen = pygame.display.set_mode((screen_width, map_height), pygame.SRCALPHA)
pygame.display.set_caption('Sonar Dungeon')

gui_manager = pygame_gui.UIManager((screen_width, map_height))

list_rect = pygame.Rect(map_width, 0, ui_width, map_height)
list_container = pygame_gui.elements.UIPanel(
    relative_rect=list_rect,
    manager=gui_manager,
)

list_height = 20
scrollable_list = pygame_gui.elements.UIScrollingContainer(
    relative_rect=pygame.Rect((0, 0), (ui_width, map_height)),
    manager=gui_manager,
    container=list_container,
)
print(map_height)

player_color = BLUE
player_size = tile_size
player_x, player_y = 0, 0
player_x += ui_width // tile_size

while True:
    player_x = randint(0, themap.size_x - 1)
    player_y = randint(0, themap.size_y - 1)
    if themap.mapArr[player_y][player_x] == 0:
        break


class Player:
    def __init__(self, color, size, mapArr):
        self.color = color
        self.size = size
        self.mapArr = mapArr
        self.x, self.y = self.find_start_position()
        self.collision_circles = []
        self.health = 100
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def take_damage(self, damage):
        self.health -= damage

    def find_start_position(self):
        while True:
            x = randint(0, len(self.mapArr[0]) - 1)
            y = randint(0, len(self.mapArr) - 1)
            if self.mapArr[y][x] == 0:
                return x, y

    def draw(self, screen_):
        rect = pygame.Rect(self.x * self.size, self.y * self.size, self.size, self.size)
        pygame.draw.rect(screen_, self.color, rect)

    def move(self, dx, dy, enemies_):
        new_x = self.x + dx
        new_y = self.y + dy

        for enemy_ in enemies_:
            if new_x == enemy_.x and new_y == enemy_.y:
                enemy_.take_damage(randint(40, 100))
                if enemy_.killed is True:
                    continue
                return

        if len(self.mapArr[0]) > new_x >= 0 == self.mapArr[new_y][new_x] and 0 <= new_y < len(self.mapArr):
            self.x = new_x
            self.y = new_y
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def draw_rays(self, tile_size_):
        mapArr = self.mapArr
        ray_length = 3 * tile_size_
        rays = 16

        for angle in range(0, 360, int(360 / rays)):
            radians = math.radians(angle)
            dx = int(math.cos(radians) * ray_length)
            dy = int(math.sin(radians) * ray_length)

            start_pos = (self.x * tile_size_ + tile_size_ // 2, self.y * tile_size_ + tile_size_ // 2)
            end_pos = (start_pos[0] + dx, start_pos[1] + dy)
            collision_pos = self.cast_ray(start_pos, end_pos, mapArr, tile_size_)
            if collision_pos != end_pos:
                self.collision_circles.append((collision_pos, 255))

    def decrease_transparency(self, decrement=5):
        new_circles = []

        for circle_pos, transparency in self.collision_circles:
            new_transparency = max(0, transparency - decrement)
            new_circles.append((circle_pos, new_transparency))

        self.collision_circles = new_circles

    def draw_collision_circles(self, screen_, radius):
        for circle_pos, transparency in self.collision_circles:
            circle_surface = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)

            pygame.draw.circle(circle_surface, (255, 255, 0, transparency), (radius, radius), radius)

            screen_.blit(circle_surface, (int(circle_pos[0]) - radius, int(circle_pos[1]) - radius))

    @staticmethod
    def cast_ray(start, end, mapArr, tile_size_):
        step_size = 5

        direction = (end[0] - start[0], end[1] - start[1])
        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        direction = (direction[0] / length, direction[1] / length)

        steps = 0

        while steps < length:
            steps += step_size
            current_pos = (start[0] + direction[0] * steps, start[1] + direction[1] * steps)

            tile_x = int(current_pos[0] // tile_size_)
            tile_y = int(current_pos[1] // tile_size_)

            if 0 <= tile_x < len(mapArr[0]) and 0 <= tile_y < len(mapArr):
                if mapArr[tile_y][tile_x] == 2:
                    return current_pos

        return end


player = Player(BLUE, tile_size, themap.mapArr)


def check_player_enemy_collisions(player_, enemies_):
    for enemy_ in enemies_:
        if (
                abs(player_.x - enemy_.x) <= 1
                and abs(player_.y - enemy_.y) <= 1
                and player_.health > 0
                and enemy_.killed is False
        ):
            player_.take_damage(enemy_.damage)


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
        self.damage = randint(20, 40)
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

    def distance_to_player(self, player_x_, player_y_):
        return math.sqrt((self.x - player_x_) ** 2 + (self.y - player_y_) ** 2)

    def draw(self, screen_, player_x_, player_y_):
        if self.killed is False:
            distance = self.distance_to_player(player_x_, player_y_)
            transparency = max(0, min(255, int((5 - distance) * 5.1 * 50 / 5)))

            enemy_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)

            pygame.draw.rect(enemy_surface, (255, 0, 0, transparency), (0, 0, self.size, self.size))

            screen_.blit(enemy_surface, (self.x * self.size, self.y * self.size))

    def move_randomly(self):
        if self.steps_in_preferred_direction < self.max_steps_in_preferred_direction:
            self.steps_in_preferred_direction += 1
        else:
            self.preferred_direction = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            self.steps_in_preferred_direction = 0

        possible_moves = self.get_neighbors((self.x, self.y))
        if possible_moves:
            if random() < 0.2:
                self.preferred_direction = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

            dx, dy = self.preferred_direction
            new_x = self.x + dx
            new_y = self.y + dy

            if self.mapArr[new_y][new_x] == 0:
                self.x = new_x
                self.y = new_y
            else:
                self.steps_in_preferred_direction = self.max_steps_in_preferred_direction
                self.move_randomly()

    def move_towards_player(self, player_x_, player_y_):
        path = self.a_star_pathfinding((self.x, self.y), (player_x_, player_y_))

        if path:
            next_x, next_y = path[0]
            if (next_x, next_y) == (player_x_, player_y_):
                return

            if self.mapArr[next_y][next_x] == 0:
                self.x = next_x
                self.y = next_y

    def update(self, player_x_, player_y_):
        if self.can_see_player(player_x_, player_y_):
            self.target_position = (player_x_, player_y_)
        elif self.last_player_position is not None:
            self.target_position = self.last_player_position

        if self.target_position is not None:
            self.move_towards_player(*self.target_position)
            if (self.x, self.y) == self.target_position:
                self.target_position = None
        else:
            self.move_randomly()

    def can_see_player(self, player_x_, player_y_):
        return abs(player_x_ - self.x) <= 5 and abs(player_y_ - self.y) <= 5

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

    @staticmethod
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [neighbor for neighbor in neighbors if
                len(self.mapArr[0]) > neighbor[0] >= 0 == self.mapArr[neighbor[1]][neighbor[0]]
                and 0 <= neighbor[1] < len(self.mapArr)]


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

    def draw(self, screen_, color=(245, 222, 179)):
        if self.active:
            frame_index = int(self.current_frame) % len(self.frames)
            frame = self.frames[frame_index]

            colored_frame = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
            colored_frame.fill((*color, 0))
            frame.blit(colored_frame, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

            rect = frame.get_rect(center=self.position)
            screen_.blit(frame, rect.topleft)


num_enemies = 5
enemies = [Enemy(RED, tile_size, themap.mapArr) for _ in range(num_enemies)]
im = Image.open("eea.png")
effects = [Effect("eea.png", columns=9, scale=1 / im.size[1] * tile_size, speed=0.1) for i in range(num_enemies)]


class Particle:
    def __init__(self, position, color, size):
        self.position = position
        self.color = color
        self.size = size
        self.transparency = 255

    def decrease_transparency(self, decrement=5):
        self.transparency = max(0, self.transparency - decrement)

    def draw(self, screen_):
        particle_surface = pygame.Surface((2 * self.size, 2 * self.size), pygame.SRCALPHA)

        pygame.draw.circle(particle_surface, (self.color[0], self.color[1], self.color[2], self.transparency),
                           (self.size, self.size), self.size)

        screen_.blit(particle_surface, (int(self.position[0]) - self.size, int(self.position[1]) - self.size))


class PlayerSquare:
    def __init__(self, color, size, transparency, image_path_):
        self.color = color
        self.size = size
        self.position = (0, 0)
        self.active = False
        self.particles = []
        self.transparency = transparency
        self.image = pygame.image.load(image_path_).convert_alpha() if image_path_ else None

    def activate(self, position):
        self.position = position
        self.active = True

    def deactivate(self, enemies_):
        if self.active:
            particle_color = (0, 255, 0)
            particle_size = 2
            particle_count = 10

            for enemy_ in enemies_:
                if self.check_collision(enemy_):
                    for _ in range(particle_count):
                        particle_ = Particle(
                            (enemy_.x * tile_size + tile_size // 2, enemy_.y * tile_size + tile_size // 2),
                            particle_color, particle_size)
                        self.particles.append(particle_)

            self.active = False

    def update(self, enemies_):
        if self.active:
            for enemy_ in enemies_:
                if self.check_collision(enemy_):
                    pass

    def check_collision(self, enemy_):
        x, y = self.position
        enemy_x, enemy_y = enemy_.x * tile_size, enemy_.y * tile_size
        return abs(x - enemy_x) < self.size and abs(y - enemy_y) < self.size

    def draw(self, screen_):
        if self.active:
            square_surface = pygame.Surface((2 * self.size, 2 * self.size), SRCALPHA)

            pygame.draw.rect(square_surface, (self.color[0], self.color[1], self.color[2], self.transparency),
                             (0, 0, 2 * self.size, 2 * self.size))

            screen_.blit(square_surface, (int(self.position[0]) - self.size, int(self.position[1]) - self.size))

            if self.image:
                screen_.blit(self.image, (int(self.position[0]) - self.size, int(self.position[1]) - self.size))

        # Draw particles
        for particle_ in self.particles:
            particle_.decrease_transparency(5)
            particle_.draw(screen_)

            if particle_.transparency == 0:
                self.particles.remove(particle_)


player_square = PlayerSquare((0, 0, 255), 100, 25, "radar area.png")


class DungeonObject:
    def __init__(self, color, size):
        self.color = color
        self.size = size
        self.position = (0, 0)
        self.visible = False
        self.image = None
        self.original_image = None
        self.transparency = 255
        self.rect = pygame.Rect(self.position[0], self.position[1], self.size, self.size)

    def place_randomly(self, dungeon_map, tile_size_):
        while True:
            x = randint(0, len(dungeon_map[0]) - 1)
            y = randint(0, len(dungeon_map) - 1)
            if dungeon_map[y][x] == 0:
                self.position = (x * tile_size_, y * tile_size_)
                self.rect = pygame.Rect(self.position[0], self.position[1], self.size, self.size)
                break

    def update_visibility(self, player_position, visibility_range_):
        distance_to_player = math.hypot(player_position[0] - self.position[0], player_position[1] - self.position[1])
        self.visible = distance_to_player <= visibility_range_

        max_transparency_distance = visibility_range_
        transparency_ratio = min(1, distance_to_player / max_transparency_distance)
        self.transparency = int((1 - transparency_ratio) * 255)

    def draw(self, screen_):
        if self.visible:
            if self.image is None:
                self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
                self.original_image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
                pygame.draw.rect(self.original_image, self.color, (0, 0, self.size, self.size))
                self.image.blit(self.original_image, (0, 0))
            self.image.set_alpha(self.transparency)
            screen_.blit(self.image, self.position)


dungeon_object = DungeonObject((255, 165, 0), tile_size)
dungeon_object.place_randomly(themap.mapArr, tile_size)

sound = pygame.mixer.Sound("servernaya-gudenie-ventilyatorov-37229.wav")
sound.set_volume(0)
move_sound = pygame.mixer.Sound("move sound.mp3")
move_sound.set_volume(0.125)


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


sound.play(loops=-1)


def control_volume():
    distance = float("inf")
    for i in enemies:
        distance = min(distance, calculate_distance((player.x, player.y), (i.x, i.y)))
        if i.killed is True:
            distance = float("inf")

    max_distance = 25

    exponent = 15
    volume = max(0, min(1, math.exp(-exponent * distance / max_distance)))

    sound.set_volume(volume * 0.125)


class PowerUp(DungeonObject):
    def __init__(self, color, size):
        super().__init__(color, size)


power_up = PowerUp((255, 165, 165), tile_size)
power_up.place_randomly(themap.mapArr, tile_size)

visibility_range = 100
clock = pygame.time.Clock()
running = True
player_moved = False
reseted = False

score = 0


def reset_dungeon_dead():
    themap = DungeonMap()
    themap.make_map(startx, starty, 700, 15, 10, mrsize=7)
    player.mapArr = themap.mapArr
    player.collision_circles.clear()
    player.x, player.y = player.find_start_position()
    for el in enemies:
        el.mapArr = themap.mapArr
        el.x, el.y = el.find_start_position()
        el.killed = False
        el.last_player_position = None
        el.target_position = None
        el.hp = randint(75, 100)
    dungeon_object.place_randomly(themap.mapArr, tile_size)
    power_up.place_randomly(themap.mapArr, tile_size)
    for i in themap.mapArr:
        print(i)


def reset_dungeon():
    themap = DungeonMap()
    themap.make_map(startx, starty, 700, 15, 10 + score * 10, mrsize=7)
    player.mapArr = themap.mapArr
    player.x, player.y = player.find_start_position()
    for el in enemies:
        el.mapArr = themap.mapArr
        el.x, el.y = el.find_start_position()
        el.killed = False
        el.last_player_position = None
        el.target_position = None
        el.hp = randint(75 + score * 2, 100 + score * 2)

    dungeon_object.place_randomly(themap.mapArr, tile_size)
    power_up.place_randomly(themap.mapArr, tile_size)
    for i in themap.mapArr:
        print(i)


class InstructionBox:
    def __init__(self, font_images_, text_metrics):
        self.text = "instructions:\n\nawsd     to     move\nradar - space\nheal - r\nexplosion - q"
        self.font_images = font_images_
        self.text_metrics = text_metrics

    def draw(self, screen_):
        x, y = (100, 100)
        self.text_metrics.draw_text(screen_, self.text, (x, y))
        print("wtf")


class ImageToText:
    def __init__(self, image_path_, char_width_, char_height_, chars_):
        self.image = Image.open(image_path_)
        self.char_width = char_width_
        self.char_height = char_height_
        self.chars = chars_

    def extract_characters(self):
        char_images = []
        for y in range(0, self.image.size[1], self.char_height):
            for x in range(0, self.image.size[0], self.char_width):
                char_region = (x, y, x + self.char_width, y + self.char_height)
                char_images.append(self.image.crop(char_region))
        return char_images


class TextMetrics:
    def __init__(self, font_images_, char_gap, line_gap, starting_position=(0, 0)):
        self.font_images = font_images_
        self.char_gap = char_gap
        self.line_gap = line_gap
        self.starting_position = starting_position

    def calculate_starting_position(self):
        x, y = self.starting_position
        return x, y

    def calculate_starting_x(self):
        return self.starting_position[0]

    def calculate_width(self, text):
        chars_ = "abcdefghijklmnopqrstuvwxyz1234567890#,.!?:*%()+-/\=><"
        width = 0
        for char in text:
            if char == ' ':
                width += self.char_gap
            elif char == '\n':
                break
            try:
                char_index = chars_.index(char)
                width += self.font_images[char_index].get_width() + self.char_gap
            except ValueError:
                pass
        return width

    def calculate_height(self, text):
        height = 0
        for _ in text.split('\n'):
            height += self.font_images[0].get_height() + self.line_gap
        return height

    def draw_text(self, screen_, text, position):
        chars_ = "abcdefghijklmnopqrstuvwxyz1234567890#,.!?:*%()+-/\=><"
        x, y = position
        max_width, max_height = screen_.get_width(), screen_.get_height()

        for char in text:
            if char == '\n':
                x = position[0]
                y += self.font_images[0].get_height() + self.line_gap
            elif char == ' ':
                space_width = self.calculate_width(' ') + 5
                x += space_width
            else:
                try:
                    char_index = chars_.index(char)
                    char_image = self.font_images[char_index]
                    char_width_ = char_image.get_width()
                    char_height_ = char_image.get_height()

                    if x + char_width_ <= max_width and y + char_height_ <= max_height:
                        screen_.blit(char_image, (x, y))
                        x += char_width_ + self.char_gap
                    else:
                        x = self.starting_position[0]
                        y += char_height_ + self.line_gap
                        if y + char_height_ > max_height:
                            break
                        screen_.blit(char_image, (x, y))
                        x += char_width_ + self.char_gap

                except ValueError:
                    pass


class Button:
    def __init__(self, position, text, font_images_, text_metrics, hover_font_images_):
        self.position = position
        self.text = text
        self.font_images = font_images_
        self.text_metrics = deepcopy(text_metrics)
        self.hovered = False
        self.hover_font_images = hover_font_images_

    def draw(self, screen_):
        if self.hovered:
            self.text_metrics.font_images = self.hover_font_images
        else:
            self.text_metrics.font_images = self.font_images

        self.text_metrics.draw_text(screen_, self.text, (self.position[0], self.position[1]))

    @staticmethod
    def collide_point(rect, point):
        x, y = point
        return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

    def is_hovered(self, mouse_pos):
        x, y = mouse_pos
        text_width = self.text_metrics.calculate_width(self.text)
        text_height = self.text_metrics.calculate_height(self.text)
        text_rect = pygame.Rect(self.position[0], self.position[1], text_width, text_height)

        self.hovered = text_rect.collidepoint(x, y)
        return self.hovered


in_menu = True


class EndWindow:
    def __init__(self, font_images_, text_metrics):
        self.text = "goodbye!"
        self.font_images = font_images_
        self.text_metrics = text_metrics

    def draw(self, screen_):
        x, y = (200, 200)
        self.text_metrics.draw_text(screen_, self.text, (x, y))

    @staticmethod
    def close_after_delay(delay):
        pygame.time.delay(delay)
        pygame.quit()
        sys.exit()


class PygameTextDisplay:
    def __init__(self, font_images_, hover_font_images_, screen_size, text_size, char_gap, line_gap,
                 starting_position=(0, 0), close=False):
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()
        self.font_images = [self.image_to_surface(image, text_size) for image in font_images_]
        self.hover_font_images = [self.image_to_surface(image, text_size) for image in hover_font_images_]
        self.text_metrics = TextMetrics(self.font_images, char_gap, line_gap, starting_position)

        button_text = "play!"
        button_position = (150, 300)
        self.button = Button(button_position, button_text, self.font_images, self.text_metrics, self.hover_font_images)
        self.instruction_button = Button((150, 400), "instructions", self.font_images, self.text_metrics,
                                         self.hover_font_images)
        self.instruction_box = InstructionBox(self.font_images, self.text_metrics)
        self.end_window = EndWindow(self.font_images, self.text_metrics)
        self.close = close

    @staticmethod
    def image_to_surface(image, text_size):
        img_data = image.tobytes()
        img_surface = pygame.image.fromstring(img_data, image.size, image.mode)
        return pygame.transform.scale(img_surface, (text_size, text_size))

    def run(self, text):
        if self.close:
            self.end_window.draw(self.screen)
            pygame.display.flip()
            self.end_window.close_after_delay(2000)
        else:
            running_ = True
            inspecting = False
            while running_:
                for event_ in pygame.event.get():

                    if event_.type == pygame.QUIT:
                        self.close = True
                        running_ = False
                    elif event_.type == pygame.MOUSEMOTION:
                        mouse_pos = event_.pos
                        self.button.is_hovered(mouse_pos)
                        self.instruction_button.is_hovered(mouse_pos)
                    elif event_.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = event_.pos
                        inspecting = False
                        if self.button.is_hovered(mouse_pos):
                            running_ = False
                        elif self.instruction_button.is_hovered(mouse_pos):
                            inspecting = True

                if inspecting:
                    self.screen.fill((0, 0, 0))
                    self.instruction_box.draw(self.screen)
                    pygame.display.flip()
                else:
                    self.screen.fill((0, 0, 0))
                    self.text_metrics.draw_text(self.screen, text, (150, 120))
                    self.button.draw(self.screen)
                    self.instruction_button.draw(self.screen)
                    pygame.display.flip()
                self.clock.tick(60)


def menu(close=False):
    image_path_ = "font 8x7 outline.png"
    hover_image_path_ = "font 8x7.png"

    char_width_, char_height_ = 8, 8
    chars_ = "abcdefghijklmnopqrstuvwxyz1234567890#,.!?:*%()+-/\=><"

    image_to_text_ = ImageToText(image_path_, char_width_, char_height_, chars_)

    font_images_ = image_to_text_.extract_characters()

    hover_image_to_text_ = ImageToText(hover_image_path_, char_width_, char_height_, chars_)

    hover_font_images_ = hover_image_to_text_.extract_characters()

    screen_size = (800, 600)
    text_size = 50
    char_gap = 0
    line_gap = 10

    pygame_text_display = PygameTextDisplay(font_images_, hover_font_images_, screen_size, text_size,
                                            char_gap, line_gap, close=close)

    text_to_display = "sonar\ndungeon"

    pygame_text_display.run(text_to_display)
    return pygame_text_display.close


powerups_d = {
    "heal": 0,
    "explosion": 10,
    "radar": 10
}
powerups_da = {
    "heal": False,
    "explosion": False,
    "radar": False
}
explosion_square = PlayerSquare((255, 10, 0), 50, 25, "explosion area.png")


def image_to_surface(image, text_size):
    img_data = image.tobytes()
    img_surface = pygame.image.fromstring(img_data, image.size, image.mode)
    return pygame.transform.scale(img_surface, (text_size, text_size))


image_path = "font 8x7 outline.png"
hover_image_path = "font 8x7.png"

char_width, char_height = 8, 8
chars = "abcdefghijklmnopqrstuvwxyz1234567890#,.!?:*%()+-/\=><"

image_to_text = ImageToText(image_path, char_width, char_height, chars)

font_images = image_to_text.extract_characters()

hover_image_to_text = ImageToText(hover_image_path, char_width, char_height, chars)

hover_font_images = hover_image_to_text.extract_characters()
font_images = [image_to_surface(image, 20) for image in font_images]
texts = TextMetrics(font_images, 0, 0)
while running:
    time_delta = clock.tick(60) / 1000.0
    if in_menu:
        if menu():
            break
        in_menu = False
        screen_width = map_width + ui_width
        screen = pygame.display.set_mode((screen_width, map_height), pygame.SRCALPHA)
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
                if powerups_da["radar"] is True:
                    player_square.activate(pygame.mouse.get_pos())
                    for i in range(len(effects)):
                        if player_square.check_collision(enemies[i]) and enemies[i].killed is False:
                            effects[i].activate([enemies[i].x * tile_size, enemies[i].y * tile_size])
                    powerups_da["radar"] = False
                    powerups_d["radar"] -= 1

                elif powerups_d["radar"] > 0:
                    powerups_da["radar"] = True
            elif event.key == pygame.K_q:
                if powerups_da["explosion"]:
                    explosion_square.activate(pygame.mouse.get_pos())
                    for i in range(len(effects)):
                        if explosion_square.check_collision(enemies[i]) and enemies[i].killed is False:
                            enemies[i].take_damage(60)
                    powerups_da["explosion"] = False
                    powerups_d["explosion"] -= 1

                elif powerups_d["explosion"] > 0:
                    powerups_da["explosion"] = True
            elif event.key == pygame.K_r:
                if powerups_d["heal"] > 0:
                    player.health += 20
                    if player.health > 100:
                        player.health = 100
                    powerups_d["heal"] -= 1
            elif event.key == pygame.K_ESCAPE:
                menu()
                screen_width = map_width + ui_width
                screen = pygame.display.set_mode((screen_width, map_height), pygame.SRCALPHA)
                reset_dungeon_dead()
        gui_manager.process_events(event)

    gui_manager.update(time_delta)

    player.decrease_transparency(1)

    cursor_position = pygame.mouse.get_pos()
    player_square.activate(cursor_position)
    explosion_square.activate(cursor_position)

    player_square.update(enemies)
    if player_moved:
        move_sound.play()
        for enemy in enemies:
            enemy.update(player.x, player.y)

    dungeon_object.update_visibility((player.x * tile_size, player.y * tile_size), visibility_range)

    power_up.update_visibility((player.x * tile_size, player.y * tile_size), visibility_range)

    screen.fill(BLACK)
    player.draw(screen)
    dungeon_object.draw(screen)
    power_up.draw(screen)
    if powerups_da["radar"]:
        player_square.draw(screen)
    if powerups_da["explosion"] is True:
        explosion_square.draw(screen)
    if player_moved:
        player.draw_rays(tile_size)
    player.draw_collision_circles(screen, 2)
    for i in range(len(enemies)):
        enemies[i].draw(screen, player.x, player.y)

        effects[i].draw(screen)
        effects[i].update()

    for particle in player_square.particles:
        particle.decrease_transparency(5)
        particle.draw(screen)

        if particle.transparency == 0:
            player_square.particles.remove(particle)
    c = 0
    gui_manager.draw_ui(screen)
    dungeon_object.draw(screen)
    if player_moved:
        check_player_enemy_collisions(player, enemies)

    if power_up.position == (player.x * tile_size, player.y * tile_size):
        r = randint(1, 3)
        if r == 1:
            powerups_d["radar"] += 1
        elif r == 2:
            powerups_d["explosion"] += 1
        elif r == 3:
            powerups_d["heal"] += 1
        power_up.position = (-100, -100)

    if type(bestscore) is list:
        if bestscore[0] != "":
            bestscore = max(int(bestscore[0]), score)
        else:
            bestscore = 0
    else:
        bestscore = max(int(bestscore), score)

    if dungeon_object.position == (player.x * tile_size, player.y * tile_size) and reseted is False:
        reset_dungeon()
        reseted = True
        score += 1

    if player.health <= 0:
        reset_dungeon_dead()
        score = 0
        player.health = 100
    control_volume()
    texts.draw_text(screen, f'explosion\n{powerups_d["explosion"]}\nradar\n{powerups_d["radar"]}\nheal'
                            f'\n{powerups_d["heal"]}\nbest score:{bestscore}\nscore: {score}\nhealth'
                            f': {player.health}', (520, 50))

    pygame.display.flip()
    player_moved = False
    reseted = False

with open("save.txt", "w") as txt_file:
    txt_file.write(str(bestscore))
menu(close=True)
pygame.quit()
