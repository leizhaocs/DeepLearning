import socket
import gym
import pickle
import numpy as np
import struct
from PIL import Image

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

CHARSIZE  = 1    # num bytes of a char
INTSIZE   = 4    # num bytes of an int
FLOATSIZE = 4    # num bytes of a float

# convert image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

# send all data of [size] bytes
def recvall(conn, size):
    data = conn.recv(size)
    while (len(data) < size):
        new_data = conn.recv(size-len(data))
        data = data + new_data
    return data

# open socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    # wait for client to connect
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()

    # connected
    with conn:
        print('Connected by', addr)
        env = None

        while True:

            # read command, one char
            command = recvall(conn, CHARSIZE)
            if not command:
                break

            # 'm' -> create environment
            if command == b'm':
                size = recvall(conn, INTSIZE)
                size = int.from_bytes(size, byteorder='little')
                data = recvall(conn, size*CHARSIZE)
                env_name = data.decode('utf-8')
                env = gym.make(env_name)
                print("Environment "+env_name+" created.")

            # 'c' -> close environment
            elif command == b'c':
                env.close()
                print("Environment "+env_name+" closed.")

            # 'r' -> reset environment
            elif command == b'r':
                env.reset()

            # 'g' -> get image dimension
            elif command == b'g':
                screen = env.render(mode='rgb_array')

                top = int(screen.shape[0] * 0.4)
                bottom = int(screen.shape[0] * 0.8)
                screen = screen[top:bottom, :, :]
                image = Image.fromarray(screen)
                image = image.resize((90, 40))
                screen = np.array(image)
                screen = screen.astype(np.float32) / 255
                screen = rgb2gray(screen)
                screen = np.expand_dims(screen, axis=0)
                channel = screen.shape[0].to_bytes(INTSIZE, 'little')
                height = screen.shape[1].to_bytes(INTSIZE, 'little')
                width = screen.shape[2].to_bytes(INTSIZE, 'little')
                conn.sendall(channel)
                conn.sendall(height)
                conn.sendall(width)

            # 'p' -> render environment
            elif command == b'p':
                env.render()
                screen = env.render(mode='rgb_array')
                top = int(screen.shape[0] * 0.4)
                bottom = int(screen.shape[0] * 0.8)
                screen = screen[top:bottom, :, :]
                image = Image.fromarray(screen)
                image = image.resize((90, 40))
                screen = np.array(image)
                screen = screen.astype(np.float32) / 255
                screen = rgb2gray(screen)
                screen = np.expand_dims(screen, axis=0)
                screen = screen.tobytes()
                conn.sendall(screen)

            # 'd' -> one step in enviroment with discrete action
            elif command == b'd':
                action = recvall(conn, INTSIZE)
                action = int.from_bytes(action, byteorder='little')
                observation, reward, done, info = env.step(action)
                reward = struct.pack('f', reward)
                conn.sendall(reward)
                if done:
                    finished = 1
                    finished = finished.to_bytes(INTSIZE, 'little')
                    conn.sendall(finished)
                else:
                    finished = 0
                    finished = finished.to_bytes(INTSIZE, 'little')
                    conn.sendall(finished)

            # 'b' -> one step in enviroment with box action
            elif command == b'b':
                num = recvall(conn, INTSIZE)
                num = int.from_bytes(num, byteorder='little')
                action = recvall(conn, num*FLOATSIZE)
                action = np.frombuffer(action, dtype=np.float32)
                observation, reward, done, info = env.step(action)
                reward = struct.pack('f', reward)
                conn.sendall(reward)
                if done:
                    finished = 1
                    finished = finished.to_bytes(INTSIZE, 'little')
                    conn.sendall(finished)
                else:
                    finished = 0
                    finished = finished.to_bytes(INTSIZE, 'little')
                    conn.sendall(finished)

            # 't' -> close the connection
            elif command == b't':
                break

            else:
                print(command)
    
