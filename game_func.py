import pydirectinput
import pyautogui
import time

class Player:
    def __init__(self):
        self.start()

    def start(self):
        self.jump()

    def jump(self):
        pydirectinput.press('space')

    def duck(self):
        pydirectinput.press('down')

def reward_func():
    if pyautogui.locateOnScreen('restart_white.png')==None or pyautogui.locateOnScreen('restart_black.png')==None:
        return 1
    else:
        return -10

def test():
    time.sleep(5)
    p = Player()
    for _ in range(200):
        if reward_func()==-10:
            p.start()
        p.jump()
        time.sleep(1)
        p.duck()

if __name__=='__main__':
    test()
