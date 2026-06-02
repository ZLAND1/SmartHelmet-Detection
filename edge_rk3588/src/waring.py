import os
import pygame


def waring_alarm():
    pygame.mixer.init()
    # 使用脚本所在目录的绝对路径，确保任意工作目录都能运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sound_path = os.path.join(script_dir, "jingao_3d.wav")
    sound = pygame.mixer.Sound(sound_path)
    sound.play()