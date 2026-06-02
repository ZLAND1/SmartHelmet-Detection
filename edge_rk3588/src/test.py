import os
import pygame

pygame.mixer.init()
script_dir = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(script_dir, "alarm.wav")
sound = pygame.mixer.Sound(sound_path)
sound.play()