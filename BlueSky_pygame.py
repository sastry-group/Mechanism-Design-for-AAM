#!/usr/bin/env python
""" Pygame BlueSky start script """
from __future__ import print_function
import pygame as pg
import bluesky as bs
from bluesky.ui.pygame import splash


def start():
    """ Start the mainloop (and possible other threads) """
    splash.show()
    bs.init()
    bs.sim.operate()
    bs.scr.init()

    # Main loop for tmx object
    while not bs.sim.mode == bs.sim.end:
        bs.sim.update()   # Update sim
        bs.scr.update()   # GUI update

        # Restart traffic simulation:
        if bs.sim.mode == bs.sim.init:
            bs.sim.reset()
            bs.scr.objdel()     # Delete user defined objects

def cleanup():
    """ After the simulation is done, close the gui """
    bs.sim.stop()
    pg.quit()

    print('BlueSky normal end.')


if __name__ == '__main__':
    print("   *****   BlueSky Open ATM simulator *****")
    print("Distributed under GNU General Public License v3")
    # Run mainloop if BlueSky_pygame is called directly
    start()

    # Cleanup after returning from start()
    cleanup()
