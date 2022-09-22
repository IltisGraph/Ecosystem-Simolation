import os
import random
import neat
import pygame as pg


oldPlayers = []
FPS = 60
clock = pg.time.Clock()
Fenster = pg.display.set_mode((500, 500))

def main(genomes, config):

    global oldPlayers, FPS

    ge = []
    nets = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)



    running = True
    while running:
        clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    FPS = 144
                if event.key == pg.K_DOWN:
                    FPS = 60

        Fenster.fill((255, 255, 255))




        pg.display.update()





def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(main)


if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "config-forward.txt")
    run(config_path)

