__author__ = 'miguel.herraez'

# --- imports ---
# import Box2D # The main library
# from Box2D.b2 import * # This maps Box2D.b2Vec2 to vec2 (and so on)
import pygame
from pygame.locals import *
from Box2D import *
import RVE
import numpy as np
import time
import os


# --- functions

def my_draw_polygon(polygon, body):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT_PX - v[1]) for v in vertices]
    material = body.userData.material  # Fiber material
    fibercolour, textcolour, edgecolour = RVE.colourFiber(material)
    pygame.draw.polygon(screen, colors[fibercolour], vertices, 0)


b2PolygonShape.draw = my_draw_polygon


def my_draw_circle(circle, body):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT_PX - position[1])
    material = body.userData.material
    fibercolour, textcolour, edgecolour = RVE.colourFiber(material)
    pygame.draw.circle(screen, colors[fibercolour], [int(x) for x in position], int(circle.radius * PPM))


b2CircleShape.draw = my_draw_circle


def my_draw_edge(edge, body):
    vertices = [(body.transform * v) * PPM for v in edge.vertices]
    vertices = [(v[0], SCREEN_HEIGHT_PX - v[1]) for v in vertices]
    # vertices = fix_vertices([body.transform*edge.vertex1*PPM, body.transform*edge.vertex2*PPM])
    pygame.draw.line(screen, (0, 255, 64), vertices[0], vertices[1])


b2EdgeShape.draw = my_draw_edge


# def _draw_loop(loop, body, fixture):
#     transform = body.transform
#     vertices = fix_vertices([transform*v*PPM for v in loop.vertices])
#     v1 = vertices[-1]
#     for v2 in vertices:
#         pygame.draw.line(screen, colors[body.type], v1, v2)
#         v1 = v2
#
# b2LoopShape.draw = _draw_loop

# def createFiberBody(body, fiber, density=1., friction=0.10):
#
#     if fiber.geometry.upper() == RVE.CIRCULAR:
#         body.CreateCircleFixture(radius=fiber.L*0.5/.99, density=density, friction=friction)
#     else:
#         raise TypeError, 'Fiber geometry not supported (yet!): %s' % fiber.geometry

def createBody(fiber, center, tol=0., dynamic=True, periodic=0):
    if tol:
        fiber.set_size(fiber.L + tol)

    if dynamic:
        body = world.CreateDynamicBody(position=center,
                                       #restitution=1.0,
                                       userData=fiber)
                                       #density=1.0)
    else:
        body = world.CreateStaticBody(position=center,
                                      #restitution=1.0,
                                      userData=fiber)
                                      #density=1.0)

    # Polygonal fibres
    if (fiber.geometry.upper() == RVE.POLYGON) and (len(fiber.vertices[0]) < b2_maxPolygonVertices):
        new_x = fiber.vertices[0] - fiber.center[0]
        new_y = fiber.vertices[1] - fiber.center[1]
        verts = zip(new_x, new_y)[-1:0:-1]
        myShape = b2PolygonShape(vertices=verts)
        body.CreateFixture(shape=myShape, density=1, friction=0.1)
        if periodic != 0:
            new_x += periodicSettings[periodic]['dx']
            new_y += periodicSettings[periodic]['dy']
            verts = zip(new_x, new_y)[-1:0:-1]
            myShape = b2PolygonShape(vertices=verts)
            body.CreateFixture(shape=myShape, density=1, friction=0.1)
    # Circular fibres
    elif fiber.geometry.upper() == RVE.CIRCULAR:
        myShape = b2CircleShape(radius=fiber.L * 0.5)
        body.CreateFixture(shape=myShape, density=1, friction=0.1)
        if periodic != 0:
            centerP = (periodicSettings[periodic]['dx'],
                       periodicSettings[periodic]['dy'])
            myShape = b2CircleShape(radius=fiber.L * 0.5, pos=centerP)
            body.CreateFixture(shape=myShape, density=1, friction=0.1)
    # Vertices-based fibres
    else:
        new_x = fiber.vertices[0] - fiber.center[0]
        new_y = fiber.vertices[1] - fiber.center[1]
        verts = zip(new_x, new_y)[-1::-1]
        for i in range(1, len(verts)):
            # myShape = b2CircleShape(center=verts[i], radius=fiber.L*0.15)
            myShape = b2PolygonShape(vertices=[verts[i - 1], verts[i], (0, 0)])
            body.CreateFixture(shape=myShape, density=1)
        myShape = b2PolygonShape(vertices=[verts[i], verts[0], (0, 0)])
        body.CreateFixture(shape=myShape, density=1, friction=0.1)
        if periodic != 0:
            offset = (periodicSettings[periodic]['dx'],
                      periodicSettings[periodic]['dy'])
            new_x += offset[0]
            new_y += offset[1]
            verts = zip(new_x, new_y)[-1::-1]
            for i in range(1, len(verts)):
                myShape = b2PolygonShape(vertices=[verts[i - 1], verts[i], offset])
                body.CreateFixture(shape=myShape, density=1)
            myShape = b2PolygonShape(vertices=[verts[i], verts[0], offset])
            body.CreateFixture(shape=myShape, density=1, friction=0.1)

    # else:
    #     raise NotImplementedError('Fibre shape {} not implemented yet'.format(fiber.geometry))

    return body


# --- constants ---
TARGET_FPS = 0.5
TIME_STEP = 1.0 / TARGET_FPS

colors = {
    b2_staticBody: (255, 255, 255),  # white
    b2_dynamicBody: (192, 192, 192),  # grey
    b2_kinematicBody: (255, 0, 255),  # magenta
    'background': (105, 105, 105),  # background - dimgrey
    'silver': (192, 192, 192),
    'black': (0, 0, 0),
    'beige': (200, 200, 179),
    'khaki': (240, 230, 140),
    'cadetblue': (95, 158, 160),
    'white': (255, 255, 255),
    'grey': (128, 128, 128),
    'palegreen': (152, 251, 152),
}

if __name__ == "__main__":
    # Read a microstructure
    # directory = r'D:\ABAQUSWD\fracture_SSY\Previous tests\Micro7_Restart'
    # directory = r'C:\Users\miguel.herraez\Desktop\VIPPER project\Microstructures'
    directory = r'..\Microstructures'
    fileName = r'13_box2d'
    micro = RVE.Microstructure(read_microstructure=os.path.join(directory, fileName))
    # micro.plot_rve(mute=False, numbering=True, title=True)

    global PPM
    global SCREEN_HEIGHT_PX
    global screen

    lmax = 600.0  # Pixels
    margin = 0.1 * lmax
    tolerance = 0.5
    gravity = (0, 0)
    autoStop = True

    maxDimension = float(max(micro.rveSize))
    PPM = lmax / maxDimension  # pixels per unit length
    # print 'PPM = ', PPM

    w, h = micro.rveSize
    RVE_WIDTH_PX = round(w * PPM)
    RVE_HEIGHT_PX = round(h * PPM)
    SCREEN_OFFSETX_PX, SCREEN_OFFSETY_PX = margin, margin

    SCREEN_WIDTH_PX = int(RVE_WIDTH_PX + 2 * SCREEN_OFFSETX_PX)
    SCREEN_HEIGHT_PX = int(RVE_HEIGHT_PX + 2 * SCREEN_OFFSETY_PX)

    # print 'Screen (pixels): %d x %d' % (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
    # print 'RVE (pixels): %d x %d' % (RVE_WIDTH_PX, RVE_HEIGHT_PX)

    W = SCREEN_WIDTH_PX / PPM
    H = SCREEN_HEIGHT_PX / PPM
    margin_x, margin_y = SCREEN_OFFSETX_PX / PPM, SCREEN_OFFSETY_PX / PPM

    # Handler for periodic fibres
    periodicSettings = {
        # Parameters to generate the periodic fiber of a p1-fiber (p3)
        1: {'dx': w, 'dy': 0.0, 'limit': h,  # 'localAnchor': (W,0),
            'moveAxis': (0, 1), 'constraintAxis': (1, 0), 'margin': margin_y},

        # Parameters to generate the periodic fiber of a p4-fiber (p2)
        4: {'dx': 0.0, 'dy': h, 'limit': w,  # 'localAnchor': (0,H),
            'moveAxis': (1, 0), 'constraintAxis': (0, 1), 'margin': margin_x},
    }

    # --- pygame setup ---
    screen = pygame.display.set_mode((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), 0, 32)
    pygame.display.set_caption('Physical compaction')
    clock = pygame.time.Clock()

    # --- pybox2d world setup ---
    # Create the world
    world = b2World(gravity=gravity, doSleep=True)
    # world = b2World(doSleep=True)

    # Static edges to hold the fibers inside
    shapes = [
        b2EdgeShape(vertices=[(margin_x + tolerance, margin_y), (margin_x + tolerance, H - margin_y)]),  # Left edge
        b2EdgeShape(vertices=[(margin_x, H - margin_y - tolerance), (W - margin_x, H - margin_y - tolerance)]),
        # Top edge
        b2EdgeShape(vertices=[(W - margin_x - tolerance, H - margin_y), (W - margin_x - tolerance, margin_y)]),
        # Right edge
        b2EdgeShape(vertices=[(W - margin_x, margin_y + tolerance), (margin_x, margin_y + tolerance)]),  # Bottom edge
    ]

    rve = world.CreateStaticBody(shapes=shapes)

    fiberBodies = list()
    numberOfDynamicBodies = 0
    myPrevPos = []
    for fiber in micro.sq:
        center = (fiber.center[0] + margin_x, fiber.center[1] + margin_y)
        # minDist = tolerance
        # print 'Time for distance: %.2f' % (time.time()-t1)
        p = fiber.period

        if p == 0:
            body = createBody(fiber, center, tolerance)
            numberOfDynamicBodies += 1
            fiberBodies.append(body)
            myPrevPos.append(body.position.length)

        elif p in [5, 6, 7, 8]:
            body = createBody(fiber, center, tolerance, dynamic=False)
            fiberBodies.append(body)
            myPrevPos.append(body.position.length)

        elif p in [1, 4]:
            body = createBody(fiber, center, tolerance, periodic=p)
            numberOfDynamicBodies += 1
            fiberBodies.append(body)
            myPrevPos.append(body.position.length)

            # Implement periodic constraint
            if p == 1:
                c = (center[0], 0)
            else:
                # p = 4
                c = (0, center[1])

            # Fiber - RVE
            world.CreatePrismaticJoint(
                bodyA=rve,
                bodyB=body,
                enableLimit=False,
                localAnchorA=c,
                localAnchorB=(0, 0),
                axis=periodicSettings[p]['moveAxis'],
                # lowerTranslation=-h,
                # upperTranslation=h,
            )

    # --- main game loop ---
    nbodies = len(fiberBodies)
    running = True
    time0 = time.time()
    time2stop = 0.1
    velAverageRecord = list()
    velMaxRecord = list()
    timeRecord = list()

    # micro.save_rve(filename='pre.txt')
    inc = 0
    while running:
        # Check the event queue
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False

        screen.fill(colors['background'])

        # Draw the fibers
        for body in fiberBodies:
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # velocity += body.linearVelocity.length
                try:
                    fixture.shape.draw(body)
                except:
                    print 'Could not draw body'

        if (time.time() - time0 >= time2stop) and autoStop:
            velocities = [abs(fiberBodies[i].position.length - myPrevPos[i]) / TIME_STEP for i in range(nbodies)]
            # velAverage = sum([body.linearVelocity.length for body in fiberBodies])/numberOfDynamicBodies
            # velMax = max([body.linearVelocity.length for body in fiberBodies])
            velAverage = sum(velocities) / numberOfDynamicBodies
            velMax = max(velocities)
            # print 'Average velocity %f' % velocity
            velAverageRecord.append(velAverage)
            velMaxRecord.append(velMax)
            timeRecord.append(time2stop)
            myPrevPos = [body.position.length for body in fiberBodies]
            if velMax < 0.05 and inc > 0:  # or (velocity < max(velocityRecord)/50.0):
                running = False
            time2stop += TIME_STEP

        # Draw the RVE boundaries
        for body in (rve,):
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                fixture.shape.draw(body)

        # Make Box2D simulate the physics of our world for one step.
        # Instruct the world to perform a single step of simulation. It is
        # generally best to keep the time step and iterations fixed.
        # See the manual (Section "Simulating the World") for further discussion
        # on these parameters and their implications.
        world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        # clock.tick(TARGET_FPS)
        inc += 1

    pygame.quit()

    print 'Total time: {0:.1f} s'.format(time.time()-time0)
    print 'Iterations: {0:d}'.format(len(velAverageRecord))
    # Plot average velocity history
    if autoStop:
        import matplotlib.pyplot as plt

        # plt.plot(timeRecord, velAverageRecord, 'bs-')
        plt.plot(timeRecord, velMaxRecord, 'r^-')
        plt.show()

    # print micro.fiber_volume()
    # micro.save_rve(filename='post.txt')
    # Update fibres position and plot final RVE
    for i, body in enumerate(fiberBodies):
        fiber = body.userData
        center = (body.position.x - margin_x, body.position.y - margin_y)
        if fiber.period in [1, 4]:
            # Find periodic fiber
            p = fiber.period
            fiberPeriodic = micro.findPeriodic(fiber)
            fiberPeriodic.set_center((center[0] + periodicSettings[p]['dx'],
                                      center[1] + periodicSettings[p]['dy']))
            fiberPeriodic.set_phi(fiberPeriodic.phi + body.angle * 180. / np.pi)
            fiberPeriodic.set_size(fiberPeriodic.L)  # Do not substract the tolerance of periodic fibres!

        fiber.set_center(center)
        fiber.set_phi(fiber.phi + body.angle * 180. / np.pi)
        fiber.set_size(fiber.L - tolerance)
    # micro.save_rve(filename='post_tol.txt')

    micro.save_rve(filename=fileName+'-box2d_after')
    micro.plot_rve(show_plot=True, filename=fileName+'-box2dafter', save=fileName+'-box2dafter')
