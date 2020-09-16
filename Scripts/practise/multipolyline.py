import turtle
from math import sin, cos, pi

# setup screen and turtle
wn = turtle.Screen()
wn.setup(1500, 1000, 0, 0)
ct = turtle.Turtle()
ct.color('red')
ct.speed(0)
ct.shape('blank')

r = 300
num = 30
colorize = ['red', 'blue', 'cyan', 'orange', 'brown', 'black']

# define points
step = 180 / num
# 360/2pi = angle/x,  so x = angle * 2pi / 360
points_x = [r - r * cos(x * step * 2 * 2 * pi / 360) for x in range(0, num)]
points_y = [r * sin(x * step * 2 * 2 * pi / 360) for x in range(0, num)]

for i in range(0, num-1):
    ct.penup()
    for j in range(i+1, num):
        ct.goto(points_x[i], points_y[i])
        ct.pendown()
        ct.color(colorize[i % len(colorize)])
        ct.goto(points_x[j], points_y[j])

wn.exitonclick()
