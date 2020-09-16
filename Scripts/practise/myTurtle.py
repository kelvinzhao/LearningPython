import turtle

wn = turtle.Screen()
a = 1
ct = turtle.Turtle()
ct.color('red')
ct.fillcolor('yellow')
ct.speed(10)
ct.shape('arrow')
ct.shapesize(1)
ct.begin_fill()
while True:
    ct.forward(300)
    ct.left(170)
    if abs(ct.pos()) < 1:
        break

ct.end_fill()
wn.exitonclick()
