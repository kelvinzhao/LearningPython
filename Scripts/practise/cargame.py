
car_status = 'stop'
print('Hello, welcome to Car Game, you may input "help" to check out how '
      'to play')
while True:
    userinput = input('>')
    userinput = userinput.lower()
    if(userinput == 'help'):
        print('''
        Start - start the car
        Stop - stop the car
        Quit - quit this gamn
        help - show this message
        ''')
    elif(userinput == 'start'):
        if(car_status == 'stop'):
            print(' The car is ready .... set ....Go!')
            car_status = 'start'
        else:
            print(' The car had started already')
    elif(userinput == 'stop'):
        if(car_status == 'start'):
            print(' Alright, the car stops now')
            car_status = 'stop'
        else:
            print(' The car had stopped already')
    elif(userinput == 'quit'):
        print(' Bye~ ')
        break
    else:
        print(" Sorry, I don't understand.")
