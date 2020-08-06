# practise for OOP

class Pet:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def intro(self):
        print(f'Hello, my name is {self.name} and I am {self.age} years old')

    def speak(self):
        print("I don't know what to say")


class Cat(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def speak(self):
        print("Meow")

    def intro(self):
        print(f'Hello, my name is {self.name} and I am {self.age} years old, I am {self.color}')


class Dog(Pet):
    def speak(self):
        print("Woof~~")


p = Pet("Tim", 12)
c = Cat("Bill", 5, 'red')
d = Dog("Jill", 9)

p.intro()
c.intro()
d.intro()

p.speak()
c.speak()
d.speak()
