__all__ = ['Person', 'Staff']


class Person:
    def __init__(self, name):
        self.name = name

    def talk(self, message):
        print(f"{self.name} says that {message}")


class Staff(Person):
    def getup(self, clock):
        print(f"I get up at {clock}")

    def intro(self):
        print(f"Hi, my name is {self.name}")


if __name__ == '__main__':
    person = Person(name='John')
    person.talk('Hello world')

    bob = Person('Bob Dylon')
    bob.talk('Hello')

    li = Staff(name='Li')
    li.talk('hi, I am li')
    li.intro()
    li.getup("six o'clock")
