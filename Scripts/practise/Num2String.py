# change number into string using dictionary

digits_mapping = {
        '1': 'One',
        '2': 'Two',
        '3': 'Three',
        '4': 'Four',
        '5': 'Five',
        '6': 'Six',
        '7': 'Seven',
        '8': 'Eight',
        '9': 'Nine',
        '0': 'Zero'
        }
phone = input("give your phone number: ")
str1 = ''
print(str1.join([f"{digits_mapping[x]} " for x in phone]))
