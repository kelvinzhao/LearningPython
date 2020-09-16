# Guss Number Game
# Created by Kelvinzhao
# This is a simple game, player has 5 chances to guess a random integer number
# between 1 to 9. system will catch ValueError with 'try...except' statement.

import random
import pdb

secret_num = random.randrange(1, 10)
guess_count = 0
guess_limit = 5

pdb.set_trace()

while guess_count < guess_limit:
    try:
        user_num = input(f"guess a number from 1 to 9, "
                         f"you still have {guess_limit-guess_count}"
                         f"{' chance' if guess_count == guess_limit-1 else ' chances'} : ")
        user_num = int(user_num)
    except ValueError:
        print("invalid input, try again.")
    else:
        if user_num == secret_num:
            print('You Win, the secret number is ', secret_num)
            break
        else:
            print('Ops, not correct.')
            guess_count += 1
else:
    print('You Lose !')
