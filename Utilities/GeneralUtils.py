import numpy as np
import pandas as pd
import sys

def get_random_idx(array, size=40000, Repeat=False):
    rand_idx = np.random.choice(array[100:-100].index, size, replace=Repeat)
    return rand_idx

#combine the smoothed and the raw epochs
def expand_epochs(m):
    return pd.concat([m.Sxx_df.add_suffix('_sm'), m.multitaper_df], axis=1)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def query_option(question, valid_options=[1,2,3]):
    """Ask a question via raw_input() and return a numbered answer.

    "question" is a string that is presented to the user.
    "options" is a list with the presumed answer if the user just hits <Enter>.
    It must be "1" , "2", "3", etc (an answer is required of the user).
    """
    while True:
        sys.stdout.write(question)
        choice = int(input())
        if choice in valid_options:
            print(choice)
            return choice
            break
        else:
            sys.stdout.write("Valid options include one of the following: {}.\n".format(valid_options))