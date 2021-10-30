


import sys
from collections import defaultdict
from tqdm import tqdm


# maybe multi proceess this?

# usage enity_converstion.py <tab seperated txt file enitity \t string>
if __name__ == "__main__":

    txt_path = sys.argv[1]

    d = defaultdict(list)

    with open(txt_path) as fileobject:
        for line in tqdm(fileobject):
            split = line.split("\t",1)
            entity = split[0]
            subj = split[1]
            d[entity].append(subj)

    for k in d.keys():
        if len(d[k]) > 1:
            print(d[k])
