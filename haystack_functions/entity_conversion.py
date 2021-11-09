


import sys
from collections import defaultdict
from tqdm import tqdm



class EntityMatches:

    def __init__(self,txt_path):
        self.id_to_group = defaultdict(list)
        self.title_to_id = {}

        with open(txt_path) as fileobject:
            for line in tqdm(fileobject):
                split = line.split("\t",1)
                entity = split[0].strip()
                subj = split[1].strip()
                self.id_to_group[entity].append(subj)
                self.title_to_id[subj] = entity
        
    def get_synonyms(self, title):
        if title in self.title_to_id:
            id = self.title_to_id[title]
            return self.id_to_group[id]
        else: 
            return []
    




# usage enity_converstion.py <tab seperated txt file enitity \t string>
if __name__ == "__main__":

    txt_path = sys.argv[1]

    em = EntityMatches(txt_path)
    print(em.get_synonyms("Tricho-Dento-Osseous_Syndrome"))
