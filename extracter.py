from tqdm import tqdm
from config import *
import spacy


class Extracter:
    '''
    Extract potential-aspects and potential-opinion words
    '''

    def __init__(self):
        self.smodel = spacy.load('en_core_web_sm')
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]

    def __call__(self):
        # Extract potential-aspects and potential-opinions
        sentences = []
        aspects = []
        opinions = []

        with open(f'{self.root_path}/train.txt') as f:
            for line in tqdm(f):  # Loop over every sentence in the training set.
                text = line.strip()
                sentences.append(text)  # Add sentence to sentence list
                words = self.smodel(text)  # Pass sentence into spacy model
                o = []
                a = []
                for word in words:  # For each word in the sentence
                    if word.tag_.startswith('JJ') or word.tag_.startswith(
                            'RR'):  # I think the adverb (RR*), might be broken
                        # Adjective or Adverb. If the word is an adjective or adverb, then add it to our opinions list.
                        o.append(word.text)
                    if word.tag_.startswith('NN'):
                        # Noun. If the word is a noun, add it to our aspect list. Could be any noun at this point.
                        a.append(word.text)
                opinions.append(' '.join(o) if len(o) > 0 else '##')
                aspects.append(' '.join(a) if len(a) > 0 else '##')

        return sentences, aspects, opinions
        # Aspects and opinions are both lists.
        # Each list has the same length as the number of sentences
        # e.g. if the training set has 1000 sentences, then aspects and opinion
        # both have a length of 1000. For aspects, the entry is a string of all
        # the nouns in the sentence, separated by spaces. E.g. 'place earth mexican fare carne asada fries place'
        # Same true for opinions, except those are all adverbs or adjectives.
        # Note that this list not filtered by anything category related.
