from transformers import AutoTokenizer, TFBertForMaskedLM
from config import *
from tqdm import tqdm
import tensorflow as tf
from filter_words import filter_words

class ScoreComputer:
    '''
    Computes unnormalised overlap scores for each aspect category and sentiment polarity and saves in "scores.txt" file
    '''
    def __init__(self, aspect_vocabularies, sentiment_vocabularies):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.mlm_model = TFBertForMaskedLM.from_pretrained(self.bert_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.aspect_vocabularies = aspect_vocabularies # This is the expanded vocab list we generated in vocab_generator
        self.sentiment_vocabularies = sentiment_vocabularies # Same here
    
    def __call__(self, sentences, aspects, opinions):
        categories = aspect_category_mapper[self.domain] # List: ['food', 'place', 'service']
        polarities = sentiment_category_mapper[self.domain] # ['negative', 'positive']
        K = K_2 # 30

        aspect_sets = self.load_vocabulary(self.aspect_vocabularies, M[self.domain]) # this is a dict. key is each category (food, place, service). Value is a set of all 'similar' words that we generated as our expanded vocabulary
        polarity_sets = self.load_vocabulary(self.sentiment_vocabularies, M[self.domain])

        f = open(f'{self.root_path}/scores.txt', 'w')
        
        for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)): # Loop over each sentence. Aspect is all the nouns in that sentence. Opinion is all the adjectives and adberbs. Remember the aspects and opinions are literally all nouns and adjectives. No filter for just food stuff.
            aspect_words = set()
            opinion_words = set()
            if aspect != '##':
                aspect_words = set(aspect.split()) # Take long string of nouns and split on spaces. Turn into a set.
            if opinion != '##':
                opinion_words = set(opinion.split())
            ids = self.tokenizer(sentence, return_tensors='tf', truncation=True)['input_ids'] # Tokenize sentence into IDs
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0]) # Convert ids back into words. Note that our tokens are actually a little smaller than words. And if we don't have a word in the vocab, then I think it is replaced with ## or #
            word_predictions = self.mlm_model(ids)[0] # Pass tokens into model. Get likelihood of replacement for each one. E.g. tensor is size: # words in sentence x # words in BERT vocab
            word_predictions = tf.cast(word_predictions, tf.float16) # Cast as float16 to avoid bug in TensorFlow-Metal - https://developer.apple.com/forums/thread/689299
            word_scores, word_ids = tf.math.top_k(input=word_predictions, k=K) # Get top 30 predictions for each word
            word_ids = tf.squeeze(word_ids) # get rid of unused batch dimension
            
            cat_scores = {} # One of these dicts per sentence. Basically we are trying to determine, 'is this sentence primarily about 'place', 'food', or 'service'?
            pol_scores = {} # And we determine that by looking at each noun and adjective in the sentence, and seeing if the replacement words are in our vocab.

            cntAspects = 0 # Will be the number of nouns in the sentence
            cntOpinions = 0 # Will be the number of adjectives and adverbs in the sentence

            for idx, token in enumerate(tokens): # Loop over each word in the sentence
                if token in aspect_words: # Is the word one of our aspect words? E.g. is the word a noun? Remember aspect words are just all nouns right now
                    cntAspects += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx]) # Get the top 30 replacements for this noun.
                    for repl in replacements: # Loop over each replacement
                        if repl in filter_words or '##' in repl: # If the replacement is a filter word, ignore it.
                            continue
                        for cat in categories: # For category in ['food', 'place', 'service']
                            if repl in aspect_sets[cat]: # Is the replacement in our generated vocab for this category? Is this the top M words? Or all of them? When do we filter just the top M? I think the authors forget to filter by the top M words.
                                cat_scores[cat] = cat_scores.get(cat, 0) + 1 # Increment the 'score' for this category by 1
                                break
                if token in opinion_words:
                    cntOpinions += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for pol in polarities:
                            if repl in polarity_sets[pol]:
                                pol_scores[pol] = pol_scores.get(pol, 0) + 1
                                break
            summary = f'{sentence}\n'
            for cat in categories: # ['food', 'place', 'service']
                val = cat_scores.get(cat, 0) / max(cntAspects, 1) # Of all the nouns in the sentence, we get ~30 replacement words. If one of the replacements is in our vocab, then increment the score for that category. Here we get the score for this category for this sentence, then divide by number of nouns. So essentially this is 'for each noun in the sentence, on average, how many of the replacement words are in our vocab?'
                summary = summary + f' {cat}: {val}' # Basically calculate this 'normalized' score for each category.
            
            for pol in polarities:
                val = pol_scores.get(pol, 0) / max(cntOpinions, 1)
                summary = summary + f' {pol}: {val}'
            # So in this sentence, most of the adjectives are positive. And a lot of the nouns are related to food and place, but not service. Place narrowly beats out food.
            f.write(summary) # Example summary string: this place is not pretentious and down to earth . very simple and affordable mexican # # # fare . it 's comparable to los favs . carne # # # asada fries are my favorite . great mexican place within walking # # # distance from asu . \n food: 3.888888888888889 place: 5.111111111111111 service: 0.7777777777777778 negative: 0.0 positive: 3.3125
            f.write('\n')
        f.close()

    def load_vocabulary(self, source, limit):
        target = {}
        for key in source:
            words = []
            for freq, word in source[key][:limit]:
                words.append(word)
            target[key] = set(words)
        return target
