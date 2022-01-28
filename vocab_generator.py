from transformers import AutoTokenizer, TFBertForMaskedLM
from config import *
from filter_words import filter_words
import tensorflow as tf
from tqdm import tqdm

class VocabGenerator:

    def __init__(self, save_results=True):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.mlm_model = TFBertForMaskedLM.from_pretrained(self.bert_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.save_results = save_results
    
    def __call__(self):
        aspect_categories = aspect_category_mapper[self.domain]
        aspect_seeds = aspect_seed_mapper[self.domain]
        aspect_vocabularies = self.generate_vocabularies(aspect_categories, aspect_seeds)

        sentiment_categories = sentiment_category_mapper[self.domain]
        sentiment_seeds = sentiment_seed_mapper[self.domain]
        sentiment_vocabularies = self.generate_vocabularies(sentiment_categories, sentiment_seeds)

        return aspect_vocabularies, sentiment_vocabularies

    def generate_vocabularies(self, categories, seeds):
        # Initialise empty frequency table
        freq_table = {}
        for cat in categories:
            freq_table[cat] = {}
        
        # Populate vocabulary frequencies for each category
        for category in categories: # Loop over each category (food, place, service)
            print(f'Generating vocabulary for {category} category...')
            with open(f'{self.root_path}/train.txt') as f:
                for line in tqdm(f): # Loop over each review
                    text = line.strip()
                    # Wouldn't this be better as something like: if seeds[category] in text:
                    for seed in seeds[category]:
                        if seed in text:
                            # Take the sentence and tokenize it. Result is an integer for each word.
                            ids = self.tokenizer(text, return_tensors='tf', truncation=True)['input_ids']
                            # Convert the IDs (integer for each word) into words again
                            tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
                            # Pass the whole sentence (as IDs/integers) into the MLM
                            # Model outputs a tensor of sentence length x BERT vocab size
                            # For each word in the sentence, we output how likely each
                            # word in the vocab is likely to be a replacement.
                            word_predictions = self.mlm_model(ids)[0]
                            word_predictions = tf.cast(word_predictions, tf.float16)  # Cast as float16 to avoid bug in TensorFlow-Metal - https://developer.apple.com/forums/thread/689299
                            # Find the top K predictions for each word.
                            word_scores, word_ids = tf.math.top_k(input=word_predictions, k=K_1)
                            # Remove unnecessary batch dimension?
                            word_ids = tf.squeeze(word_ids)
                            # Loop over each word in the sentence
                            for idx, token in enumerate(tokens):
                                # Is the word in our sentence one of the seed words for our category?
                                # e.g. category service. seed words are: tips, manager, waitress...
                                if token in seeds[category]: # This probably isn't optimal either. I might need to re-write this whole function.
                                    # If so, then take the top K predictions for that word, and add them to our freq table.
                                    self.update_table(freq_table, category, self.tokenizer.convert_ids_to_tokens(word_ids[idx]))
        
        # Remove words appearing in multiple vocabularies (generate disjoint sets)
        for category in categories:
            for key in freq_table[category]:
                for cat in categories:
                    if freq_table[cat].get(key) != None and freq_table[cat][key] < freq_table[category][key]:
                        del freq_table[cat][key]
        
        vocabularies = {}

        for category in categories:
            words = []
            for key in freq_table[category]:
                words.append((freq_table[category][key], key))
            words.sort(reverse=True)
            vocabularies[category] = words

            if self.save_results:
                # Saving vocabularies. I think there is a bug here! I think we should only be writing the top M words. I think we write all of them.
                f = open(f'{self.root_path}/dict_{category}.txt', 'w')
                for freq, word in words:
                    f.write(f'{word} {freq}\n')
                f.close()

        return vocabularies
    
    def update_table(self, freq_table, cat, tokens):
        for token in tokens:
            if token in filter_words or '##' in token:
                continue
            freq_table[cat][token] = freq_table[cat].get(token, 0) + 1



    
