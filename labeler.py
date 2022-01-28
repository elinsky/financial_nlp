from config import *
import numpy as np

class Labeler:

    def __init__(self):
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
    
    def __call__(self):
        categories = aspect_category_mapper[self.domain] # ['food', 'place', 'service']
        polarities = sentiment_category_mapper[self.domain] # ['negative', 'positive']

        # Distributions
        dist = {} # This will become a dict. Keys are categories (food, place, service, positive, negative). Values are a list. List will be same length as the whole dataset. So every sentence is included, even if it doesn't have any of the category words. And the values will be the scores for that sentence.
        for cat in categories:
            dist[cat] = []
        for pol in polarities:
            dist[pol] = []

        # Read scores
        with open(f'{self.root_path}/scores.txt', 'r') as f:
            for idx, line in enumerate(f): # Loop over each sentence in the training set. This is the scores file. So line 1 is a sentence. Line 2 are the scores for each category
                if idx % 2 == 1: # If line is the scores (not the sentence)
                    values = line.strip().split() # values = ['food:', '3.888888888888889', 'place:', '5.111111111111111', 'service:', '0.7777777777777778', 'negative:', '0.0', 'positive:', '3.3125']
                    for j, val in enumerate(values):
                        if j % 2 == 1:
                            dist[values[j-1][:-1]].append(float(val)) # Save the sentence scores to the distribution dict
        
        # Compute mean and sigma for each category
        means = {} # These are the average scores of each sentence: {'food': 3.688839230487442, 'place': 3.0244711181340325, 'service': 1.891517212210749, 'negative': 0.6808272938144453, 'positive': 7.53198284481678}
        sigma = {}
        for key in dist: # Keys are food, place, service, positive, negative
            means[key] = np.mean(dist[key]) # Calculate the average score for each category
            sigma[key] = np.std(dist[key]) # Calculate the standard deviation for each category
        
        nf = open(f'{self.root_path}/label.txt', 'w')
        cnt = {}
        with open(f'{self.root_path}/scores.txt', 'r') as f:
            sentence = None
            for idx, line in enumerate(f): # Loop over all the lines in the scores file.
                if idx % 2 == 1: # Only look at lines that are scores, not the actual sentences
                    aspect = [] # This is a list of aspects (food, place, service) that have z-scores above our lambda threshold (0.5). The list applies to just our sentence.
                    sentiment = []
                    key = None
                    for j, val in enumerate(line.strip().split()):
                        if j % 2 == 1: # val example: '3.888888888888889'. These are the scores for the sentence.
                            # Normalise score
                            dev = (float(val) - means[key]) / sigma[key] # It looks like we calculate a z-score here
                            if dev >= lambda_threshold: # Only keep ones that are 0.5 stdev above average
                                if key in categories:
                                    aspect.append(key)
                                else:
                                    sentiment.append(key)
                        else:
                            key = val[:-1]
                    # No conflict (avoid multi-class sentences)
                    if len(aspect) == 1 and len(sentiment) == 1: # Only add to label.txt the sentences that clearly have 1 dominant aspect category, and 1 dominant opinion category.
                        nf.write(sentence)
                        nf.write(f'{aspect[0]} {sentiment[0]}\n')
                        keyword = f'{aspect[0]}-{sentiment[0]}'
                        cnt[keyword] = cnt.get(keyword, 0) + 1
                else:
                    sentence = line
        nf.close()
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)