config = {
    'domain': 'restaurant'
}
bert_mapper = {
    'laptop': 'bert-base-uncased',
    'restaurant': 'bert-base-uncased'
}
path_mapper = {
    'laptop': './datasets/laptop',
    'restaurant': './datasets/restaurant'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant': ['food', 'place', 'service']
}
aspect_seed_mapper = {
    'laptop': {
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'os': {"os", "windows", "ios", "mac", "system", "linux"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse': {"mouse", "touch", "track", "button", "pad"},
        'software': {"software", "programs", "applications", "itunes", "photo"},
        'keyboard': {"keyboard", "key", "space", "type", "keys"}
    },
    'restaurant': {
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "drinks", "flavourful"},
        'place': {"ambience", "atmosphere", "seating", "surroundings", "environment", "location", "decoration",
                  "spacious", "comfortable", "place"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"}
    }
}
sentiment_category_mapper = {
    'laptop': ['negative', 'positive'],
    'restaurant': ['negative', 'positive']
}
sentiment_seed_mapper = {
    'laptop': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "slow", "broken", "complaint", "malware",
                     "virus", "junk", "crap", "cramped", "cramp"}
    },
    'restaurant': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "fresh", "warm", "friendly", "delicious", "fast",
                     "quick", "clean"},
        'negative': {"bad", "terrible", "horrible", "tasteless", "awful", "smelled", "unorganized", "gross",
                     "disappointment", "spoiled", "vomit", "cold", "slow", "dirty", "rotten", "ugly"}
    }
}
M = { # Desired vocab size
    'laptop': 150,
    'restaurant': 100
}
K_1 = 10
K_2 = 30
lambda_threshold = 0.5
# In the BERT paper, they had success with a batch size of 16 and 32 when
# fine-tuning.
batch_size = 32
validation_data_size = 100
# In the bert paper, they used Adam and found the following range of possible
# values worked well across all tasks: 5e-5, 3e-5, 2e-5
learning_rate = 1e-5
# In the BERT paper they had success fine-tuning with 2, 3, and 4 epochs.
epochs = 20
