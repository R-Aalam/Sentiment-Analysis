import string
import re
import contractions
import emoji
import pickle

class Preprocessing:
    def __init__(self,max_length=64):
        self.max_length = max_length
        self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
              "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
              "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
              "having", "do", "does", "did", "doing", "a", "an", "the", "and", "if", "or", "because", "as", "until", "while", "of", "at",
              "by", "for", "with", "about", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
              "up", "down", "in", "on", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
              "how", "all", "any", "both", "each", "other", "some", "such", "own", "same", 
              "than", "too", "now"]
        
        self.punctuation_chars = string.punctuation
        self.regrex_pattern = f"[{re.escape(self.punctuation_chars)}]"
        self.url_pattern = r"https?://\S+|www\.\S+"

    def clean_text(self,text):
        de_emoji = emoji.demojize(text,delimiters=(" ", " "))
        url_remove = re.sub(self.url_pattern,"",de_emoji)  #remove URLS
        expanded = contractions.fix(url_remove).lower()
        punctuation_remove = re.sub(self.regrex_pattern,"",expanded)
        cleaned_token = [token for token in punctuation_remove.split(" ") if token not in self.stop_words]
        
        return " ".join(cleaned_token)
        

    def padding_text(self,text):
        if ((self.max_length - len(text)) < 0):
            return text[:self.max_length-len(text)]
        padded_sequence = text + [0] * (self.max_length - len(text))

        return padded_sequence


class Tokenization:
    def __init__(self,data):
        self.data = data
        all_words = " ".join(self.data).lower().split()

        self.vocab = set(all_words)
        pad_unknown = {"<PAD>":0,"<UNK>":1}
        self.word2idx = {word: i+2 for i, word in enumerate(self.vocab)}  #  0 for padding and 1 for Unknown
        self.word2idx.update(pad_unknown)

    def encode(self,text):
        encoded = []
        for word in text.lower().split():
            if word in self.vocab:
                encoded.append(self.word2idx[word])
            else:
                encoded.append(1)
        return encoded

    def decode(self,encode_text):
        decode_list = []
        for token in encode_text:
            for key,value in self.word2idx.items():
                if token==value:
                    decode_list.append(key)
        return " ".join(decode_list)
                    

