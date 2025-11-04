import re
import emoji
import nltk
from nltk.corpus import stopwords

class TextProcessor:
    def __init__(self):
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def remove_emojis(self, text):
        """Remove emojis from text"""
        return emoji.replace_emoji(text, replace='')
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]+', '', text)
        return text
    
    def remove_mentions_hashtags(self, text):
        """Remove @mentions and #hashtags"""
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        return text
    
    def remove_special_chars(self, text):
        """Remove special characters but keep basic punctuation"""
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        return text
    
    def remove_extra_spaces(self, text):
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def preprocess(self, text, remove_stopwords=False):
        """Complete preprocessing pipeline"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove emojis
        text = self.remove_emojis(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove mentions and hashtags
        text = self.remove_mentions_hashtags(text)
        
        # Remove special characters
        text = self.remove_special_chars(text)
        
        # Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        # Optionally remove stopwords
        if remove_stopwords and self.stop_words:
            words = text.split()
            text = ' '.join([word for word in words if word not in self.stop_words])
        
        return text.strip()
