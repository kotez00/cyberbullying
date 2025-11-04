import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from text_processor import TextProcessor

# Sample training data for cyberbullying detection
# 1 = cyberbullying/hate speech, 0 = normal
TRAINING_DATA = [
    # Cyberbullying examples
    ("you are so stupid and worthless", 1),
    ("nobody likes you, go away", 1),
    ("you're such a loser and ugly", 1),
    ("kill yourself", 1),
    ("you should die", 1),
    ("everyone hates you", 1),
    ("you're pathetic and useless", 1),
    ("what a freak", 1),
    ("you're disgusting", 1),
    ("get lost loser", 1),
    ("you're a complete idiot", 1),
    ("no one wants you here", 1),
    ("you're trash", 1),
    ("shut up stupid", 1),
    ("you're the worst person ever", 1),
    ("i hate you so much", 1),
    ("you're so dumb", 1),
    ("what a waste of space", 1),
    ("go die in a hole", 1),
    ("you're absolutely worthless", 1),
    ("nobody cares about you", 1),
    ("you're such a failure", 1),
    ("you make me sick", 1),
    ("kill yourself nobody wants you", 1),
    ("you're a joke", 1),
    ("pathetic loser", 1),
    ("you're so ugly inside and out", 1),
    ("die already", 1),
    ("you don't deserve to live", 1),
    ("everyone thinks you're stupid", 1),
    
    # Normal/positive examples
    ("have a great day", 0),
    ("thank you so much for your help", 0),
    ("that's really interesting", 0),
    ("i appreciate your perspective", 0),
    ("great job on the project", 0),
    ("nice work", 0),
    ("that's a good point", 0),
    ("i agree with you", 0),
    ("thanks for sharing", 0),
    ("how are you doing today", 0),
    ("what do you think about this", 0),
    ("i'm excited about the weekend", 0),
    ("let's meet up later", 0),
    ("that sounds fun", 0),
    ("i love this song", 0),
    ("beautiful day today", 0),
    ("congratulations on your achievement", 0),
    ("you did well", 0),
    ("keep up the good work", 0),
    ("i'm looking forward to it", 0),
    ("that's awesome", 0),
    ("good morning everyone", 0),
    ("hope you feel better soon", 0),
    ("thanks for being a good friend", 0),
    ("i appreciate you", 0),
    ("you're amazing", 0),
    ("well done", 0),
    ("that's helpful", 0),
    ("i understand", 0),
    ("have a wonderful evening", 0),
]

def train_models():
    """Train and save the cyberbullying detection models"""
    print("Starting model training...")
    
    # Initialize text processor
    processor = TextProcessor()
    
    # Prepare data
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    
    # Preprocess texts
    processed_texts = [processor.preprocess(text) for text in texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_texts)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes
    print("Training Naive Bayes classifier...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_accuracy = nb_model.score(X_test, y_test)
    print(f"Naive Bayes accuracy: {nb_accuracy:.2f}")
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"Random Forest accuracy: {rf_accuracy:.2f}")
    
    # Save models
    print("Saving models...")
    with open('nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Models trained and saved successfully!")
    return nb_model, rf_model, vectorizer

if __name__ == "__main__":
    train_models()
