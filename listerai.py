from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Functie om de gegevens uit een tekstbestand te lezen
def read_data(file_path):
    items = []
    labels = []
    categories = set()  # Verzamel alle unieke categorieën
    
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")  # Split de lijn op de komma's
            items.append(parts[0])  # Het eerste onderdeel is het item (bijv. 'apple')
            labels.append(parts[1:])  # De rest zijn de labels (bijv. 'fruit', 'rood')
            categories.update(parts[1:])  # Voeg alle labels toe aan de set van categorieën
    
    return items, labels, list(categories)  # Geef ook de lijst van unieke categorieën terug

# Functie om de woorden te splitsen en in groepen te verdelen
def split_into_groups(words, num_groups):
    word_list = words.split("-")
    word_list.sort()
    group_size = len(word_list) // num_groups
    remainder = len(word_list) % num_groups
    groups = []
    start_index = 0
    
    for i in range(num_groups):
        end_index = start_index + group_size + (1 if i < remainder else 0)
        groups.append(word_list[start_index:end_index])
        start_index = end_index
    
    return groups

# Functie om groepen te categoriseren
def categorize_groups(groups, categories):
    categorized_groups = []
    for i, group in enumerate(groups):
        category = categories[i % len(categories)]  # Herhaalt de categorieën indien nodig
        categorized_groups.append(f"{i+1}-{category}: {', '.join(group)}")
    return categorized_groups

# Functie om voorspellingen te doen met het getrainde model
# Functie om voorspellingen te doen met het getrainde model
# Functie om voorspellingen te doen met het getrainde model
def predict_labels(item, vectorizer, model, items, labels):
    # Controleer of het item al in de dataset staat
    if item in items:
        # Zoek de bijbehorende labels op
        index = items.index(item)
        return ''.join(labels[index])  # Geef de labels als een string, gescheiden door een komma
    else:
        # Als het item niet in de dataset staat, doe een voorspelling met het model
        item_vec = vectorizer.transform([item])
        prediction = model.predict(item_vec)
        return ''.join(prediction[0].split())  # Zet de voorspelde labels samen als een string

# Functie om feedback uit het bestand te lezen
def read_feedback(file_path, item):
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")  # Split de lijn op de komma's
            if parts[0] == item:  # Als het item overeenkomt
                return parts[-1].strip().lower()  # Retourneer de feedback (ja/nee)
    return None  # Geen feedback gevonden voor dit item

# Functie om het model bij te werken op basis van feedback
def update_model(new_item, new_labels, vectorizer, model, items, labels):
    # Voeg het nieuwe item en labels toe aan de datasets
    items.append(new_item)
    labels.append(new_labels)
    
    # Hertrain het model met de bijgewerkte datasets
    X = vectorizer.fit_transform(items)
    model.fit(X, [', '.join(label) for label in labels])
    print("Model is bijgewerkt met nieuwe gegevens!")

# Functie om de nieuwe gegevens op te slaan in een bestand
def save_new_data(file_path, new_item, new_labels):
    with open(file_path, "a") as file:
        file.write(f"{new_item},{','.join(new_labels)}\n")
    print(f"Nieuwe gegevens zijn opgeslagen in {file_path}")

# Hoofdfunctie
def main():
    while True:
        # Stap 1: Gegevens inleiden uit een tekstbestand voor AI training
        items, labels, categories = read_data("listerai.txt")
        
        # Stap 2: Vectoriseer de teksten (gebruik alleen het item als tekst)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(items)
        
        # Stap 3: Splits de gegevens in trainings- en testsets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
        
        # Stap 4: Train een Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train, [', '.join(label) for label in y_train])  # Labels samenvoegen tot een string voor multi-label classificatie
        
        # Stap 5: Test het model en kijk naar de nauwkeurigheid
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score([', '.join(label) for label in y_test], y_pred)}")
        
        # Stap 6: Vraag om nieuwe invoer van de gebruiker voor de groepen
        words = input("\nEnter words separated by '-': ")
        num_groups = int(input("How many groups: "))
        
        # Print de invoer
        print(f"\n>words: {words}")
        print(f">how many groups: {num_groups}")
        
        # Stap 7: Verdeel de woorden in groepen
        groups = split_into_groups(words, num_groups)
        
        # Stap 8: Categoriseer de groepen met de dynamisch verzamelde categorieën
        categorized_groups = categorize_groups(groups, categories)
        
        # Print de gecategoriseerde groepen in het gewenste formaat
        print("\nResult:")
        for group in categorized_groups:
            print(group)
        
        # Stap 9: Vraag om een nieuw item en voorspel de labels
        new_item = input("\nEnter an item to predict its labels: ")
        predicted_labels = predict_labels(new_item, vectorizer, model, items, labels)
        print(f"The predicted labels for '{new_item}' are: {predicted_labels}")
        
        # Stap 10: Vraag feedback van de gebruiker en update het model indien nodig
        feedback = input("\nWas the prediction correct? (yes/no): ").strip().lower()
        if feedback == "no":
            new_labels = input("Enter the correct labels, separated by commas: ").strip().split(",")
            update_model(new_item, new_labels, vectorizer, model, items, labels)
            # Sla de nieuwe gegevens op in het bestand
            save_new_data("listerai.txt", new_item, new_labels)

if __name__ == "__main__":
    main()
