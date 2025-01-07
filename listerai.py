from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def read_data(file_path):
    items = []
    labels = []
    categories = set() 
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")  
            items.append(parts[0])  
            labels.append(parts[1:])  
            categories.update(parts[1:])  
    
    return items, labels, list(categories)  


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


def categorize_groups(groups, categories):
    categorized_groups = []
    for i, group in enumerate(groups):
        category = categories[i % len(categories)]  
        categorized_groups.append(f"{i+1}-{category}: {', '.join(group)}")
    return categorized_groups




def predict_labels(item, vectorizer, model, items, labels):
    
    if item in items:
        
        index = items.index(item)
        return ''.join(labels[index])  
    else:
        
        item_vec = vectorizer.transform([item])
        prediction = model.predict(item_vec)
        return ''.join(prediction[0].split())  


def read_feedback(file_path, item):
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")  
            if parts[0] == item:  
                return parts[-1].strip().lower()  
    return None  


def update_model(new_item, new_labels, vectorizer, model, items, labels):
    
    items.append(new_item)
    labels.append(new_labels)
    
    
    X = vectorizer.fit_transform(items)
    model.fit(X, [', '.join(label) for label in labels])
    print("Model updated!")


def save_new_data(file_path, new_item, new_labels):
    with open(file_path, "a") as file:
        file.write(f"{new_item},{','.join(new_labels)}\n")
    print(f"New data saved to {file_path}!")


def main():
    while True:
        
        items, labels, categories = read_data("listerai.txt")
        
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(items)
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
        
        
        model = MultinomialNB()
        model.fit(X_train, [', '.join(label) for label in y_train])  
        
        
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score([', '.join(label) for label in y_test], y_pred)}")
        
        
        words = input("\nEnter words separated by '-': ")
        num_groups = int(input("How many groups: "))
        
        
        print(f"\n>words: {words}")
        print(f">how many groups: {num_groups}")
        
        
        groups = split_into_groups(words, num_groups)
        
        
        categorized_groups = categorize_groups(groups, categories)
        
        
        print("\nResult:")
        for group in categorized_groups:
            print(group)
        
        
        new_item = input("\nEnter an item to predict its labels: ")
        predicted_labels = predict_labels(new_item, vectorizer, model, items, labels)
        print(f"The predicted labels for '{new_item}' are: {predicted_labels}")
        
        
        feedback = input("\nWas the prediction correct? (yes/no): ").strip().lower()
        if feedback == "no":
            new_labels = input("Enter the correct labels, separated by commas: ").strip().split(",")
            update_model(new_item, new_labels, vectorizer, model, items, labels)
            
            save_new_data("listerai.txt", new_item, new_labels)

if __name__ == "__main__":
    main()
