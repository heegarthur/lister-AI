# lister-AI
an AI for putting words in categories

I made listerai.txt for data learning, and the structure has to be like this if you want to add something:

word,category,category,category

you can add as much categories as you want

listerat.txt is small not that big so NOW the accuracy is 0.01266624445851805 that not good so don't be suprised if the AI is bad

you can type a few words and say how many categories you want, you can also find the categories of 1 specefic word

a basic input can be like this:


```
Accuracy: 0.01266624445851805

Enter words separated by '-': apple,monkey,pineapple,elephant
How many groups: 2

>words: apple,monkey,pineapple,elephant
>how many groups: 2

Result:
1-animals: apple,monkey
2-food: pineapple,elephant

Enter an item to predict its labels: dark
The predicted labels for 'dark' are: math,logic,concept

Was the prediction correct? (yes/no): no
Enter the correct labels, separated by commas: black,sight,light
Model updated!
New data saved to listerai.txt
Accuracy: 0.013291139240506329
```

but you didn't do anything wrong if he says something super dumb, it's because it's not trained that much
