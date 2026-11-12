#Importing libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

#Intiating sentiment analyzer
sentiment = SentimentIntensityAnalyzer()

#Getting input and setting 2 examples
text_1 = input("Enter a sentence you want to know the sentiment of: ")
text_2 = "It was very nice to meet you."
text_3 = "The refereeing today was awful"

#Determining the sentiment
sent1 = sentiment.polarity_scores(text_1)
sent2 = sentiment.polarity_scores(text_2)
sent3 = sentiment.polarity_scores(text_3)

#Determining the Polarity 
p_1 = TextBlob(text_1).sentiment.polarity
p_2 = TextBlob(text_2).sentiment.polarity
p_3 = TextBlob(text_3).sentiment.polarity

#Determining the Subjectivity
s_1 = TextBlob(text_1).sentiment.subjectivity
s_2 = TextBlob(text_2).sentiment.subjectivity
s_3 = TextBlob(text_3).sentiment.subjectivity

#Printing the results
print("\n")
print("Text 1:", text_1)
print("Sentiment of text 1:", sent1)
print("Second opinion (polarity):", p_1)
print("Subjectivity of Text 1 is", s_1)
print("\n")

print("Text 2:",text_2)
print("Sentiment of text 2:", sent2)
print("Second opinion (polarity):", p_2)
print("Subjectivity of Text 1 is", s_2)
print("\n")

print("Text 3:", text_3)
print("Sentiment of text 1:", sent3)
print("Second opinion (polarity):", p_3)
print("Subjectivity of Text 1 is", s_3)
print("\n")