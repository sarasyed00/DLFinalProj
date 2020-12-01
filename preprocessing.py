import json
from nltk.tokenize import sent_tokenize, word_tokenize

def get_data(file_path):
    reviews = []
    ratings = []
    items = []
    print("Started Reading JSON file which contains multiple JSON items")
    with open(file_path, encoding="utf8") as f:
        counter = 0
        for jsonObj in f:
            counter += 1
            item = json.loads(jsonObj)
            items.append(item)

            if counter is 100:
                break

    for item in items:
        text = item["text"]
        # print(text)
        reviews.append(text)
        stars = item["stars"]
        # print(stars)
        ratings.append(stars)


    return reviews, ratings

def get_tokenized_reviews(reviews, byWord=False):
    reviews_tokenized = []
    
    for review in reviews:    
        #Tokenize by sentence or word based on bool
        if byWord:
            reviews_tokenized.append(word_tokenize(review))
        else:
            reviews_tokenized.append(sent_tokenize(review))

    return reviews_tokenized

#def main():
#    reviews, ratings = get_data('./yelp_dataset/review.json')

    # To get sent_tokenize to work, In python terminal: 
    # >>> import nltk
    # >>> nltk.download('punkt')

#    reviews_tokenized = tokenize_reviews(reviews)

    #reviews, ratings = get_data('yelp_dataset/review.json')

#if __name__ == '__main__':
#    main()