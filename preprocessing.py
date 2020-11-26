import json

def get_data(file_path):
    reviews = []
    ratings = []
    items = []
    print("Started Reading JSON file which contains multiple JSON items")
    with open(file_path) as f:
        for jsonObj in f:
            item = json.loads(jsonObj)
            items.append(item)
    for item in items:
        text = item["text"]
        # print(text)
        reviews.append(text)
        stars = item["stars"]
        # print(stars)
        ratings.append(stars)

    return reviews, ratings


def main():
    get_data('yelp_dataset/review.json')
    return

if __name__ == '__main__':
    main()
