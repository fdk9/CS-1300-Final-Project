from reziew_algorithm import ReziewAlgorithm

model = ReziewAlgorithm('reziew.csv', 'review_text', 'grade')
model.train_model()

model.reviews_ratings.head()

print(model.max_c)

model.test_model()
model.get_words_and_scores()
model.get_most_positive_negative_words()

long_review = ''
revs = model.all_reviews
for i in range(0, len(revs)):
    if len(long_review) < 999000:
        try:
            long_review += (revs[i] + '. ')
        except:
            pass
#rev = reviews['reviews'][4]
# print(long_review)
result = model.analyze(long_review)
print(result)

json_list = []
for k, v in result.items():
    ct = model.get_review_count(k)
    if (ct > 0):
        new_dict = {}
        new_dict['noun'] = k
        new_dict['noun_count'] = ct
        new_dict['adjs'] = [tup[0] for tup in v]
        new_dict['adjs_score'] = [tup[1] for tup in v]
        new_dict['adjs_count'] = [tup[2] for tup in v]
        new_dict['noun_score'] = (np.sum(np.multiply(
            new_dict['adjs_score'], new_dict['adjs_count'])))/np.sum(new_dict['adjs_count'])
        json_list.append(new_dict)


def get_noun_count(json):
    try:
        return int(json['noun_count'])
    except KeyError:
        return 0


def get_noun_score(json):
    try:
        return int(json['noun_score'])
    except KeyError:
        return 0


json_list.sort(key=get_noun_score, reverse=True)
json_output = json.dumps(json_list)
print(json_output)

print(model.word_score_dict)

# LEFTOVER CODE FOR GETTING REVIEW IN WHICH A WORD APPEARS
# for word in word_dict['positive']['words']:
#     hit_list = [idx for idx, s in enumerate(reviews_clean) if word in s]
#     first_hit = hit_list[0]
#     word_dict['positive']['reviews'].append(review_list[first_hit])
#     word_dict['positive']['hits'].append(len(hit_list))
#     print(review_list[first_hit])
#     print(len(hit_list))
#     print("\n")

# for word in word_dict['negative']['words']:
#     hit_list = [idx for idx, s in enumerate(reviews_clean) if word in s]
#     first_hit = hit_list[0]
#     word_dict['negative']['reviews'].append(review_list[first_hit])
#     word_dict['negative']['hits'].append(len(hit_list))
#     print(review_list[first_hit])
#     print(len(hit_list))
#     print("\n")
