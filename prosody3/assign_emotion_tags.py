# assign_emotion_tags.py
def assign_emotion_tags(classifier, weight_learner, vader_scores, prosody_features, 
                       emotion_names=['joy', 'sadness', 'anger', 'neutral', 'surprise', 'fear']):
    vader, prosody = vader_scores.to('cuda' if torch.cuda.is_available() else 'cpu'), \
                    prosody_features.to('cuda' if torch.cuda.is_available() else 'cpu')
    w = weight_learner(vader, prosody)
    emotion_vector = w * vader + (1 - w) * prosody
    with torch.no_grad():
        probs = classifier(emotion_vector.unsqueeze(0)).cpu()
    primary_idx, secondary_idx = torch.topk(probs, k=2).indices
    return emotion_names[primary_idx], emotion_names[secondary_idx], probs