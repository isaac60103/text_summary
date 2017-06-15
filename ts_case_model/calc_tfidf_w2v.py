import statics


w2vfile = '/home/ubuntu/workspace/text_summary_data/w2v.pickle'
tfidf_path = '/home/ubuntu/workspace/text_summary_data/tfidf_score/tfidfscore.pickle'
save_path = '/home/ubuntu/workspace/text_summary_data/tfidf_score/w2v_tfidf.pickle'

w2v = statics.loadfrompickle(w2vfile)
tiidf = statics.loadfrompickle(tfidf_path)

tfidf_w2v = {}

for w in w2v:
    tfidf_w2v[w] = w2v[w]*tiidf[1][w][2]
    
statics.savetopickle(save_path, tfidf_w2v)