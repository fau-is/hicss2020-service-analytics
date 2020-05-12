import csv
import gensim
import numpy as np

if __name__ == '__main__':

    csvfile = open("../data/data.csv", 'r')
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    next(spamreader, None)
    data_set = list()
    documents_context_str = list()
    documents_context_emb = list()
    embedding_size = 32

    # learn embedding model
    index = 0
    for row in spamreader:
        data_set.append(row)
        taggedDocument = gensim.models.doc2vec.TaggedDocument(
            gensim.utils.to_unicode(str.encode(' '.join(row[3:len(row)]))).split(), [index])

        index = index + 1
        documents_context_str.append(taggedDocument)

    # train model
    model = gensim.models.Doc2Vec(documents_context_str, dm=0, vector_size=embedding_size, window=5, min_count=1,
                                  alpha=0.025, min_alpha=0.025, worker=12)
    nrEpochs = 1
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print('Now training epoch %s' % epoch)
        model.train(documents_context_str, total_examples=len(documents_context_str), epochs=nrEpochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    # save and get model
    model.save('checkpoints/' + str(0) + '_context_attributes_doc2vec_2d' + str(embedding_size) + '.model',
               sep_limit=2000000000)
    model = gensim.models.Doc2Vec.load(
        'checkpoints/' + str(0) + '_context_attributes_doc2vec_2d' + str(embedding_size) + '.model')

    # apply embedding model and save data set
    for document_context_str in documents_context_str:
        try:
            documents_context_emb.append(model.infer_vector(document_context_str.words))
        except:
            documents_context_emb.append([0] * embedding_size)
            print(document_context_str.words, 'not found')

    # concate
    data_set_new = np.zeros((len(data_set), 3 + embedding_size), dtype=np.dtype('U20'))

    # fill data
    for index in range(0, len(data_set)):
        # process
        for sub_index_process in range(0, 3):
            data_set_new[index, sub_index_process] = data_set[index][sub_index_process]
        # context
        for sub_index_context in range(0, embedding_size):
            data_set_new[index, sub_index_context + 3] = documents_context_emb[index][sub_index_context]

    # write dataset
    with open("Data.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["case", "event", "time"])

        for row in data_set_new:
            try:
                spamwriter.writerow(['{:f}'.format(cell) for cell in (row)])
            except:
                spamwriter.writerow(['{:s}'.format(cell) for cell in (row)])
