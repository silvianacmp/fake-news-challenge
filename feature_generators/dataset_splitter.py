DISCUSS = 'discuss'
UNRELATED = 'unrelated'
AGREE = 'agree'
DISAGREE = 'disagree'

classes = [DISCUSS, UNRELATED, AGREE, DISAGREE]


def get_splits(dataset, train_split=0.9):
    # generate separate datasets for each article type
    classwise_datasets = {DISCUSS: [],
                          UNRELATED: [],
                          AGREE: [],
                          DISAGREE: []}

    for s in dataset.stances:
        classwise_datasets[s['Stance']].append(s)

    # now separate in train and test splits
    train_dataset = []
    test_dataset = []
    for k, v in classwise_datasets.items():
        train_count = int(train_split * len(v))
        train_dataset += v[:train_count]
        test_dataset += v[train_count:]
        train_proportion = train_count / float(len(v))
        print('{}: total_count={} train_prop={:.5f}, test_prop={:.5f}'.format(k,
                                                                      len(v),
                                                                      train_proportion,
                                                                      1 - train_proportion))

    print('Total train: {} \nTotal test: {}'.format(len(train_dataset), len(test_dataset)))
    return dataset.articles, train_dataset, test_dataset

