from tdc.multi_pred import PPI

def download_huri():
    data = PPI(name = 'HuRI')
    split = data.get_split()
    data_neg = data.neg_sample(frac = 1)

    # for key in split:
    #     print(key)

    train_data = split['train']
    val_data = split['valid']
    test_data = split['test']

    train_data.to_csv('../data/huri_train.csv', index=False)
    val_data.to_csv('../data/huri_val.csv', index=False)
    test_data.to_csv('../data/huri_test.csv', index=False)

if __name__ == "__main__":
    download_huri()
