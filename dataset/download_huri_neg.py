from tdc.multi_pred import PPI

def download_huri_neg():
    data = PPI(name = 'HuRI')
    data_neg = data.neg_sample(frac = 1)

    data_neg = data_neg.get_split()
    
    train_data = data_neg['train']
    val_data = data_neg['valid']
    test_data = data_neg['test']

    train_data.to_csv('../data/huri_neg_train.csv', index=False)
    val_data.to_csv('../data/huri_neg_val.csv', index=False)
    test_data.to_csv('../data/huri_neg_test.csv', index=False)

if __name__ == "__main__":
    download_huri_neg()