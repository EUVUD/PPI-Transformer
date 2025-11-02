import pandas as pd

def generate_unique_protein_list():

    test_df = pd.read_csv('../data/huri_neg_test.csv')
    val_df = pd.read_csv('../data/huri_neg_val.csv')
    train_df = pd.read_csv('../data/huri_neg_train.csv')

    all_proteins = pd.concat([test_df, val_df, train_df], ignore_index=True)

    # Select and rename Protein1 columns
    protein1_df = all_proteins[['Protein1_ID', 'Protein1']].rename(
        columns={'Protein1_ID': 'Protein_ID', 'Protein1': 'Protein'}
    )

    # Select and rename Protein2 columns
    protein2_df = all_proteins[['Protein2_ID', 'Protein2']].rename(
        columns={'Protein2_ID': 'Protein_ID', 'Protein2': 'Protein'}
    )

    # Concatenate the two vertically
    all_protein_list = pd.concat([protein1_df, protein2_df], ignore_index=True)

    # Drop duplicates
    all_protein_list = all_protein_list.drop_duplicates()

    all_protein_list.to_csv('../data/huri_neg_unique_proteins.csv', index=False)

if __name__ == "__main__":
    generate_unique_protein_list()