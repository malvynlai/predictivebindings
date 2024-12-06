import os
import sys
from random import sample
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    pos_neg_ratio = 3
    buildingblock_ratio = [16, 32, 48]
    random_seed = 1207
    batch_size = 100000
    num_batches = 2953
    total_data_path = os.path.join(data_path, 'total_data_new.parquet')

    if not os.path.exists(total_data_path):
        file_exists = False
        output_file = total_data_path
        parquet_file = pq.ParquetFile(os.path.join(data_path, 'train.parquet'))
        for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), total=num_batches, desc="Processing batches"):
            data_set = batch.to_pandas()

            data_set['molecule_smiles'] = data_set['molecule_smiles'].str.replace("\[Dy\]", "")
            reshape = data_set.pivot(index = 'molecule_smiles', columns = 'protein_name', values = 'binds')
            reshape = reshape.fillna(0)
            final_dataset = data_set[
                ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']
            ].merge(reshape, left_on='molecule_smiles', right_index=True, how='left')

            final_dataset.drop_duplicates(
                subset = ['molecule_smiles'],
                keep = 'first',
                inplace = True,
                ignore_index = True
            )
            # If the file exists, append the batch data to the existing file
            # calculate the sample score
            final_dataset['score'] = final_dataset['BRD4'] + final_dataset['HSA'] + final_dataset['sEH']
            numeric_columns = ['BRD4', 'HSA', 'sEH', 'score']
            for col in numeric_columns:
                if col in final_dataset:
                    final_dataset[col] = final_dataset[col].astype('int64')
            table = pa.Table.from_pandas(final_dataset) 

            if file_exists:
                writer.write_table(table)  # Write to the existing file
            else:
                # If the file doesn't exist, write the first batch to create the file
                pq.write_table(table, output_file)
                file_exists = True  # Set the flag to True after the first batch is written
                writer = pq.ParquetWriter(output_file, schema=table.schema)  # Initialize the writer
                
            # Optional explicit cleanup
            del data_set, reshape, final_dataset, table
            import gc
            gc.collect()
        if writer:
            writer.close()


    #         brd4_data = pd.read_csv(f'{data_path}/BRD4.csv')
    #         seh_data = pd.read_csv(f'{data_path}/sEH.csv')
    #         hsa_data = pd.read_csv(f'{data_path}/HSA.csv')
    #         brd4_data['protein_name'] = 'BRD4'
    #         brd4_data['Label'] = brd4_data['pChEMBL Value'].apply(lambda x: 0 if x < 5 else 1)
    #         seh_data['protein_name'] = 'sEH'
    #         seh_data['Label'] = seh_data['pChEMBL Value'].apply(lambda x: 0 if x < 5 else 1)
    #         hsa_data['protein_name'] = 'HSA'
    #         brd4_data.drop_duplicates(subset = ['Smiles'], inplace = True)
    #         seh_data.drop_duplicates(subset = ['Smiles'], inplace = True)
    #         hsa_data.drop_duplicates(subset = ['Smiles'], inplace = True)
    #         chembl_total = pd.concat(
    #             [
    #                 brd4_data[['Smiles', 'protein_name', 'Label']],
    #                 seh_data[['Smiles', 'protein_name', 'Label']],
    #                 hsa_data[['Smiles', 'protein_name', 'Label']],
    #             ],
    #             ignore_index = True
    #         ).pivot(index = 'Smiles', columns = 'protein_name', values = 'Label')
    #         chembl_total = chembl_total.fillna(0)
    #         chembl_total = chembl_total.reset_index()
    #         chembl_total.columns = ['molecule_smiles', 'BRD4', 'HSA', 'sEH']
    #         chembl_total['score'] = chembl_total['BRD4'] + chembl_total['HSA'] + chembl_total['sEH']
    #         chembl_total.to_parquet(f'{data_path}/chembl_extrnal_data.parquet')
    # else:
    #     final_dataset = pd.read_parquet(f'{data_path}/total_data.parquet')
    # # postive samples
    # positive_samples = final_dataset[final_dataset['score'] > 0]
    # number_of_positive_samples = len(positive_samples)
    # # negative samples
    # negative_samples = final_dataset[final_dataset['score'] == 0].sample(
    #     n = number_of_positive_samples * pos_neg_ratio, 
    #     random_state = random_seed
    # )
    # merged_samples = pd.concat([positive_samples, negative_samples], ignore_index = True).sample(
    #     frac = 1, 
    #     random_state = random_seed
    # )
    # # sample the valid and test set
    # bb1_positive = merged_samples['buildingblock1_smiles'].unique().tolist()
    # bb2_positive = merged_samples['buildingblock2_smiles'].unique().tolist()
    # bb3_positive = merged_samples['buildingblock3_smiles'].unique().tolist()
    # bb1_positive = sample(bb1_positive, k = len(bb1_positive))
    # bb2_positive = sample(bb2_positive, k = len(bb2_positive))
    # bb3_positive = sample(bb3_positive, k = len(bb3_positive))
    # bb1_test, bb1_train = bb1_positive[0:buildingblock_ratio[0]], bb1_positive[buildingblock_ratio[0]:]
    # bb2_test, bb2_train = bb2_positive[0:buildingblock_ratio[1]], bb2_positive[buildingblock_ratio[1]:]
    # bb3_test, bb3_train = bb3_positive[0:buildingblock_ratio[2]], bb3_positive[buildingblock_ratio[2]:]
    # train_set = merged_samples[
    #     (merged_samples['buildingblock1_smiles'].isin(bb1_train)) &
    #     (merged_samples['buildingblock2_smiles'].isin(bb2_train)) &
    #     (merged_samples['buildingblock3_smiles'].isin(bb3_train))
    # ]
    # print(f"NOTE: the {len(train_set)} samples in the training set.")
    # print(f"NOTE: {len(train_set[train_set['score'] > 1])} molecules have more than 1 targets.")
    # print(
    #     f"NOTE: {len(train_set[(train_set['BRD4'] == 1)&(train_set['score'] == 1)])} molecules target to BRD4.")
    # print(
    #     f"NOTE: {len(train_set[(train_set['HSA'] == 1)&(train_set['score'] == 1)])} molecules target to HSA.")
    # print(
    #     f"NOTE: {len(train_set[(train_set['sEH'] == 1)&(train_set['score'] == 1)])} molecules target to sEH.")
    # train_set.to_parquet(f'{output_path}/train.parquet')
    # test_set = merged_samples[
    #     (merged_samples['buildingblock1_smiles'].isin(bb1_test)) &
    #     (merged_samples['buildingblock2_smiles'].isin(bb2_test)) &
    #     (merged_samples['buildingblock3_smiles'].isin(bb3_test))
    # ]
    # print(f"NOTE: total {len(test_set)} samples in the test set.")
    # test_set.to_parquet(f'{output_path}/test.parquet')
    # other_set = merged_samples.drop(index = train_set.index.tolist() + test_set.index.tolist())
    # val_set = pd.concat([
    #     other_set[other_set['score'] > 1].sample(n = 256, random_state = random_seed), 
    #     other_set[
    #         (other_set['BRD4'] == 1)&(other_set['score'] == 1)
    #     ].sample(n = 256, random_state = random_seed),
    #     other_set[
    #         (other_set['HSA'] == 1)&(other_set['score'] == 1)
    #     ].sample(n = 512, random_state = random_seed),
    #     other_set[
    #         (other_set['sEH'] == 1)&(other_set['score'] == 1)
    #     ].sample(n = 256, random_state = random_seed),
    #     other_set[other_set['score'] == 0].sample(n = 3072, random_state = random_seed)
    # ])
    # print(f"NOTE: total {len(val_set)} samples in the validation set.")
    # val_set.to_parquet(f'{output_path}/val.parquet')
    # # other_set = other_set.drop(index = val_set.index.tolist())
    # # other_set.to_parquet(f'{output_path}/other.parquet')
    