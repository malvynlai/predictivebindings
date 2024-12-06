import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


output_path = '/home/cs229/subset'
data_path = '/home/cs229/leash-BELKA/total_data_new.parquet'

def split_data_subsets(data_path, output_path):
    data = pq.ParquetFile(data_path)
    batch_size = 100000
    for file_num in range(1):  
        rows_collected = 0
        output_file = f'{output_path}/train{file_num}.parquet'
        writer = None
        try:
            for batch in tqdm(data.iter_batches(batch_size=batch_size), desc=f'Creating file {file_num}'):
                batch_df = batch.to_pandas()
                batch_df = batch_df.sample(frac=1).reset_index(drop=True)
                rows_needed = 20000000 - rows_collected
                if len(batch_df) > rows_needed:
                    batch_df = batch_df.iloc[:rows_needed]
                table = pa.Table.from_pandas(batch_df)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, schema=table.schema)
                writer.write_table(table)
                rows_collected += len(batch_df)
                if rows_collected >= 20000000:
                    break
        finally:
            if writer is not None:
                writer.close()
                
split_data_subsets(data_path, output_path)