# DCAR


# To generate the data. 


# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5



To train the model using DCAR, we need to use the following command.

python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml
