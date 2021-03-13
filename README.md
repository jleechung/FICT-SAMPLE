## Sample code for FICT-FISH Iterative Cell Typing

### Installation
```bash
git clone https://github.com/haotianteng/FICT-SAMPLE.git
cd FICT-SAMPLE
git clone https://github.com/haotianteng/FICT
git clone https://github.com/haotianteng/GECT
export PYTHONPATH="$(pwd)/FICT/:$(pwd)/GECT/:$PYTHONPATH"
```
**Install dependency**
```bash
conda activate YOUR_ENVIRONMENT
pip install -r requirements.txt
```

### Preparing datasets
**osmFISH**: datasets/osmFISH_SScortex_mouse_all_cells.loom  
**seqFISH**: datasets/seqFISH/{fcortex.coordinates, fcortex.expression, fcortex.genes}  
**MERFISH**: datasets/aau5324_Moffitt_Table-S7.xlsx  
*Download full MERFISH data matrix *Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv* from [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248).
Unzip it to the datasets folder.

```bash
python prepare_osmFISH.py
python prepare_seqFISH.py
python prepare_MERFISH.py
```

### Simulation
```bash
python sim_from_real.py
python dummy_train.py --prefix simulation/addictive
python dummy_train.py --prefix simulation/exclusive
python dummy_train.py --prefix simulation/stripe
python dummy_train.py --prefix simulation/real
```

### Cross Validation
```bash
EMBEDDING_SIZE=20
python GECT/gect/gect_train_embedding.py -i Benchmark/MERFISH/data/1 -o Benchmark/MERFISH/ -m embedding --embedding-size $EMBEDDING_SIZE -b 200 --epoches 10 -t 4e-3
DATA=Benchmark/MERFISH/data
python FICT/cross_validation.py -i $DATA/1,$DATA/2,$DATA/3 -o Benchmark/MERFISH/FICT_CV/ --renew_round 40 --n_class 7 -d $EMBEDDING_SIZE --spatio_factor 0.1 --mode multi --reduced_method Embedding --embedding_file Benchmark/MERFISH/embedding 
```
For direct cell type assignment without cross validation just run each data loader separately, e.g.  
```bash
python FICT/cross_validation.py -i $DATA/1 -o Benchmark/Merfish/FICT_OUT1/ --renew_round 40 --n_class 7 -d 20 --spatio_factor 0.1 --mode multi --reduced_method Embedding --embedding_file Benchmark/MERFISH/embedding 
python FICT/cross_validation.py -i $DATA/2 -o Benchmark/Merfish/FICT_OUT2/ --renew_round 40 --n_class 7 -d 20 --spatio_factor 0.1 --mode multi --reduced_method Embedding --embedding_file Benchmark/MERFISH/embedding 
python FICT/cross_validation.py -i $DATA/3 -o Benchmark/Merfish/FICT_OUT3/ --renew_round 40 --n_class 7 -d 20 --spatio_factor 0.1 --mode multi --reduced_method Embedding --embedding_file Benchmark/MERFISH/embedding 

```

