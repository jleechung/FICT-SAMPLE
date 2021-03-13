## Sample code for FICT-FISH Iterative Cell Typing

### Installation
```bash
git clone https://github.com/haotianteng/FICT
git clone https://github.com/haotianteng/GECT
export PYTHONPATH="$(pwd)/FICT/:$(pwd)/GECT/:$PYTHONPATH"
pip install -r requirements.txt
```

### Download datasets
**osmFISH**: *osmFISH_SScortex_mouse_all_cells.loom* osmFISH dataset is already being downloaded at datasets folder.  
**seqFISH**: TODO  
### Simulation
```bash
python sim_from_real.py
python dummy_train.py --prefix simulation/addictive
python dummy_train.py --prefix simulation/exclusive
python dummy_train.py --prefix simulation/stripe
python dummy_train.py --prefix simulation/real
```

### Cross Validation
TODO
