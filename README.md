# nas-gcn
Neural Architecture Search for Graph Message Passing Neural Network using DeepHyper.

## Install

```bash
conda install -c conda-forge rdkit
cd nas-gcn/
pip install -e.
```

## Testing on LCRC Bebop

Regularized evolution for QM9 dataset

```bash
srun -n 30 python -m tuster.system.bebop.run 'python -m deephyper.search.nas.regevo --evaluator ray --redis-address {redis_address} --problem nas_gcn.qm9.problem.Problem'
```

ThetaGPU

```bash
deephyper nas random --evaluator ray --ray-address auto --problem nas_gcn.esol.problem.Problem --max-evals 10 --num-cpus-per-task 1 --num-gpus-per-task 1
```
