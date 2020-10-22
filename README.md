# nas-gcn
Neural Architecture Search for Graph Message Passing Neural Network using DeepHyper.

## Testing on LCRC Bebop
Regularized evolution for QM9 dataset
```bash
srun -n 30 python -m tuster.system.bebop.run 'python -m deephyper.search.nas.regevo --evaluator ray --redis-address {redis_address} --problem nas_gcn.qm9.problem.Problem'
```

