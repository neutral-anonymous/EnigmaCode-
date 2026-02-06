

### Obfuscation Techniques (5 methods)

1. **Topology Trimmer** - Reduces problem size by fixing variables
2. **Structure Camouflage** - Adds decoy nodes/edges to disguise graph topology  
3. **Regularizer** - Makes all nodes uniform degree (optional)
4. **Value Guard** - Rescales and flips coefficients
5. **Ising Sanitizer** - Cleans and normalizes the final model

### Graph Types Supported

- **ER** (Erdős–Rényi): Random graphs
- **BA** (Barabási–Albert): Scale-free networks
- **REG** (d-Regular): Uniform degree graphs  
- **SK** (Sherrington-Kirkpatrick): Fully connected graphs

## Code Structure

```
core_utils.py              # Ising/QUBO conversion, graph utilities
obfuscation_techniques.py  # All 5 obfuscation methods
enigma.py                  # Main obfuscation pipeline
experiment_runner.py       # Dataset generation (parallel/sequential)
```

## Installation

```bash
pip install numpy networkx joblib
```

