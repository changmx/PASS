# Example 1: Particle Distribution Generation



## Description

The simulation begins by generating various particle distributions through the injection command in the sequence.



## How to Run

```bash	
cd PASS\example\01_generate_distribution
python run.py --beam0=./beam0.json
```



## How to modify the particle distribution

1. Open the input file: `PASS\example\01_generate_distribution\beam0.json`.

2. Change the value of `"Transverse dist"` or `"Longitudinal dist"`.

Currently, the transverse distribution supports: `gaussian`, `kv`, and `uniform`. The longitudinal distribution supports: `gaussian`, `coasting`, `matchz` and `matchdp`.