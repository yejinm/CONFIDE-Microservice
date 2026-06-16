# Third-party subject systems (not redistributed)

This repository is an anonymous reproducibility package. The original subject systems used in experiments are **third-party projects** and are **not redistributed** here due to licensing and redistribution constraints.

To run the full pipeline on raw subject systems (beyond the provided processed snapshot), please download the following repositories and place them under this folder.

## Upstream repositories

- DayTrader (WASdev): https://github.com/WASdev/sample.daytrader7
- PlantsByWebSphere (WASdev): https://github.com/WASdev/sample.plantsbywebsphere
- AcmeAir: https://github.com/acmeair/acmeair
- jPetStore: https://github.com/KimJongSung/jPetStore

## Expected local directory layout

After cloning/downloading, your `data/raw/` directory should look like:

```
<data-repo-root>/
  data/
    raw/
      daytrader7/
      plantsbywebsphere/
      acmeair/
      jPetStore/
```

Notes:
- Folder names can be different if you prefer, but then you must update any scripts/configuration that assume these names.
- All scripts in this repo assume the **repository root** as the working directory and use **relative paths**.

## Quick clone commands (PowerShell)

From the repository root:

```powershell
cd data\raw

git clone https://github.com/WASdev/sample.daytrader7.git
git clone https://github.com/WASdev/sample.plantsbywebsphere.git
git clone https://github.com/acmeair/acmeair.git
git clone https://github.com/KimJongSung/jPetStore.git
```

## Using the provided snapshot (paper reproduction)

If you only want to reproduce the paper's main tables/figures, you do **not** need to download these raw subject systems.
Use the curated processed snapshot and run:

```powershell
.\scripts\reproduce_paper.ps1 -Tag paper_v1
```
