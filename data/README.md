# Data

The two datasets are public but **not redistributed here**. Download them from
Mendeley Data and place the files where the pipeline can find them (the working
directory, `/content` on Colab, or a folder pointed to by `RIDEGUARD_DATA_DIR`).

1. **Behavioural dataset** — M. M. H. Matin, *Motorbike Accident Severity Analysis
   Dataset*, Mendeley Data, 2025. doi: 10.17632/rm8t8kmhpj
   Expected filename: `MotorBike_Accident.xlsx`

2. **Dhaka crash dataset** — F. Nur and A. Sahin Afridi, *Bangladesh Road Traffic
   Accident Dataset (2007-2024): Multi-Source Integration from ARI, BRTA, DMP, MP
   Unit*, Mendeley Data, 2025. doi: 10.17632/p2kgkb5cd2
   Expected (cleaned) filename: `Accident_Dataset_Dhaka.xlsx`
   (numeric Latitude/Longitude split out, derived Severity_Category/Severity_Score, Year, etc.)

Dhaka coordinates are obtained by geocoding the free-text location field with the
Google Maps Geocoding API; ~99.9% of points fall within metropolitan Dhaka.
