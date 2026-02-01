# Project Objectives and Status

## To-Do:
- Marco Fiona Discussion 2/1/26
  - Marco
    - Fix Bertalanffy/Gompertz overflow errors
    - Drop experiment 1 (best-fit stuff)
      - Don't need n < 6 data anymore (can filter it out or just delete from csv)
    - Don't need treatment arm info
    - Graph outputs probs fine as is
    - Output graph ODE vs IDE per base model per patient
    - Perform paired t test for ODE vs IDE error per base model per patient
      - MAE/RMSE, potentially both, possibly one
    - See how rejecting t < 1 data affects fitting
      - Could help us determine if we need only data from one regime, or we can mix pre-treatment with treatment values
    - MAE only for holdout points (?)
  - Fiona
    - Research RECIST / up, down, fluctuate as means to select representative patients
    - Research RMSE vs MAE for results, especially given normalization / data scale per patient
    - Research normalization (full dataset vs patient max vs patient baseline vs none)
    - Research MAE/RMSE for holdout points or full curve


## Status:

### Technical:
- Wrote new and improved Laleh Code that ...
  - Intakes data from CSV file (sample is in GitHub)
  - Creates 12 models
    - 6 continuous, 6 impulsive, one of each for every base ODE
  - Guesses initial DE parameters using differential evolution to optimize SSE
  - Uses scipy curve fitting with 'LSODA' (Laleh used 'trf') algorithm to fit parameters to data given guesses
  - Performs the two experiments (curve fitting and holdout prediction) and records MAE/RMSE for all 12 models
  - Outputs graphical depiction of model fit
  
### Logistical:
- Wait for response from Dr. Kather (corresponding author from Laleh paper)
  - If no response, ping Dr. Becker to send a follow-up
  - If still no response, email Heiko Enderling (also from that paper – a connection to one of my old math professors)
- Wait for data from Dr. Becker
  - Ping if nothing in the next few weeks
  - Anonymize and deidentify once received before sending to the rest of the team / putting it on GitHub

### Writing:
- Introduction
  - Literature Review 
    - Work on compiling sources at this point, categorizing them for later review 
    - Try to categorize them into related buckets of sorts 
  - Methods 
    - Continue updating as we continue to change things
  - Results, Discussion, Conclusion
    - Use generated graphs
    - Table for all data
    - Continue updating as we receive the real data and implement the experiments
