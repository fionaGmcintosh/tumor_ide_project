# Project Objectives and Status

## To-Do:

### Technical:
- Refine / Recreate the code
  - Intake data from CSV file (sample is in GitHub)
  - Create 12 models
    - 6 continuous, 6 impulsive as described in Plan doc
  - Differential evolution to get initial guess parameters
  - Non-linear least squares loss function using the “scipy” package in Python with the trf algorithm to pick the parameters afterward based on the datapoints given
  - Perform the two experiments and analyses as detailed in the plan doc for all 12 models
  - Graphical depiction of model fit 
    - Possibly depict a sample of the model on one particular patient dataset, then a graphical depiction of the MSE, MAE, and RMSE (I think for now we should use all 3 and decide later if we want to focus it down)
    - Recalling also that for Experiment 2, the errors should only be based on the predicted points, not those that were fed to generate parameter values 
- Discuss use of AI as possible additional point of evaluation for picking parameter values
  - Possible hybrid of utilizing previous data for similar conditions as well as using the given patient datapoints for patient-specific predictions with some kind of weighting? 
  - Possible alternatives – need to discuss and flesh this out more
  
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

    
## Progress / Status Notes:

### Technical:
- Refine / Recreate the code
  - Currently, there is purely AI generated code
- Discuss use of AI as possible additional point of evaluation for picking parameter values

### Logistical:
- Wait for response from Dr. Kather (corresponding author from Laleh paper)
  - Waiting
- Wait for data from Dr. Becker
  - Waiting

### Writing:
- Introduction
  - WIP
- Literature Review
  - Doc file uploaded to GitHub with some initial sources, continuing to add
- Methods
  - WIP; updating as needed
- Results, Discussion, Conclusion
  - Cannot yet complete