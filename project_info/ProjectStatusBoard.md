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
    - MAE only for holdout points (?) --> YES!
  
  
  - Fiona
    - Research RECIST / up, down, fluctuate as means to select representative patients
        * I think we want to do RECIST evaluations to see if the models end point matches to the RECIST status of the actual results - seems to be a common, good thing to do in this sort of research
        * So if we can write a short script to do a runthrough of each patient and determine their RECIST status, we could have that stored as a variable in the data and not have to do it each time (since we're using the same data set every time, to save time when running)
        * Since this is a categorical thing (and it is binary of yes/no for if it ends up in the same category) I think a specific kind of chi-square family value called McNemar's test statistic would be the best thing for looking at the difference in prediction for the IDE vs ODE of each model (ie. exponential IDE vs exponential ODE, etc.)
        * There's a python library called 'statsmodels' that I believe has the method you need "from statsmodels.stats.contingency_tables import mcnemar"
        * The function takes in a table in the format of [a, b]
                                                         [c, d]
        where a+b+c+d will equal the total number of patients, and a = both models are correct, b = model 1 is correct, model 2 is wrong, c = model 1 incorrect, model 2 correct, d = both modles incorrect
        * It's then just doing a chi-square test with 1 degree of freedom to give you the p-value of a statistically significant difference in the model having the correct categorical RECIST end point
    
    - Research RMSE vs MAE for results, especially given normalization / data scale per patient --> Use MAE
        * If we are just doing one, I think we want to look at MAE. RMSE catastrophizes bigger errors, whereas MAE looks at just raw errors (absolute error) and it is a little less sensitive to the odd outlier, which is probably good
        * I was wrong before - RMSE has nothing to do with the r^2 value; we probably don't actually care about r^2 that much for this. RMSE is just MAE but you square the errors before you average them essentially, so bigger erros get really exploded up
    
    - Research normalization (full dataset vs patient max vs patient baseline vs none) --> MAX OF FULL DATASET
        * I say we keep normalization across the entire dataset (could go through in advance and do one run of the data since we aren't changing what we're using each run, to save time)
        * Essentially, it is identical to not doing any normalization, but it gives you a scale of the error in comparison to the largest tumor
        * So for RMSE and MAE, it would tell you that the error number that you're looking at is x% of the largest tumor volume in the dataset
            * You could definitely argue that you don't need to do this and just say what the biggest tumor is for a comparison, but this sort of makes it easier to look at
        * Regarding why we don't just normalize to each individual patient, this would lead to us disregarding errors that are actually more significant. For example, say patien A has tumor size of 50 and you approximated it as 100, versus patient B who had tumor size of 5 and you apprxoimated as 10. Normalizing by patient counts those errors as the same, when they should really be weighted differently, since an error in 50 is really more substantial than an error of 5, even with the relative scale of things. So I would say do not normalize patient by patient
        * If normalization across the whole thing is a giant pain, then functionally no normalization is really the same (as long as we give the max value or some range of values as a reference point for interpretation of the results) 
        * I would also say that we should think about putting the V_max for each patient maybe as a note on the graphs (for the ones we end up showing) - so that they can know that for additional information possibly
    
    - Research MAE/RMSE for holdout points or full curve
        * Only calculate MAE/RMSE for the holdout points (not the full curve); I think this makes sense from a standpoint of having a meaningful result. It also seems to be what they did in the paper when they describe experiment two as "estimating the predictive accuracy for the *excluded* data points"
    
    - Research if we should use negative time points
        * I think We should keep the negative time points. The paper says they use "all availble time points" for the experiments. Let's follow the same pathway.
        * We could *consider* doing another comparison where we do and do not use negative points
        * Could also *consider* doing something where we only do the differential evolution for the phi value (for the IDE) using positive time points, even if that means using less (i.e. if the first 3 time points are -14, 2, 8, then we woudl only use 2 and 8 for determining phi)
            * this is probably useless so maybe don't bother, just noting it here as a thought in case we somehow run out of other stuff to do lol
    
    - Why use SSE for differential evolution?
        * Sort of what we had guessed: it's fast and easy computationally, penalizes larger errors more to try to get the best fit you can, and also apparently it gives a cleaner input to the differential evolution process which could otherwise get screwy around zeros
        * I think we stick with it - don't see a reason not to tbh


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
- Wait for response from Laleh
  - Ping him soon (he asked for data, I told him it is WIP)
  - If still no response, email Heiko Enderling (also from that paper – a connection to one of my old math professors)
- Wait for data from Dr. Fenyo
  - Ping soon if nothing
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
