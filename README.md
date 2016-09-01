# trust-inference-API
An opensource API exposing multiple algorithms for infering trust in a network

## Instructions
### sTrust
- **USAGE**: `python api.py sTrust`
- **DATASETS**: It requires 3 files,
  - **epinion_trust_with_timestamp.mat**: It contains a dictionary with key as "trust" while the value is a 2-d array with each row of the format `[i, j, t]` representing, user i trust user j at time t (time represents the time when the realtionship established).
  - **trust.mat**: It contains the dictionary with key as "trust" while the value represents a 2-d array with each row of the format `[i, j]` representing, user i trusts user j.
  - **rating_with_timestamp.mat**: It contains a dictionary with key as "rating_with_timestamp" while the value represents a 2-d array with each row of the format `[1, 2, 3, 4, 5, 6]` representing, user 1 gives a rating of 4 to the product 2 from the category 3. The helpfulness of this rating is 5 in the time stamp 6. 
  - The dataset used is Epinions dataset which was released in the month of May, 2011. [1]


### hTrust
- **USAGE**: `python api.py hTrust`
- **DATASETS** : It also requires the same dataset as used by `sTrust`

### MATRI
- **USAGE**: `python api.py matri`
- **DATASETS** : It requires a **.dot** file in the directory "dataset" which contains the information about the network graph in the format:
  ```
  strict digraph  {
          eivind -> wsanchez       [level=Master];
          eivind -> nbm    [level=Journeyer];
          eivind -> cg     [level=Journeyer];
          eivind -> jmock  [level=Journeyer];
          eivind -> unfurl         [level=Journeyer];
          eivind -> quiet1         [level=Apprentice];
          eivind -> cynick         [level=Apprentice];
          eivind -> kkenn  [level=Journeyer];
          ....
  
  ```
- The type of matrix-factorization used can be changed. The accuracy of the resutl depends a lot on the accuracy of factorization. By default, there are 4 types of factorizations saved in the directory "factorization", namely,
  - Alternating Factorization
  - Gradient Descent
  - state-of-the-art factorization provided by `pymf` library
  - state-of-the-art factorization provided by `sklearn` library  
- The `globs.py` contains the valious configurable parameters used in the code, like, Max no of factorization iterations, Path to save the computed matrices, Total no of bias factors, latent factors, and the regularization parameter for updation.


### aeTrust
#### Trust inference using Autoencoders
- **USAGE**: `python api.py aeTrust`
- **DATASETS** : data.mat: Advagato trust dataset
- To train on a custom dataset, replace data.mat in the 'data' folder in 'aeTrust' with your own dataset formatted in MATLAB compatible format - '.mat'.

### mTrust
- **USAGE**: `python api.py mTrust`
- **DATASETS**: It requires 2 files,
  - **trust.mat**: It contains the dictionary with key as "trust" while the value represents a 2-d array with each row of the format `[i, j]` representing, user i trusts user j.
  - **rating_with_timestamp.mat**: It contains a dictionary with key as "rating_with_timestamp" while the value represents a 2-d array with each row of the format `[1, 2, 3, 4, 5, 6]` representing, user 1 gives a rating of 4 to the product 2 from the category 3. The helpfulness of this rating is 5 in the time stamp 6. 
  - The dataset used is Epinions dataset which was released in the month of May, 2011. [1]


## References
(Add all the links to the papers)  
[1] `mTrust-datasets` : http://www.jiliang.xyz/trust.html  
[2] `MATRI-paper` : https://ptpb.pw/UIp1.pdf  
[3] `MATRI-report` : https://ptpb.pw/0LX2.pdf  
[4] `AutoRec: Autoencoders Meet Collaborative Filtering`: http://dl.acm.org/citation.cfm?id=2742726 

