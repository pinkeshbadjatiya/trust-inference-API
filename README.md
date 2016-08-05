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

## References
(Add all the links to the papers)  
[1] `mTrust-datasets` : http://www.jiliang.xyz/trust.html  
[2] `MATRI-paper` : https://ptpb.pw/UIp1.pdf  
[3] `MATRI-report` : https://ptpb.pw/0LX2.pdf  

