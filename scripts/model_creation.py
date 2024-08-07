# Import necessary libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

# Query ChEMBL Database
target = new_client.target
target_query = target.search('acetylcholinesterase')
targets = pd.DataFrame.from_dict(target_query)
selected_target = targets.target_chembl_id[0]

# Query activity data
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
data = pd.DataFrame.from_dict(res)

# Data Cleaning and Processing
data_clean = data.dropna(subset=['standard_value', 'canonical_smiles']).drop_duplicates('canonical_smiles')

def clean_smiles(smiles):
    cleaned_smiles = []
    for i in smiles:
        cpd = str(i).split('.')
        cpd_longest = max(cpd, key=len)
        cleaned_smiles.append(cpd_longest)
    return cleaned_smiles

data_clean['canonical_smiles'] = clean_smiles(data_clean['canonical_smiles'])

# Bioactivity classification
def classify_bioactivity(df):
    thresholds = {'inactive': 10000, 'active': 1000}
    df['class'] = pd.cut(df['standard_value'],
                         bins=[-np.inf, thresholds['active'], thresholds['inactive'], np.inf],
                         labels=['active', 'intermediate', 'inactive'])
    return df

data_clean = classify_bioactivity(data_clean)

# Compute Lipinski descriptors
def lipinski(smiles):
    moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
    desc_list = [[
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol)
    ] for mol in moldata]
    return pd.DataFrame(desc_list, columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])

descriptors = lipinski(data_clean['canonical_smiles'])

# Combine data and descriptors
data_combined = pd.concat([data_clean[['molecule_chembl_id', 'class']], descriptors], axis=1)

# Normalize values and compute pIC50
def normalize_and_compute_pIC50(df):
    df['standard_value_norm'] = df['standard_value'].clip(upper=100000000)
    df['pIC50'] = -np.log10(df['standard_value_norm'] * 1e-9)
    return df.drop(columns=['standard_value', 'standard_value_norm'])

data_final = normalize_and_compute_pIC50(data_combined)

# Prepare dataset for modeling
df_final = data_final[['canonical_smiles', 'molecule_chembl_id', 'pIC50']].join(descriptors)
df_final.to_csv('acetylcholinesterase_final_data.csv', index=False)

# Machine Learning Model
X = df_final.drop(columns='pIC50')
Y = df_final['pIC50']

# Feature selection
selection_fp = VarianceThreshold(threshold=(.8 * (1 -.8)))
X_selected = selection_fp.fit_transform(X)

# Split data and train model
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)

# Evaluate model
r2_score = model.score(X_test, Y_test)
print(f'R^2 Score: {r2_score}')

# Predict and display predictions
Y_pred = model.predict(X_test)
print(Y_pred)
