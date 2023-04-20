import os
import numpy as np
import pickle
import py3Dmol
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax

# setup which model params to use
# note: for this demo, we only use model 1, for all five models uncomments the others!
model_runners = {}
models = ["model_1"] #,"model_2","model_3","model_4","model_5"]
for model_name in models:
  model_config = config.model_config(model_name)
  model_config.data.eval.num_ensemble = 1
  model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/alphafold-data")
  model_runner = model.RunModel(model_config, model_params)
  model_runners[model_name] = model_runner

import numpy as np

def mk_mock_template(query_sequence):
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features

def predict_structure(prefix, feature_dict, model_runners, do_relax=True, random_seed=0):  
  """Predicts structure using AlphaFold for the given sequence."""

  # Run the models.
  plddts = {}
  for model_name, model_runner in model_runners.items():
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
    prediction_result = model_runner.predict(processed_feature_dict)
    unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
    unrelaxed_pdb_path = f'{prefix}_unrelaxed_3_{model_name}.pdb'
    plddts[model_name] = prediction_result['plddt']

    print(f"{model_name} {plddts[model_name].mean()}")

    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(protein.to_pdb(unrelaxed_protein))

    if do_relax:
      # Relax the prediction.
      amber_relaxer = relax.AmberRelaxation(max_iterations=0,tolerance=2.39,
                                            stiffness=10.0,exclude_residues=[],
                                            max_outer_iterations=20)      
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      relaxed_pdb_path = f'{prefix}_relaxed_3_{model_name}.pdb'
      with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_str)

  return plddts

query_sequence = ""

%%time
feature_dict = {
    **pipeline.make_sequence_features(sequence=query_sequence,
                                      description="none",
                                      num_res=len(query_sequence)),
    **pipeline.make_msa_features(msas=[[query_sequence]],
                                 deletion_matrices=[[[0]*len(query_sequence)]]),
    **mk_mock_template(query_sequence)
}
plddts = predict_structure("test",feature_dict,model_runners)

# confidence per position
plt.figure(dpi=100)
for model,value in plddts.items():
  plt.plot(value,label=model)
plt.legend()
plt.ylim(0,100)
plt.ylabel("predicted LDDT")
plt.xlabel("positions")
plt.show()

p = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
p.addModel(open("test_relaxed_model_1.pdb",'r').read(),'pdb')
p.setStyle({'cartoon': {'color':'spectrum'}})
p.zoomTo()
p.show()

p = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
p.addModel(open("test_relaxed_model_1.pdb",'r').read(),'pdb')
p.setStyle({'cartoon': {'color':'spectrum'},'stick':{}})
p.zoomTo()
p.show()

