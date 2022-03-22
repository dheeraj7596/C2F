gpu=$1
data_dir=$2
model_dir=$3
echo "ITERATION-1"
echo "Running GPT2 iteration 1"
CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_ce_hinge.py ${data_dir} ${model_dir} 1
echo "Generating Arts Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} arts 500
echo "Training Arts BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 1 arts
echo "Generating Politics Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} politics 250
echo "Training Politics BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 1 politics
echo "Generating Science Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} science 90
echo "Training Science BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 1 science
echo "Generating Business Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} business 500
echo "Training Business BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 1 business
echo "Generating Sports Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} sports 2000
echo "Training Sports BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 1 sports
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE-Hinge iteration 2"
CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_ce_hinge.py ${data_dir} ${model_dir} 2
echo "Generating Arts Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} arts 500
echo "Training Arts BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 2 arts
echo "Generating Politics Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} politics 250
echo "Training Politics BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 2 politics
echo "Generating Science Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} science 90
echo "Training Science BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 2 science
echo "Generating Business Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} business 500
echo "Training Business BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 2 business
echo "Generating Sports Data.."
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_data_ce_hinge.py ${data_dir} ${model_dir} sports 2000
echo "Training Sports BERT.."
CUDA_VISIBLE_DEVICES=${gpu} python3 bert_train.py ${data_dir} ${model_dir} 2 sports
echo "***********************************************************************"
CUDA_VISIBLE_DEVICES=${gpu} python3 evaluate.py ${data_dir}
