# Date: 2021-08-20 11:24:00
#CUDA_VISIBLE_DEVICES=1

for model in 'LSTM' #'Transformer' 'RNN_RELU' 'RNN_TANH' 'LSTM' 'GRU' 
do
    for hidden_sizes in 800
    do
        for dropouts in 0.5 
        do
            for learning_rates in 5
            do
                for batch_sizes in 32
                do
                    OUTPUT_DIR=model/new_test/${model}/hidden${hidden_sizes}-dropout${dropouts}-lr${learning_rates}-batch${batch_sizes}

                    if [ -d "$OUTPUT_DIR" ]; then
                      OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
                    fi

                    mkdir -p ${OUTPUT_DIR}

                    python main.py --data data/gigaspeech \
                        --cuda \
                        --epochs 40 \
                        --model "${model}" \
                        --emsize "${hidden_sizes}" \
                        --nhid "${hidden_sizes}" \
                        --dropout "${dropouts}" \
                        --lr "${learning_rates}" \
                        --batch_size "${batch_sizes}" \
                        2>&1 | tee ${OUTPUT_DIR}/log.txt
                done
            done
        done
    done
done