ds=$1
runfile=$2

# both cannot be empty
if [ -z "$ds" ] || [ -z "$runfile" ]; then
    echo "Error: dataset and runfile cannot be empty"
    exit 1
fi

# assert $ds is one of 2019, 2020
if [ "$ds" != "2019" ] && [ "$ds" != "2020" ]; then
    echo "Error: dataset must be one of 2019, 2020"
    exit 1
fi


# assert $runfile is a valid file
if [ ! -f "$runfile" ]; then
    echo "Error: runfile must be a valid file"
    exit 1
fi



trec_eval ../../../trec-dl-$ds.qrels $runfile -m ndcg_cut.10
