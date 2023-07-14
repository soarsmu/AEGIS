declare -a envs=("quadcopter" "pendulum" "cartpole")

mkdir logs

for num in {1..10}
do
    for env in "${envs[@]}"
    do
        python shield.py --env "$envs" 2>&1 | tee logs/"$env"_RQ23_"$num".log
    done
done