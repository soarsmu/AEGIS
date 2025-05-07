declare -a envs=("quadcopter" "pendulum" "cartpole" "self_driving")

mkdir logs

for env in "${envs[@]}"
do
    echo "Running $env"
    python3 shield.py --env "$env" --rounds 1000 2>&1 | tee logs/"$env"_RQ23_new.log
done
