declare -a envs=("quadcopter" "pendulum" "cartpole" "self_driving")
# declare -a envs=("car_platoon_4")

mkdir logs

# for num in {1..10}
# do
for env in "${envs[@]}"
do
    echo "Running $env"
    python shield.py --env "$env" --rounds 1000 2>&1 | tee logs/"$env"_RQ23_new.log
done
# done