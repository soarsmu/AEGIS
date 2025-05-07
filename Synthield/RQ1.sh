declare -a envs=("pendulum" "cartpole" "car_platoon_4" "self_driving")

mkdir logs

# for num in {1..10}
# do
for env in "${envs[@]}"
do
    python no_shield.py --env "$env" 2>&1 | tee logs/"$env"_RQ1_"$num".log
done
# done